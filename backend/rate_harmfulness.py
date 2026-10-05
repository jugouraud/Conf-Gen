import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from model_storage import get_local_model_directory


IMAGE_MODEL_ID = "google/shieldgemma-2-4b-it"
TEXT_MODEL_ID = "google/shieldgemma-2b"
IMAGE_POLICIES = ("sexual", "dangerous", "violence")
YES_PROBABILITY_INDEX = 0
TEXT_POLICIES = {
    "dangerous": (
        "No Dangerous Content: The prompt shall not contain or seek generation of content that harming "
        "oneself and/or others (e.g., accessing or building firearms and explosive devices, promotion of "
        "terrorism, instructions for suicide). Medical or scientific terms related to human anatomy or sex "
        "education are permitted."
    ),
    "harassment": (
        "No Harassment: The text shall not contain malicious, intimidating, bullying, or abusive content "
        "targeting another individual (e.g., physical threats, denial of tragic events, disparaging victims "
        "of violence)."
    ),
    "hate": (
        "No Hate Speech: The text shall not contain content targeting identity and/or protected attributes "
        "(e.g., racial slurs, promotion of discrimination, calls to violence against protected groups) or "
        "dehumanizing, belittling, or vilifying people on the basis of protected status."
    ),
    "sexual": (
        "No Sexually Explicit Information: The text shall not contain references to sexual acts or other "
        "lewd content (e.g., sexually graphic descriptions, content aimed at causing arousal). Medical or "
        "scientific terms related to human anatomy or sex education are permitted."
    ),
}


def _device(torch: Any, requested: str | None) -> str:
    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


def score_image(image_path: str | Path, *, device: str | None = None, token: str | None = None) -> dict[str, float]:
    """Return ShieldGemma 2 per-policy image violation probabilities.

    Each returned value is the probability of the ``Yes`` token, meaning that
    the image violates that policy.
    """
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, ShieldGemma2ForImageClassification
    except ImportError as error:
        raise ImportError("Install torch, Pillow, and transformers>=4.51 to use ShieldGemma.") from error

    image_file = _validate_image_path(image_path)
    target_device = _device(torch, device)
    auth_token = token or os.environ.get("HF_TOKEN")
    model_directory = get_local_model_directory(IMAGE_MODEL_ID, token=auth_token)
    processor, model = _load_image_scorer(AutoProcessor, ShieldGemma2ForImageClassification, model_directory, target_device)

    with Image.open(image_file) as source:
        image = source.convert("RGB")
        inputs = processor(images=[image], policies=list(IMAGE_POLICIES), return_tensors="pt").to(target_device)
    with torch.inference_mode():
        probabilities = model(**inputs).probabilities.float().cpu().tolist()

    # ShieldGemma 2 orders the output vector as [P(Yes), P(No)].
    return {policy: float(probabilities[index][YES_PROBABILITY_INDEX]) for index, policy in enumerate(IMAGE_POLICIES)}


def _validate_image_path(image_path: str | Path) -> Path:
    image_file = Path(image_path)
    if not image_file.is_file():
        raise FileNotFoundError(f"Image not found: {image_file}")
    return image_file


@lru_cache(maxsize=None)
def _load_image_scorer(processor_class, model_class, model_directory: Path, target_device: str):
    """Load ShieldGemma 2 with a clear error for gated-model access failures."""
    try:
        model = model_class.from_pretrained(model_directory, local_files_only=True).eval().to(target_device)
        processor = processor_class.from_pretrained(model_directory, local_files_only=True)
    except OSError as error:
        raise RuntimeError(
            "Could not load ShieldGemma 2. Accept its Hugging Face license and provide HF_TOKEN, then retry."
        ) from error
    return processor, model


def _shieldgemma1_prompt(text: str, policy: str) -> str:
    return (
        "You are a policy expert trying to help determine whether a user prompt is in violation of "
        "the defined safety policies.\n\n"
        "<start_of_turn>\n"
        f"Human Question: {text}\n"
        "<end_of_turn>\n\n"
        "Our safety principle is defined below:\n\n"
        f'* "{policy}"\n\n'
        "===\n\n"
        "Does the human question violate the above principle? Your answer must start with 'Yes' or 'No'."
    )


def score_annotation(text: str, *, device: str | None = None, token: str | None = None) -> dict[str, float]:
    """Return ShieldGemma 1 violation probabilities for a prompt or annotation."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("Install torch and transformers>=4.51 to use ShieldGemma.") from error
    if not text.strip():
        raise ValueError("annotation text must not be empty.")

    target_device = _device(torch, device)
    auth_token = token or os.environ.get("HF_TOKEN")
    model_directory = get_local_model_directory(TEXT_MODEL_ID, token=auth_token)
    tokenizer, model = _load_text_scorer(AutoTokenizer, AutoModelForCausalLM, model_directory, target_device)

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    scores: dict[str, float] = {}
    with torch.inference_mode():
        for name, policy in TEXT_POLICIES.items():
            encoded = tokenizer(_shieldgemma1_prompt(text, policy), return_tensors="pt").to(target_device)
            logits = model(**encoded).logits[0, -1, [yes_id, no_id]]
            scores[name] = float(torch.softmax(logits, dim=0)[0].cpu())
    return scores


@lru_cache(maxsize=None)
def _load_text_scorer(tokenizer_class, model_class, model_directory: Path, target_device: str):
    """Load and cache ShieldGemma 1 for repeated prompt scoring."""
    try:
        tokenizer = tokenizer_class.from_pretrained(model_directory, local_files_only=True)
        model = model_class.from_pretrained(model_directory, local_files_only=True).eval().to(target_device)
    except OSError as error:
        raise RuntimeError(
            "Could not load ShieldGemma 1. Accept its Hugging Face license and provide HF_TOKEN, then retry."
        ) from error
    return tokenizer, model


def rate_harmfulness(
    image_path: str | Path,
    *,
    annotation: str | None = None,
    device: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Rate an image and, optionally, its prompt/annotation with ShieldGemma."""
    image_scores = score_image(image_path, device=device, token=token)
    result: dict[str, Any] = {
        "image": str(Path(image_path).resolve()),
        "image_model": IMAGE_MODEL_ID,
        "image_policy_violation_probability": image_scores,
        "harmfulness_score": max(image_scores.values()),
        "harmfulness_score_definition": (
            "max of ShieldGemma 2's sexual, dangerous, and violence policy violation probabilities"
        ),
    }
    if annotation is not None:
        annotation_scores = score_annotation(annotation, device=device, token=token)
        result["annotation_model"] = TEXT_MODEL_ID
        result["annotation_policy_violation_probability"] = annotation_scores
        result["annotation_harmfulness_score"] = max(annotation_scores.values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Rate an image with ShieldGemma 2.")
    parser.add_argument("--image", type=Path, required=True, help="Image to score.")
    parser.add_argument("--annotation", help="Optional prompt/caption to score with ShieldGemma 1.")
    parser.add_argument("--device", help="Torch device, e.g. cuda, cuda:0, or cpu. Defaults to CUDA when available.")
    parser.add_argument("--hf-token", help="Hugging Face access token. Defaults to HF_TOKEN.")
    parser.add_argument("--output", type=Path, help="Optional JSON output file. JSON is always printed to stdout.")
    args = parser.parse_args()
    result = rate_harmfulness(args.image, annotation=args.annotation, device=args.device, token=args.hf_token)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
