import argparse
import json
from functools import cache
from pathlib import Path

import numpy as np

from backend.storage.models import get_local_model_directory

DEFAULT_MODEL_ID = "openai/clip-vit-large-patch14"
EXPECTED_CONTEXT_SHAPE = (77, 768)
EXPECTED_IMAGE_EMBEDDING_SIZE = EXPECTED_CONTEXT_SHAPE[1]


def extract_inner_embeddings(
    prompt: str,
    image_path: str | Path,
    output_dir: str | Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
) -> dict[str, Path]:
    """Write the CLIP text context and image embedding for a prompt-image pair.

    The text context is the final hidden state before CLIP's text projection,
    which is the tensor used as the text condition by the U-ViT protocol that
    VeCoR references. It has one row for each of CLIP's 77 positions, including
    start, end, and padding tokens.
    """
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise ImportError(
            "This script requires torch, Pillow, and transformers. "
            "Install the project dependencies with `uv sync --group dev`."
        ) from error

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty.")

    image_file = _validate_image_path(image_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_directory = get_local_model_directory(model_id)
    processor, model = _load_clip_model(CLIPProcessor, CLIPModel, model_directory, target_device)
    context_array, image_array = _encode_prompt_image_pair(
        torch, Image, processor, model, prompt, image_file, target_device
    )
    _validate_embedding_shapes(context_array, image_array)
    return _write_embedding_files(destination, prompt, image_file, model_id, context_array, image_array)


def _validate_image_path(image_path: str | Path) -> Path:
    image_file = Path(image_path)
    if not image_file.is_file():
        raise FileNotFoundError(f"Image not found: {image_file}")
    return image_file


@cache
def _load_clip_model(processor_class, model_class, model_directory: Path, target_device):
    """Load and cache the CLIP processor/model for a model-device pair."""
    processor = processor_class.from_pretrained(model_directory, local_files_only=True)
    model = model_class.from_pretrained(model_directory, local_files_only=True).to(target_device).eval()
    return processor, model


def _encode_prompt_image_pair(torch, image_class, processor, model, prompt: str, image_file: Path, target_device):
    """Encode a prompt and RGB image into NumPy-compatible CLIP tensors."""
    with image_class.open(image_file) as opened_image:
        inputs = processor(
            text=[prompt], images=opened_image.convert("RGB"), padding="max_length", truncation=True,
            max_length=EXPECTED_CONTEXT_SHAPE[0], return_tensors="pt",
        )
    inputs = {name: value.to(target_device) for name, value in inputs.items()}
    with torch.inference_mode():
        text_context = model.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).last_hidden_state
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        image_embedding = _get_projected_image_embedding(image_features)
    return text_context.squeeze(0).float().cpu().numpy(), image_embedding.squeeze(0).float().cpu().numpy()


def _get_projected_image_embedding(image_features):
    """Return the projected image tensor across supported Transformers versions."""
    pooler_output = getattr(image_features, "pooler_output", None)
    return pooler_output if pooler_output is not None else image_features


def _validate_embedding_shapes(context_array: np.ndarray, image_array: np.ndarray) -> None:
    if tuple(context_array.shape) != EXPECTED_CONTEXT_SHAPE:
        raise RuntimeError(
            f"Expected CLIP context shape {EXPECTED_CONTEXT_SHAPE}, got {tuple(context_array.shape)}. "
            "Use a model with a 77-token, 768-wide text tower."
        )
    if tuple(image_array.shape) != (EXPECTED_IMAGE_EMBEDDING_SIZE,):
        raise RuntimeError(
            f"Expected a {EXPECTED_IMAGE_EMBEDDING_SIZE}-D projected CLIP image embedding, "
            f"got {tuple(image_array.shape)}."
        )


def _write_embedding_files(
    destination: Path, prompt: str, image_file: Path, model_id: str, context_array: np.ndarray, image_array: np.ndarray
) -> dict[str, Path]:
    """Write embedding arrays and their self-describing metadata."""
    context_path = destination / "text_context.npy"
    image_embedding_path = destination / "image_clip_embedding.npy"
    metadata_path = destination / "metadata.json"
    np.save(context_path, context_array)
    np.save(image_embedding_path, image_array)
    metadata_path.write_text(
        json.dumps(
            {
                "prompt": prompt,
                "image": str(image_file.resolve()),
                "clip_model": model_id,
                "text_context": {
                    "path": context_path.name,
                    "shape": list(context_array.shape),
                    "dtype": str(context_array.dtype),
                },
                "image_clip_embedding": {
                    "path": image_embedding_path.name,
                    "shape": list(image_array.shape),
                    "dtype": str(image_array.dtype),
                },
                "note": (
                    "text_context.npy is the VeCoR/U-ViT text conditioning tensor; "
                    "the image CLIP embedding is supplementary."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"text_context": context_path, "image_embedding": image_embedding_path, "metadata": metadata_path}


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for single prompt-image extraction."""
    parser = argparse.ArgumentParser(description="Extract VeCoR-compatible CLIP conditioning from a prompt-image pair.")
    parser.add_argument("--prompt", required=True, help="Caption or text prompt to encode.")
    parser.add_argument("--image", required=True, type=Path, help="Image paired with the prompt.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for .npy outputs and metadata.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face CLIP model (must provide 77x768 text context).",
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device", help="Torch device, e.g. cuda, cuda:0, or cpu. Defaults to CUDA when available."
    )
    device_group.add_argument(
        "--gpu", action="store_true", help="Require and use the default CUDA GPU (equivalent to --device cuda)."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    paths = extract_inner_embeddings(
        args.prompt, args.image, args.output_dir, model_id=args.model_id, device="cuda" if args.gpu else args.device
    )
    print(f"Saved VeCoR text context (77x768): {paths['text_context']}")
    print(f"Saved supplemental CLIP image embedding: {paths['image_embedding']}")


if __name__ == "__main__":
    main()
