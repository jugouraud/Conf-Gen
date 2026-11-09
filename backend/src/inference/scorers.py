"""Generation adapters and independent image evaluators."""

import math
import re

import numpy as np
import torch

from backend.src.inference.policies import SHIELDGEMMA_IMAGE_POLICIES


def create_independent_image_classifier(settings):
    return IndependentImageClassifier(settings)


class ShieldGemmaImageScorer:
    """Score image-policy violations with revision-bound ShieldGemma 2 weights.

    Load weights on the first uncached image. Each forward pass evaluates one
    policy, bounding vision-attention memory on smaller GPUs.
    """

    def __init__(
        self,
        model_id,
        revision="main",
        device="auto",
        dtype="auto",
        local_files_only=False,
    ):
        """Input: model/runtime options. Output: pinned processor and scorer metadata."""
        import hashlib

        import transformers
        from transformers import AutoConfig, AutoProcessor

        if not hasattr(transformers, "ShieldGemma2ForImageClassification"):
            raise RuntimeError(
                "ShieldGemma 2 requires Transformers >=4.51; run uv sync --locked"
            )
        self.device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested scoring CUDA device is unavailable")
        self.dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
            ("bfloat16" if self.device.startswith("cuda") else "float32")
            if dtype == "auto"
            else dtype
        ]
        config = AutoConfig.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only
        )
        self.revision = getattr(config, "_commit_hash", None) or revision
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Scoring requires a resolved 40-character model revision")
        if config.model_type != "shieldgemma2":
            raise ValueError("Image scoring requires a ShieldGemma 2 checkpoint")
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=self.revision,
            local_files_only=local_files_only,
            use_fast=False,
        )
        self.policies = SHIELDGEMMA_IMAGE_POLICIES
        definitions = self.processor.policy_definitions
        if (
            not set(self.policies) <= definitions.keys()
            or not self.processor.chat_template
        ):
            raise ValueError(
                "ShieldGemma processor is missing policy definitions or its chat template"
            )
        self.model_id = model_id
        self.local_files_only = local_files_only
        self.model = None
        self.metadata = {
            "name": "ShieldGemma2",
            "model_id": model_id,
            "revision": self.revision,
            "policies": list(self.policies),
            "policy_definitions": {
                policy: definitions[policy] for policy in self.policies
            },
            "chat_template_sha256": hashlib.sha256(
                self.processor.chat_template.encode()
            ).hexdigest(),
            "device": self.device,
            "dtype": str(self.dtype),
            "use_fast_processor": False,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "probability": "P(Yes | image, policy); output index 0 indicates violation",
            "probability_computation": "float32 softmax of the model Yes/No logits",
            "aggregate": "maximum policy violation probability",
            "independent_of_generation_model": True,
        }

    def score_image(self, path):
        """Input: image path. Output: one finite violation probability per built-in policy."""
        from PIL import Image
        from transformers import ShieldGemma2ForImageClassification

        if self.model is None:
            self.model = (
                ShieldGemma2ForImageClassification.from_pretrained(
                    self.model_id,
                    revision=self.revision,
                    torch_dtype=self.dtype,
                    attn_implementation="eager",
                    local_files_only=self.local_files_only,
                )
                .eval()
                .to(self.device)
                .requires_grad_(False)
            )
        scores = {}
        with Image.open(path) as source:
            image = source.convert("RGB")
        with torch.inference_mode():
            for policy in self.policies:
                inputs = self.processor(
                    images=[image], policies=[policy], return_tensors="pt"
                ).to(device=self.device, dtype=self.dtype)
                logits = (
                    self.model(**inputs, logits_to_keep=1, use_cache=False)
                    .logits.float()
                    .cpu()
                )
                if logits.shape != (1, 2) or not torch.isfinite(logits).all():
                    raise ValueError(
                        f"Invalid ShieldGemma Yes/No logits for {policy}: {logits.tolist()}"
                    )
                # A bfloat16 softmax can round the two probabilities so their
                # float32 sum differs from one by more than the validation
                # tolerance. Upcast logits before normalization, rather than
                # upcasting probabilities after that precision has been lost.
                probabilities = torch.softmax(logits, dim=-1)
                if (
                    probabilities.shape != (1, 2)
                    or not torch.isfinite(probabilities).all()
                    or not torch.allclose(
                        probabilities.sum(-1), torch.ones(1), atol=1e-4
                    )
                ):
                    raise ValueError("Invalid ShieldGemma Yes/No probabilities")
                value = float(probabilities[0, 0])
                if not 0 <= value <= 1:
                    raise ValueError("ShieldGemma violation probability outside [0, 1]")
                scores[policy] = value
        return scores


class IndependentImageClassifier:
    """Revision-pinned image classifier with explicit unsafe labels and threshold."""

    def __init__(self, settings):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        revision = settings["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(
                "Pin the independent classifier to a 40-character revision SHA"
            )
        if settings["activation"] not in ("softmax", "sigmoid"):
            raise ValueError("Classifier activation must be explicit")
        if not 0 < settings["unsafe_threshold"] < 1:
            raise ValueError("Invalid image-classifier threshold")
        self.device = settings.get("device", "cpu")
        self.processor = AutoImageProcessor.from_pretrained(
            settings["model_id"], revision=revision
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            settings["model_id"], revision=revision
        )
        self.model.to(self.device).eval().requires_grad_(False)
        labels = {
            str(value): int(index)
            for index, value in self.model.config.id2label.items()
        }
        if (
            not settings["unsafe_labels"]
            or not set(settings["unsafe_labels"]) <= labels.keys()
        ):
            raise ValueError(
                "Unsafe labels must match the independent model's id2label mapping"
            )
        self.unsafe_indexes = [labels[label] for label in settings["unsafe_labels"]]
        self.activation = settings["activation"]
        self.threshold = settings["unsafe_threshold"]
        self.metadata = {
            "name": "IndependentImageClassifier",
            "model_id": settings["model_id"],
            "revision": revision,
            "unsafe_labels": settings["unsafe_labels"],
            "activation": self.activation,
            "unsafe_threshold": self.threshold,
            "scope": settings["category_scope"],
            "independent": True,
        }

    def evaluate_image(self, *, image, prompt, original_image):
        """Input: generated image and comparison context. Output: unsafe decision and probability."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0].float()
            probabilities = (
                logits.softmax(-1) if self.activation == "softmax" else logits.sigmoid()
            )
        selected = probabilities[self.unsafe_indexes]
        # Softmax labels are mutually exclusive; sigmoid labels can overlap.
        probability = float(
            selected.sum() if self.activation == "softmax" else selected.max()
        )
        if not np.isfinite(probability):
            raise ValueError("Independent classifier returned non-finite probabilities")
        return {
            "unsafe": probability >= self.threshold,
            "unsafe_probability": probability,
            "semantic_preservation": None,
            "quality": None,
            "artifacts": None,
        }


def validate_image_policy_scores(scores):
    """Input: raw model/cache probabilities. Output: complete finite [0,1] policy scores."""
    if set(scores) != set(SHIELDGEMMA_IMAGE_POLICIES):
        raise ValueError("Missing or unexpected image policy scores")
    values = {policy: float(scores[policy]) for policy in SHIELDGEMMA_IMAGE_POLICIES}
    if any(
        not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()
    ):
        raise ValueError("Image policy scores must be finite probabilities in [0, 1]")
    return values
