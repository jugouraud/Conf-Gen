"""Live CLIP prompt checks against the saved COCO caption region."""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np

from backend.embeddings.clip import EXPECTED_CONTEXT_SHAPE
from backend.embeddings.define_region import DEFAULT_REGION_PATH, _load_region
from backend.embeddings.project_region import DEFAULT_PROJECTION_PATH
from backend.paths import VALIDATION_ROOT
from backend.storage.models import get_local_model_directory

DEFAULT_VALIDATION_DIRECTORY = VALIDATION_ROOT / "toxic_source"
DEFAULT_PROMPT_MAPPING_PATH = VALIDATION_ROOT / "hf_test_toxicity_privacy_real.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ValidationItem:
    image_path: Path
    prompt: str
    prompt_source: str


@dataclass(frozen=True)
class ValidationResult:
    prompt: str
    score: float
    radius: float
    blocked: bool
    pca_coordinates: np.ndarray
    tsne_coordinates: np.ndarray
    nearest_caption_id: int
    reference_caption_ids: np.ndarray
    anchor_distances: np.ndarray


def load_validation_items(
    directory: str | Path = DEFAULT_VALIDATION_DIRECTORY,
    prompt_mapping_path: str | Path | None = None,
) -> list[ValidationItem]:
    """List images and pair them with mapped prompts or marked filename proxies."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {directory}")
    mapping_path = Path(prompt_mapping_path) if prompt_mapping_path else DEFAULT_PROMPT_MAPPING_PATH
    mapping = _read_prompt_mapping(mapping_path) if mapping_path.is_file() else {}
    if prompt_mapping_path and not mapping_path.is_file():
        raise FileNotFoundError(f"Validation prompt mapping not found: {mapping_path}")
    images = sorted((path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                    key=lambda path: path.name.casefold())
    if not images:
        raise ValueError(f"No validation images found in {directory}")
    items = []
    for image in images:
        prompt = mapping.get(image.name)
        if prompt:
            items.append(ValidationItem(image, prompt, "Dataset caption"))
        else:
            category = image.stem.rsplit("_", 1)[0].replace("_", " ")
            items.append(ValidationItem(image, category, "Filename category proxy; edit if the original prompt is known"))
    return items


def _read_prompt_mapping(path: Path) -> dict[str, str]:
    """Accept the toxicity dataset records or a filename-to-prompt JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        mapping: dict[str, str] = {}
        for item in payload:
            prompt = item.get("prompt", item.get("caption"))
            images = item.get("image", [])
            if isinstance(images, str):
                images = [images]
            if not isinstance(prompt, str) or not prompt.strip() or not isinstance(images, list):
                raise ValueError(f"Dataset record has no usable caption or image: {path}")
            for image in images:
                if not isinstance(image, str):
                    raise ValueError(f"Dataset image path is invalid: {path}")
                name = Path(image.replace("\\", "/")).name
                if name in mapping and mapping[name] != prompt.strip():
                    raise ValueError(f"Image {name} has conflicting captions in {path}")
                mapping[name] = prompt.strip()
        payload = mapping
    if not isinstance(payload, dict) or any(
        not isinstance(name, str) or not isinstance(prompt, str) or not prompt.strip()
        for name, prompt in payload.items()
    ):
        raise ValueError(f"Prompt mapping must contain nonempty filename-to-prompt strings: {path}")
    return {name: prompt.strip() for name, prompt in payload.items()}


class LiveCocoRegionValidator:
    """Load CLIP once and classify new prompt EOS vectors in the original 768D region."""

    def __init__(
        self,
        region_path: str | Path = DEFAULT_REGION_PATH,
        projection_path: str | Path = DEFAULT_PROJECTION_PATH,
        *, device: str | None = None,
    ) -> None:
        import torch
        from transformers import CLIPModel, CLIPTokenizerFast

        self.region = _load_region(Path(region_path))
        self.reference_caption_ids = np.asarray(self.region.metadata["reference_caption_ids"], dtype=np.int64)
        if len(self.reference_caption_ids) != len(self.region.anchors):
            raise ValueError("The saved region has inconsistent reference IDs and anchors.")
        self.projection_path = Path(projection_path)
        with closing(sqlite3.connect(f"{self.projection_path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key = 'pca_model'").fetchone()
        if row is None:
            raise ValueError("Projection cache lacks a PCA transform; run python -m backend.embeddings.project_region.")
        pca = json.loads(row[0])
        self.pca_mean = np.asarray(pca["mean"], dtype=np.float32)
        self.pca_components = np.asarray(pca["components"], dtype=np.float32)
        self.pca_offset = np.asarray(pca.get("offset", [0.0, 0.0, 0.0]), dtype=np.float32)
        if self.pca_mean.shape != self.region.mean.shape or self.pca_components.shape != (3, len(self.pca_mean)):
            raise ValueError("Projection cache has an invalid PCA transform; rebuild it.")
        if self.pca_offset.shape != (3,):
            raise ValueError("Projection cache has an invalid PCA offset; rebuild it.")

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_directory = get_local_model_directory(self.region.metadata["model_id"])
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_directory, local_files_only=True)
        clip = CLIPModel.from_pretrained(model_directory, local_files_only=True).to(self.device).eval()
        self.text_model = clip.text_model
        del clip
        self._model_lock = Lock()

    def encode_prompts(self, prompts: list[str], *, batch_size: int = 16) -> np.ndarray:
        """Encode prompts with the same padded EOS extraction used by COCO."""
        if batch_size < 1 or any(not prompt.strip() for prompt in prompts):
            raise ValueError("Prompts must be nonempty and batch_size must be positive.")
        vectors = []
        for start in range(0, len(prompts), batch_size):
            tokens = self.tokenizer(
                prompts[start:start + batch_size], padding="max_length", truncation=True,
                max_length=EXPECTED_CONTEXT_SHAPE[0], return_tensors="pt",
            )
            eos = tokens["input_ids"].argmax(dim=1)
            inputs = {key: value.to(self.device) for key, value in tokens.items() if key in ("input_ids", "attention_mask")}
            with self._model_lock, self.torch.inference_mode():
                context = self.text_model(**inputs).last_hidden_state
            row_indices = self.torch.arange(len(eos), device=self.device)
            vectors.append(context[row_indices, eos.to(self.device)].float().cpu().numpy())
        result = np.concatenate(vectors) if vectors else np.empty((0, len(self.region.mean)), dtype=np.float32)
        if result.shape != (len(prompts), len(self.region.mean)) or not np.isfinite(result).all():
            raise ValueError("CLIP returned invalid EOS embeddings.")
        return result

    def validate(self, prompt: str) -> ValidationResult:
        """Encode one prompt, score it, and place it in both visualizations."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Enter a prompt before checking the region.")
        vector = self.encode_prompts([prompt])[0]
        whitened = self.region.whiten(vector)[0]
        distances = np.linalg.norm(self.region.anchors - whitened, axis=1)
        nearest = int(np.argmin(distances))
        score = float(distances[nearest])
        nearest_id = int(self.reference_caption_ids[nearest])
        with closing(sqlite3.connect(f"{self.projection_path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            row = connection.execute(
                "SELECT tsne_1, tsne_2, tsne_3 FROM captions WHERE caption_id = ?", (nearest_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Nearest anchor {nearest_id} is absent from the projection cache.")
        return ValidationResult(
            prompt=prompt, score=score, radius=self.region.radius,
            blocked=score > self.region.radius,
            pca_coordinates=(whitened - self.pca_mean) @ self.pca_components.T + self.pca_offset,
            tsne_coordinates=np.asarray(row, dtype=np.float32),
            nearest_caption_id=nearest_id,
            reference_caption_ids=self.reference_caption_ids,
            anchor_distances=distances,
        )
