"""Live CLIP prompt checks against the saved COCO caption region."""

import csv
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np

from backend.embeddings.clip import EXPECTED_CONTEXT_SHAPE
from backend.embeddings.define_region import DEFAULT_REGION_PATH, _load_region
from backend.embeddings.project_region import DEFAULT_PROJECTION_PATH
from backend.embeddings.transport_budget import TransportBudget, calibrate_transport_budget
from backend.paths import VALIDATION_ROOT
from backend.storage.models import get_local_model_directory
from backend.validation.prompt_cache import DEFAULT_VALIDATION_DATABASE_PATH, PromptEncodingStore

DEFAULT_PROMPTS_PATH = VALIDATION_ROOT / "i2p_benchmark.csv"


@dataclass(frozen=True)
class ValidationItem:
    row_number: int
    prompt: str
    categories: str
    prompt_toxicity: float


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
    order1_score: float | None = None
    order1_radius: float | None = None
    order1_blocked: bool | None = None
    orderinf_score: float | None = None
    orderinf_radius: float | None = None
    orderinf_blocked: bool | None = None


def load_validation_items(
    prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
) -> list[ValidationItem]:
    """Load every CSV row, preserving duplicate prompts and row order."""
    path = Path(prompts_path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"prompt", "prompt_toxicity"}.issubset(reader.fieldnames):
            raise ValueError(f"Validation CSV needs prompt and prompt_toxicity columns: {path}")
        items = []
        for row_number, row in enumerate(reader, start=1):
            prompt = row["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Empty prompt at CSV row {row_number + 1}: {path}")
            try:
                prompt_toxicity = float(row["prompt_toxicity"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid prompt_toxicity at CSV row {row_number + 1}: {path}") from error
            if not math.isfinite(prompt_toxicity):
                raise ValueError(f"Invalid prompt_toxicity at CSV row {row_number + 1}: {path}")
            items.append(ValidationItem(
                row_number, prompt.strip(), (row.get("categories") or "unknown").strip() or "unknown",
                prompt_toxicity,
            ))
    if not items:
        raise ValueError(f"No validation prompts found in {path}")
    return items


class LiveCocoRegionValidator:
    """Load CLIP once and classify new prompt EOS vectors in the original 768D region."""

    def __init__(
        self,
        region_path: str | Path = DEFAULT_REGION_PATH,
        projection_path: str | Path = DEFAULT_PROJECTION_PATH,
        *, transport_path: str | Path | None = None,
        database_path: str | Path | None = None,
        validation_database_path: str | Path = DEFAULT_VALIDATION_DATABASE_PATH,
        device: str | None = None,
    ) -> None:
        self.region = _load_region(Path(region_path))
        self.transport = None if transport_path is None else TransportBudget(
            calibrate_transport_budget(
                region_path=region_path, output_path=transport_path,
                **({"database_path": database_path} if database_path is not None else {}),
            ), self.region,
        )
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

        self.encoder_key = f"{self.region.metadata['model_id']}|padded-eos-cloud-v1|{EXPECTED_CONTEXT_SHAPE}"
        self.encoding_store = PromptEncodingStore(validation_database_path)
        self._requested_device = device
        self.text_model = None
        self._load_lock = Lock()
        self._model_lock = Lock()

    def encode_prompts(self, prompts: list[str], *, batch_size: int = 16) -> np.ndarray:
        """Encode prompts with the same padded EOS extraction used by COCO."""
        vectors, _ = self.encode_prompts_with_clouds(prompts, batch_size=batch_size)
        return vectors

    def encode_prompts_with_clouds(
        self, prompts: list[str], *, batch_size: int = 16,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return cached encodings, running CLIP only for missing prompts."""
        if batch_size < 1 or any(not prompt.strip() for prompt in prompts):
            raise ValueError("Prompts must be nonempty and batch_size must be positive.")
        prompts = [prompt.strip() for prompt in prompts]
        dimension = len(self.region.mean)
        unique_prompts = list(dict.fromkeys(prompts))
        cached = self.encoding_store.read_many(self.encoder_key, unique_prompts, dimension)
        missing = [prompt for prompt in unique_prompts if prompt not in cached]
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            vectors, clouds = self._encode_uncached(batch)
            newly_encoded = dict(zip(batch, zip(vectors, clouds, strict=True), strict=True))
            self.encoding_store.write_many(self.encoder_key, newly_encoded, dimension)
            cached.update(newly_encoded)
        result = np.stack([cached[prompt][0] for prompt in prompts]) if prompts else np.empty((0, dimension), dtype=np.float32)
        if result.shape != (len(prompts), dimension) or not np.isfinite(result).all():
            raise ValueError("CLIP returned invalid EOS embeddings.")
        return result, [cached[prompt][1] for prompt in prompts]

    def _ensure_text_model(self) -> None:
        """Keep startup light when every selected prompt already has an encoding."""
        with self._load_lock:
            if self.text_model is not None:
                return
            import torch
            from transformers import CLIPModel, CLIPTokenizerFast

            self.torch = torch
            self.device = torch.device(self._requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
            model_directory = get_local_model_directory(self.region.metadata["model_id"])
            self.tokenizer = CLIPTokenizerFast.from_pretrained(model_directory, local_files_only=True)
            clip = CLIPModel.from_pretrained(model_directory, local_files_only=True).to(self.device).eval()
            self.text_model = clip.text_model

    def _encode_uncached(self, prompts: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
        self._ensure_text_model()
        vectors = []
        clouds: list[np.ndarray] = []
        tokens = self.tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=EXPECTED_CONTEXT_SHAPE[0], return_tensors="pt",
        )
        eos = tokens["input_ids"].argmax(dim=1)
        inputs = {key: value.to(self.device) for key, value in tokens.items() if key in ("input_ids", "attention_mask")}
        with self._model_lock, self.torch.inference_mode():
            context = self.text_model(**inputs).last_hidden_state
        row_indices = self.torch.arange(len(eos), device=self.device)
        vectors.append(context[row_indices, eos.to(self.device)].float().cpu().numpy())
        contexts = context.float().cpu().numpy()
        for sequence, end in zip(contexts, eos.tolist(), strict=True):
            clouds.append(np.ascontiguousarray(
                sequence[1:end] if end > 1 else sequence[end:end + 1]
            ))
        return np.concatenate(vectors), clouds

    def validate(self, prompt: str) -> ValidationResult:
        """Encode one prompt, score it, and place it in both visualizations."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Enter a prompt before checking the region.")
        vectors, clouds = self.encode_prompts_with_clouds([prompt])
        vector, cloud = vectors[0], clouds[0]
        whitened = self.region.whiten(vector)[0]
        distances = np.linalg.norm(self.region.anchors - whitened, axis=1)
        if self.transport is not None:
            order1_score, orderinf_score = self.transport.score(whitened, cloud, distances)
            order1_radius = self.transport.summary["order1_radius"]
            orderinf_radius = self.transport.summary["orderinf_radius"]
        else:
            order1_score = orderinf_score = order1_radius = orderinf_radius = None
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
            order1_score=order1_score, order1_radius=order1_radius,
            order1_blocked=order1_score > order1_radius if order1_score is not None else None,
            orderinf_score=orderinf_score, orderinf_radius=orderinf_radius,
            orderinf_blocked=orderinf_score > orderinf_radius if orderinf_score is not None else None,
        )
