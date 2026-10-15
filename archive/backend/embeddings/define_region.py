"""Fit and cache a conformal region over COCO caption text embeddings.

Run ``python -m backend.embeddings.define_region``. The region uses the
union-of-balls geometry of the initial research's connected, zero-cut case:
Mahalanobis balls around reference [EOS] embeddings. We score nearest-anchor
distance directly, so the construction stays a union of balls even if the
larger COCO anchor graph is disconnected. Its radius is a split conformal
quantile of held-out image scores.
"""

import argparse
import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from backend.embeddings.clip import DEFAULT_MODEL_ID, EXPECTED_CONTEXT_SHAPE
from backend.paths import COCO_ROOT
from backend.storage.models import get_local_model_directory

DEFAULT_DATABASE_PATH = COCO_ROOT / "coco.sqlite3"
DEFAULT_REGION_PATH = COCO_ROOT / "coco_region.npz"
REGION_VERSION = 3


@dataclass
class CocoRegion:
    """A reusable union of balls in whitened CLIP [EOS] space."""

    anchors: np.ndarray  # whitened reference vectors, shape (M, 768)
    mean: np.ndarray
    components: np.ndarray  # top PCA directions, shape (768, m)
    eigenvalues: np.ndarray
    gamma: float
    radius: float
    metadata: dict

    def whiten(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError(f"Expected vectors with shape (N, {self.mean.size}).")
        centered = values - self.mean
        base = np.float32(1 / np.sqrt(self.gamma))
        correction = 1 / np.sqrt(self.eigenvalues) - base
        return centered * base + ((centered @ self.components) * correction) @ self.components.T

    def score(self, vectors: np.ndarray, *, batch_size: int = 128) -> np.ndarray:
        """Minimum Mahalanobis distance to any reference caption vector."""
        return _nearest_distances(self.whiten(vectors), self.anchors, batch_size=batch_size)

    def contains(self, vectors: np.ndarray, *, batch_size: int = 128) -> np.ndarray:
        return self.score(vectors, batch_size=batch_size) <= self.radius


def _source_signature(database: Path) -> dict:
    paths = [database, Path(f"{database}-wal")]
    if not database.is_file():
        raise FileNotFoundError(f"COCO database not found: {database}")
    return {
        "database": str(database.resolve()),
        "files": [
            {"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths if path.is_file()
        ],
    }


def _load_eos_vectors(database: Path, model_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read each stored context once and select the caption's [EOS] row."""
    from transformers import CLIPTokenizerFast

    model_directory = get_local_model_directory(model_id)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_directory, local_files_only=True)
    ids: list[int] = []
    image_ids: list[int] = []
    blocks: list[np.ndarray] = []
    shape = EXPECTED_CONTEXT_SHAPE
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        total, complete = connection.execute(
            "SELECT COUNT(*), COUNT(text_context) FROM captions"
        ).fetchone()
        if total == 0 or complete != total:
            raise ValueError(f"COCO embeddings are incomplete: {complete}/{total} captions have text contexts.")
        cursor = connection.execute(
            "SELECT caption_id, image_id, caption, text_context, text_context_shape, embedding_model "
            "FROM captions ORDER BY caption_id"
        )
        while rows := cursor.fetchmany(128):
            captions = [row[2] for row in rows]
            token_ids = tokenizer(
                captions, padding="max_length", truncation=True,
                max_length=shape[0], return_attention_mask=False, return_tensors="np",
            )["input_ids"]
            eos_indices = token_ids.argmax(axis=1)
            for (caption_id, image_id, _, context, stored_shape, stored_model), eos in zip(rows, eos_indices, strict=True):
                if stored_model != model_id:
                    raise ValueError(f"Caption {caption_id} uses {stored_model!r}, expected {model_id!r}.")
                if stored_shape != json.dumps(list(shape)) or len(context) != np.prod(shape) * 4:
                    raise ValueError(f"Caption {caption_id} has an invalid text-context shape or byte length.")
                ids.append(caption_id)
                image_ids.append(image_id)
                blocks.append(np.frombuffer(context, dtype="<f4").reshape(shape)[eos].copy())
    vectors = np.stack(blocks)
    if not np.isfinite(vectors).all():
        raise ValueError("COCO text contexts contain non-finite values.")
    return np.asarray(ids, dtype=np.int64), np.asarray(image_ids, dtype=np.int64), vectors


def _fit_whitener(reference: np.ndarray, m_pca: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Probabilistic-PCA covariance: top eigenvalues and an isotropic tail."""
    if len(reference) < 3:
        raise ValueError("At least three reference vectors are required.")
    mean = reference.mean(axis=0, dtype=np.float64)
    centered = reference.astype(np.float64) - mean
    covariance = centered.T @ centered / (len(reference) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    count = min(m_pca, len(reference) - 2, reference.shape[1] - 1)
    if count < 1:
        raise ValueError("m_pca must be positive.")
    # Match research.ecf.fit_regularized_covariance: average the discarded
    # nonzero sample eigenvalues, then floor the tail at 1e-8.
    tolerance = max(reference.shape) * np.finfo(np.float64).eps * eigenvalues[0]
    effective_rank = int(np.count_nonzero(eigenvalues > tolerance))
    count = min(count, max(effective_rank - 1, 1))
    tail = eigenvalues[count:effective_rank]
    gamma = max(float(tail.mean() if len(tail) else eigenvalues[max(effective_rank - 1, 0)]), 1e-8)
    top = np.maximum(eigenvalues[:count], gamma).astype(np.float32)
    components = eigenvectors[:, order[:count]].astype(np.float32)
    return mean.astype(np.float32), components, top, gamma


def _nearest_distances(queries: np.ndarray, anchors: np.ndarray, *, batch_size: int = 128) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    queries = np.asarray(queries, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)
    if len(anchors) == 0:
        raise ValueError("The region needs at least one reference anchor.")
    anchor_norms = np.einsum("ij,ij->i", anchors, anchors)
    result = np.empty(len(queries), dtype=np.float32)
    # Chunk both axes: the full COCO distance matrix would require hundreds
    # of megabytes and must never be materialized just to get row minima.
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        query_norms = np.einsum("ij,ij->i", batch, batch)
        minima = np.full(len(batch), np.inf, dtype=np.float32)
        nearest_indices = np.zeros(len(batch), dtype=np.intp)
        for left in range(0, len(anchors), 4096):
            right = left + 4096
            distances_sq = query_norms[:, None] + anchor_norms[None, left:right]
            distances_sq -= 2 * (batch @ anchors[left:right].T)
            local_indices = np.argmin(distances_sq, axis=1)
            local_minima = distances_sq[np.arange(len(batch)), local_indices]
            better = local_minima < minima
            minima[better] = local_minima[better]
            nearest_indices[better] = left + local_indices[better]
        # The squared-norm identity loses precision for points on an anchor.
        # Re-evaluate the winning pair by direct subtraction before calibration.
        result[start:start + len(batch)] = np.linalg.norm(batch - anchors[nearest_indices], axis=1)
    return result


def _load_region(path: Path) -> CocoRegion:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        return CocoRegion(
            anchors=archive["anchors"], mean=archive["mean"],
            components=archive["components"], eigenvalues=archive["eigenvalues"],
            gamma=float(archive["gamma"]), radius=float(archive["radius"]), metadata=metadata,
        )


def define_coco_region(
    *, database_path: str | Path = DEFAULT_DATABASE_PATH,
    region_path: str | Path = DEFAULT_REGION_PATH,
    model_id: str = DEFAULT_MODEL_ID,
    miscoverage: float = 0.05,
    calibration_fraction: float = 0.2,
    m_pca: int = 32,
    seed: int = 0,
    batch_size: int = 128,
    force: bool = False,
) -> CocoRegion:
    """Return a cached region or fit it from the complete COCO embedding DB."""
    if not 0 < miscoverage < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("miscoverage and calibration_fraction must both be between 0 and 1.")
    if m_pca < 1 or batch_size < 1:
        raise ValueError("m_pca and batch_size must be positive.")
    database, destination = Path(database_path), Path(region_path)
    signature = _source_signature(database)
    settings = {
        "version": REGION_VERSION, "source": signature, "model_id": model_id,
        "miscoverage": miscoverage, "calibration_fraction": calibration_fraction,
        "m_pca": m_pca, "seed": seed, "score": "nearest_mahalanobis_eos",
    }
    if destination.is_file() and not force:
        try:
            cached = _load_region(destination)
            if all(cached.metadata.get(key) == value for key, value in settings.items()):
                return cached
        except (OSError, ValueError, KeyError, TypeError):
            pass  # A corrupt or old cache is rebuilt from the database.

    caption_ids, image_ids, vectors = _load_eos_vectors(database, model_id)
    if _source_signature(database) != signature:
        raise RuntimeError("The COCO database changed while its embeddings were being read; rerun the command.")
    count = len(vectors)
    unique_images, image_groups = np.unique(image_ids, return_inverse=True)
    n_cal_images = max(1, int(np.ceil(len(unique_images) * calibration_fraction)))
    if int(np.ceil((n_cal_images + 1) * (1 - miscoverage))) > n_cal_images:
        raise ValueError("Too few COCO embeddings for a finite conformal radius at this miscoverage.")
    # Captions from the same image are dependent. Keep each image wholly in
    # one split and calibrate one maximum score per image.
    permutation = np.random.default_rng(seed).permutation(len(unique_images))
    cal_group_ids = permutation[:n_cal_images]
    cal_mask = np.isin(image_groups, cal_group_ids)
    cal_indices, ref_indices = np.flatnonzero(cal_mask), np.flatnonzero(~cal_mask)
    if len(ref_indices) < 3:
        raise ValueError("At least three reference captions are required.")
    reference, calibration = vectors[ref_indices], vectors[cal_indices]
    mean, components, eigenvalues, gamma = _fit_whitener(reference, m_pca)
    region = CocoRegion(
        anchors=np.empty((0, vectors.shape[1]), dtype=np.float32),
        mean=mean, components=components, eigenvalues=eigenvalues, gamma=gamma,
        radius=np.inf, metadata={},
    )
    region.anchors = region.whiten(reference).astype(np.float32)
    scores = _nearest_distances(region.whiten(calibration), region.anchors, batch_size=batch_size)
    per_image_scores = np.full(len(unique_images), -np.inf, dtype=np.float32)
    np.maximum.at(per_image_scores, image_groups[cal_indices], scores)
    calibration_scores = per_image_scores[cal_group_ids]
    rank = int(np.ceil((n_cal_images + 1) * (1 - miscoverage)))
    region.radius = float(np.partition(calibration_scores, rank - 1)[rank - 1])
    region.metadata = {
        **settings, "n_captions": count, "n_images": len(unique_images),
        "n_reference": len(reference), "n_calibration": len(calibration),
        "n_reference_images": len(unique_images) - n_cal_images,
        "n_calibration_images": n_cal_images, "calibration_rank": rank,
        "reference_caption_ids": caption_ids[ref_indices].tolist(),
    }
    if _source_signature(database) != signature:
        raise RuntimeError("The COCO database changed during region fitting; rerun the command.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, prefix=".coco_region_", suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(
            temporary_path, anchors=region.anchors, mean=region.mean,
            components=region.components, eigenvalues=region.eigenvalues,
            gamma=region.gamma, radius=region.radius,
            metadata=json.dumps(region.metadata),
        )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return region


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit/cache the conformal region of COCO caption embeddings.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REGION_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--miscoverage", type=float, default=0.05)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--m-pca", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true", help="Recompute even if the saved region is current.")
    args = parser.parse_args()
    region = define_coco_region(
        database_path=args.database, region_path=args.output, model_id=args.model_id,
        miscoverage=args.miscoverage, calibration_fraction=args.calibration_fraction,
        m_pca=args.m_pca, seed=args.seed, batch_size=args.batch_size, force=args.force,
    )
    print(
        f"COCO region: {region.metadata['n_reference']} reference captions, "
        f"{region.metadata['n_calibration']} calibration captions "
        f"({region.metadata['n_calibration_images']} images), "
        f"radius={region.radius:.6g}; saved at {args.output}"
    )


if __name__ == "__main__":
    main()
