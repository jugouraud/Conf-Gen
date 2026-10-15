"""Calibrate the research Wasserstein scores on the COCO region split.

The order-1 score transports the new caption's mass to every reference EOS
embedding. The adaptive order-infinity score uses the five reference token
clouds nearest in task-space W1, as in research.prompt_filter. Both use the
saved region's 768D Mahalanobis whitening and the same image-level split.
The adaptive method uses a fixed 60-anchor subset, matching the research
filter's reference size; the order-1 method uses every COCO reference anchor.
"""

import argparse
import json
import os
import sqlite3
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from backend.embeddings.clip import EXPECTED_CONTEXT_SHAPE
from backend.embeddings.define_region import DEFAULT_DATABASE_PATH, DEFAULT_REGION_PATH, _load_region, _source_signature
from backend.embeddings.project_region import _region_signature
from backend.paths import COCO_ROOT
from backend.storage.models import get_local_model_directory
from research.wasserstein import score_orderinf, task_wasserstein1

DEFAULT_TRANSPORT_PATH = COCO_ROOT / "coco_transport_budget.sqlite3"
TRANSPORT_VERSION = 2
TASK_NEIGHBORS = 5
TASK_REFERENCE_COUNT = 60
TASK_REFERENCE_SEED = 0


def order1_score(distances: np.ndarray) -> float:
    """Exact empirical W1 cost from research.wasserstein.score_order1."""
    distances = np.asarray(distances, dtype=np.float64)
    count = len(distances)
    if count == 0:
        raise ValueError("At least one reference distance is required.")
    return float(distances.sum() / (count * (count + 1)))


def _order1_scores(queries: np.ndarray, anchors: np.ndarray, *, batch_size: int = 32) -> np.ndarray:
    """Batch the full-reference transport cost without storing its distance matrix."""
    queries = np.asarray(queries, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)
    anchor_norms = np.einsum("ij,ij->i", anchors, anchors)
    result = np.empty(len(queries), dtype=np.float64)
    divisor = len(anchors) * (len(anchors) + 1)
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        query_norms = np.einsum("ij,ij->i", batch, batch)
        total = np.zeros(len(batch), dtype=np.float64)
        for left in range(0, len(anchors), 4096):
            right = left + 4096
            squared = query_norms[:, None] + anchor_norms[None, left:right]
            squared -= 2 * (batch @ anchors[left:right].T)
            np.sqrt(np.maximum(squared, 0, out=squared), out=squared)
            total += squared.sum(axis=1, dtype=np.float64)
        result[start:start + len(batch)] = total / divisor
    return result


def _inputs(database: Path, artifact: Path) -> dict:
    return {"version": TRANSPORT_VERSION, "source": _source_signature(database),
            "region": _region_signature(artifact), "task_neighbors": TASK_NEIGHBORS,
            "task_reference_count": TASK_REFERENCE_COUNT, "task_reference_seed": TASK_REFERENCE_SEED}


def _read_metadata(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key = 'summary'").fetchone()
        return json.loads(row[0]) if row else None
    except (sqlite3.Error, ValueError, TypeError):
        return None


def _task_cloud(context: bytes, eos: int) -> np.ndarray:
    sequence = np.frombuffer(context, dtype="<f4").reshape(EXPECTED_CONTEXT_SHAPE)
    return np.ascontiguousarray(sequence[1:eos] if eos > 1 else sequence[eos:eos + 1])


class TransportBudget:
    """Full-reference W1 and five task-nearest W-infinity COCO checks."""

    def __init__(self, path: str | Path, region) -> None:
        self.path = Path(path)
        self.region = region
        self.connection = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
        summary_row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'summary'"
        ).fetchone()
        self.summary = json.loads(summary_row[0]) if summary_row else {}
        rows = self.connection.execute(
            "SELECT anchor_index, caption_id, mean FROM anchor_clouds ORDER BY task_index"
        ).fetchall()
        self.anchor_indices = np.asarray([row[0] for row in rows], dtype=np.intp)
        self.caption_ids = np.asarray([row[1] for row in rows], dtype=np.int64)
        self.means = np.stack([np.frombuffer(row[2], dtype="<f4") for row in rows]).astype(np.float64)
        self.mean_norms = np.einsum("ij,ij->i", self.means, self.means)
        if len(rows) < TASK_NEIGHBORS or not np.array_equal(
            self.caption_ids, np.asarray(region.metadata["reference_caption_ids"])[self.anchor_indices]
        ):
            raise ValueError("Transport cache does not match the region anchors.")

    @lru_cache(maxsize=4096)
    def _cloud(self, index: int) -> np.ndarray:
        row = self.connection.execute(
            "SELECT cloud, n_tokens FROM anchor_clouds WHERE task_index = ?", (int(index),)
        ).fetchone()
        return np.frombuffer(row[0], dtype="<f4").reshape(row[1], -1)

    def task_nearest(self, cloud: np.ndarray) -> np.ndarray:
        """Find the exact five nearest clouds, pruning by the W1 mean bound."""
        cloud = np.asarray(cloud, dtype=np.float32)
        query_mean = cloud.mean(axis=0, dtype=np.float64)
        squared = self.mean_norms + query_mean @ query_mean - 2 * (self.means @ query_mean)
        lower_bounds = np.sqrt(np.maximum(squared, 0))
        order = np.argsort(lower_bounds)
        best: list[tuple[float, int]] = []
        for index in order:
            # Means are cached as float32; leave slack so rounding cannot
            # exclude an exact transport-nearest reference.
            if len(best) == TASK_NEIGHBORS and lower_bounds[index] > best[-1][0] + 1e-4:
                break
            distance = task_wasserstein1(cloud, self._cloud(int(index)))
            best.append((distance, int(index)))
            best.sort()
            del best[TASK_NEIGHBORS:]
        return self.anchor_indices[[index for _, index in best]]

    def score(self, whitened: np.ndarray, cloud: np.ndarray,
              distances: np.ndarray | None = None) -> tuple[float, float]:
        whitened = np.asarray(whitened, dtype=np.float32)
        if distances is None:
            distances = np.linalg.norm(self.region.anchors - whitened, axis=1)
        return order1_score(distances), self.adaptive_score(whitened, cloud)

    def adaptive_score(self, whitened: np.ndarray, cloud: np.ndarray) -> float:
        whitened = np.asarray(whitened, dtype=np.float32)
        selected = self.task_nearest(cloud)
        return float(score_orderinf(whitened[None, :], self.region.anchors[selected], n_cuts=0)[0])


def calibrate_transport_budget(
    *, database_path: str | Path = DEFAULT_DATABASE_PATH,
    region_path: str | Path = DEFAULT_REGION_PATH,
    output_path: str | Path = DEFAULT_TRANSPORT_PATH,
    force: bool = False,
) -> Path:
    """Cache task clouds and image-level conformal radii for both scores."""
    from transformers import CLIPTokenizerFast

    database, artifact, destination = Path(database_path), Path(region_path), Path(output_path)
    if destination.resolve() in {database.resolve(), artifact.resolve()}:
        raise ValueError("The transport cache must be separate from the COCO database and region artifact.")
    inputs = _inputs(database, artifact)
    if not force and (summary := _read_metadata(destination)) and summary.get("inputs") == inputs:
        return destination
    region = _load_region(artifact)
    if region.metadata.get("source") != inputs["source"]:
        raise ValueError("The saved region does not match this COCO database.")
    reference_ids = np.asarray(region.metadata["reference_caption_ids"], dtype=np.int64)
    task_indices = np.sort(np.random.default_rng(TASK_REFERENCE_SEED).choice(
        len(reference_ids), size=min(TASK_REFERENCE_COUNT, len(reference_ids)), replace=False,
    ))
    reference_index = {int(reference_ids[index]): (task_index, int(index))
                       for task_index, index in enumerate(task_indices)}
    all_reference_ids = set(map(int, reference_ids))
    tokenizer = CLIPTokenizerFast.from_pretrained(
        get_local_model_directory(region.metadata["model_id"]), local_files_only=True
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, prefix=".coco_transport_", suffix=".sqlite3", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with closing(sqlite3.connect(temporary_path)) as cache, closing(sqlite3.connect(
            f"{database.resolve().as_uri()}?mode=ro", uri=True
        )) as source:
            cache.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE anchor_clouds (
                    task_index INTEGER PRIMARY KEY, anchor_index INTEGER UNIQUE NOT NULL,
                    caption_id INTEGER UNIQUE NOT NULL,
                    mean BLOB NOT NULL, cloud BLOB NOT NULL, n_tokens INTEGER NOT NULL
                );
            """)
            calibration: list[tuple[int, np.ndarray, np.ndarray]] = []
            seen_references = 0
            cursor = source.execute(
                "SELECT caption_id, image_id, caption, text_context, text_context_shape, embedding_model "
                "FROM captions ORDER BY caption_id"
            )
            while rows := cursor.fetchmany(128):
                token_ids = tokenizer(
                    [row[2] for row in rows], padding="max_length", truncation=True,
                    max_length=EXPECTED_CONTEXT_SHAPE[0], return_attention_mask=False, return_tensors="np",
                )["input_ids"]
                for (caption_id, image_id, _, context, shape, model), eos in zip(
                    rows, token_ids.argmax(axis=1), strict=True
                ):
                    if model != region.metadata["model_id"] or shape != json.dumps(list(EXPECTED_CONTEXT_SHAPE)):
                        raise ValueError(f"Caption {caption_id} has an incompatible CLIP context.")
                    if len(context) != np.prod(EXPECTED_CONTEXT_SHAPE) * 4:
                        raise ValueError(f"Caption {caption_id} has an invalid CLIP context byte length.")
                    cloud = _task_cloud(context, int(eos))
                    if caption_id in reference_index:
                        task_index, anchor_index = reference_index[caption_id]
                        cache.execute("INSERT INTO anchor_clouds VALUES (?, ?, ?, ?, ?, ?)", (
                            task_index, anchor_index, caption_id,
                            cloud.mean(axis=0).astype("<f4").tobytes(),
                            cloud.tobytes(), len(cloud),
                        ))
                        seen_references += 1
                    elif caption_id not in all_reference_ids:
                        vector = np.frombuffer(context, dtype="<f4").reshape(EXPECTED_CONTEXT_SHAPE)[eos].copy()
                        calibration.append((image_id, vector, cloud))
            cache.commit()
            if seen_references != len(task_indices) or len(calibration) != region.metadata["n_calibration"]:
                raise ValueError("The COCO reference/calibration split is incomplete.")
            checker = TransportBudget(temporary_path, region)
            whitened_calibration = region.whiten(np.stack([row[1] for row in calibration]))
            order1_calibration = _order1_scores(whitened_calibration, region.anchors)
            by_image: dict[int, list[float]] = {}
            for index, ((image_id, _, cloud), whitened, score1) in enumerate(zip(
                calibration, whitened_calibration, order1_calibration, strict=True
            ), 1):
                scoreinf = checker.adaptive_score(whitened, cloud)
                scores = by_image.setdefault(image_id, [float("-inf"), float("-inf")])
                scores[0] = max(scores[0], score1)
                scores[1] = max(scores[1], scoreinf)
                if index % 100 == 0:
                    print(f"Calibrated transportation scores for {index:,}/{len(calibration):,} captions", flush=True)
            checker.connection.close()
            image_scores = np.asarray(list(by_image.values()), dtype=np.float64)
            n_images = len(image_scores)
            if n_images != region.metadata["n_calibration_images"]:
                raise ValueError("Calibration image count does not match the COCO region.")
            rank = int(np.ceil((n_images + 1) * (1 - region.metadata["miscoverage"])))
            if rank > n_images:
                raise ValueError("Too few calibration images for a finite transport budget.")
            radii = np.partition(image_scores, rank - 1, axis=0)[rank - 1]
            summary = {"inputs": inputs, "order1_radius": float(radii[0]),
                       "orderinf_radius": float(radii[1]), "n_calibration_images": n_images,
                       "n_reference": len(reference_ids), "n_task_reference": len(task_indices),
                       "calibration_rank": rank}
            if _inputs(database, artifact) != inputs:
                raise RuntimeError("The COCO database or region changed during transportation calibration.")
            cache.execute("INSERT INTO metadata VALUES ('summary', ?)", (json.dumps(summary),))
            cache.commit()
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate COCO transportation budgets.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--region", type=Path, default=DEFAULT_REGION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_TRANSPORT_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(calibrate_transport_budget(database_path=args.database, region_path=args.region,
                                     output_path=args.output, force=args.force))


if __name__ == "__main__":
    main()
