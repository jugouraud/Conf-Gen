"""Calibrate order-1 and task-adaptive order-infinity scores on safe T2I prompts.

Uses the same Mahalanobis metric, full-reference order-1 score, all safe-reference
task clouds, five task-nearest anchors, and zero cuts. All budgets use only the
safe calibration split.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
from transformers import CLIPModel, CLIPTokenizerFast

from backend.embeddings.clip import EXPECTED_CONTEXT_SHAPE
from backend.embeddings.define_region import _load_region
from backend.embeddings.prompt_region import (
    DEFAULT_CACHE,
    DEFAULT_CORPUS,
    DEFAULT_REGION,
    _source_sha256,
    load_corpus,
)
from backend.embeddings.transport_budget import (
    TASK_NEIGHBORS,
    _order1_scores,
)
from backend.paths import DATA_ROOT
from backend.storage.models import get_local_model_directory
from backend.validation.prompt_cache import DEFAULT_VALIDATION_DATABASE_PATH
from backend.validation.safe_prompt_analysis import _cached_vectors
from research.wasserstein import score_orderinf, task_wasserstein1

DEFAULT_TRANSPORT = DATA_ROOT / "prompts" / "safe_prompt_transport_full_reference.sqlite3"
DEFAULT_REPORT = DATA_ROOT / "prompts" / "safe_prompt_method_report_full_reference.json"
METHOD_COLUMNS = {"nearest_anchor": "nearest", "order1": "order1", "orderinf": "orderinf"}
SPLITS = ("safe_calibration", "safe_test", "toxic_test_i2p", "toxic_test_t2i_risky")


class CloudEncoder:
    """Use old validation clouds when present, otherwise encode content tokens."""

    def __init__(self, model_id: str, legacy_cache: Path | None, device: str | None):
        self.model_id = model_id
        self.legacy_cache = legacy_cache
        self.device_name = device
        self.text_model = None

    def _ensure_model(self) -> None:
        if self.text_model is not None:
            return
        import torch

        self.torch = torch
        self.device = torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        directory = get_local_model_directory(self.model_id)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(directory, local_files_only=True)
        clip = CLIPModel.from_pretrained(directory, local_files_only=True).to(self.device).eval()
        self.text_model = clip.text_model

    def _read_legacy(self, prompts: list[str]) -> dict[str, np.ndarray]:
        path = self.legacy_cache
        if path is None or not path.is_file() or not prompts:
            return {}
        encoder_key = f"{self.model_id}|padded-eos-cloud-v1|{EXPECTED_CONTEXT_SHAPE}"
        placeholders = ",".join("?" for _ in prompts)
        try:
            with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
                rows = connection.execute(
                    f"SELECT prompt, cloud, cloud_rows, dimension FROM prompt_encodings "
                    f"WHERE encoder_key=? AND prompt IN ({placeholders})",
                    (encoder_key, *prompts),
                ).fetchall()
        except sqlite3.Error:
            return {}
        found = {}
        for prompt, blob, length, dimension in rows:
            if dimension != EXPECTED_CONTEXT_SHAPE[1] or length < 1 or len(blob) != length * dimension * 4:
                continue
            cloud = np.frombuffer(blob, dtype="<f4").reshape(length, dimension)
            if np.isfinite(cloud).all():
                found[prompt] = cloud
        return found

    def clouds(self, prompts: list[str]) -> list[np.ndarray]:
        found = self._read_legacy(prompts)
        missing = [prompt for prompt in prompts if prompt not in found]
        if missing:
            self._ensure_model()
            tokens = self.tokenizer(
                missing, padding="max_length", truncation=True,
                max_length=EXPECTED_CONTEXT_SHAPE[0], return_tensors="pt",
            )
            eos = tokens["input_ids"].argmax(dim=1).tolist()
            inputs = {name: value.to(self.device) for name, value in tokens.items()
                      if name in ("input_ids", "attention_mask")}
            with self.torch.inference_mode():
                context = self.text_model(**inputs).last_hidden_state.float().cpu().numpy()
            for prompt, sequence, end in zip(missing, context, eos, strict=True):
                cloud = sequence[1:end] if end > 1 else sequence[end:end + 1]
                found[prompt] = np.ascontiguousarray(cloud, dtype=np.float32)
        return [found[prompt] for prompt in prompts]


class AdaptiveOrderInf:
    """Exact task-W1 selection followed by the research order-infinity score."""

    def __init__(self, region, indices: np.ndarray, clouds: list[np.ndarray]):
        if len(indices) < TASK_NEIGHBORS or len(indices) != len(clouds):
            raise ValueError("Too few task-reference clouds")
        self.region = region
        self.indices = indices
        self.clouds = clouds
        self.means = np.stack([cloud.mean(axis=0, dtype=np.float64) for cloud in clouds])
        self.mean_norms = np.einsum("ij,ij->i", self.means, self.means)

    def task_nearest(self, cloud: np.ndarray) -> np.ndarray:
        query_mean = cloud.mean(axis=0, dtype=np.float64)
        squared = self.mean_norms + query_mean @ query_mean - 2 * (self.means @ query_mean)
        lower_bounds = np.sqrt(np.maximum(squared, 0))
        best: list[tuple[float, int]] = []
        for index in np.argsort(lower_bounds):
            # The distance between cloud means is a lower bound on task W1.
            if len(best) == TASK_NEIGHBORS and lower_bounds[index] > best[-1][0] + 1e-4:
                break
            cost = task_wasserstein1(cloud, self.clouds[int(index)])
            best.append((cost, int(index)))
            best.sort()
            del best[TASK_NEIGHBORS:]
        return self.indices[[index for _, index in best]]

    def score(self, whitened: np.ndarray, cloud: np.ndarray) -> float:
        selected = self.task_nearest(cloud)
        return float(score_orderinf(whitened[None, :], self.region.anchors[selected], n_cuts=0)[0])


def _prepare_cache(path: Path, inputs: dict) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS anchor_clouds (
            anchor_index INTEGER PRIMARY KEY, cloud BLOB NOT NULL,
            n_tokens INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scores (
            prompt_key TEXT PRIMARY KEY, split TEXT NOT NULL,
            nearest REAL NOT NULL, order1 REAL NOT NULL, orderinf REAL NOT NULL
        );
    """)
    stored = connection.execute("SELECT value FROM metadata WHERE key='inputs'").fetchone()
    if stored is None:
        connection.execute("INSERT INTO metadata VALUES ('inputs', ?)", (json.dumps(inputs),))
        connection.commit()
    elif json.loads(stored[0]) != inputs:
        connection.close()
        raise ValueError("Transport cache was built from different inputs; choose a new output path")
    return connection


def _anchor_clouds(connection, reference, region, encoder) -> tuple[np.ndarray, list[np.ndarray]]:
    indices = np.arange(len(reference))
    if len(indices) < TASK_NEIGHBORS or len(reference) != len(region.anchors):
        raise ValueError("Reference prompts do not match region anchors")
    cached = {row[0] for row in connection.execute("SELECT anchor_index FROM anchor_clouds")}
    missing = [int(index) for index in indices if index not in cached]
    for start in range(0, len(missing), 32):
        batch = missing[start:start + 32]
        clouds = encoder.clouds(reference.iloc[batch]["prompt"].tolist())
        with connection:
            connection.executemany(
                "INSERT INTO anchor_clouds VALUES (?,?,?)",
                [(index, cloud.astype("<f4").tobytes(), len(cloud))
                 for index, cloud in zip(batch, clouds, strict=True)],
            )
    cached = {index: (blob, length) for index, blob, length in connection.execute(
        "SELECT anchor_index, cloud, n_tokens FROM anchor_clouds"
    )}
    clouds = [np.frombuffer(cached[int(index)][0], dtype="<f4").reshape(
        cached[int(index)][1], EXPECTED_CONTEXT_SHAPE[1]
    ) for index in indices]
    return indices, clouds


def _score_split(connection, frame, split: str, region, eos_cache: Path,
                 encoder: CloudEncoder, adaptive: AdaptiveOrderInf) -> None:
    total = len(frame)
    for start in range(0, total, 32):
        batch = frame.iloc[start:start + 32]
        keys = batch["prompt_key"].tolist()
        placeholders = ",".join("?" for _ in keys)
        present = {row[0] for row in connection.execute(
            f"SELECT prompt_key FROM scores WHERE prompt_key IN ({placeholders})", keys,
        )}
        missing = batch.loc[~batch["prompt_key"].isin(present)]
        if not missing.empty:
            vectors = _cached_vectors(eos_cache, region.metadata["model_id"], missing)
            whitened = region.whiten(vectors)
            nearest = region.score(vectors)
            order1 = _order1_scores(whitened, region.anchors)
            clouds = encoder.clouds(missing["prompt"].tolist())
            orderinf = [adaptive.score(vector, cloud)
                        for vector, cloud in zip(whitened, clouds, strict=True)]
            with connection:
                connection.executemany(
                    "INSERT INTO scores VALUES (?,?,?,?,?)",
                    [(key, split, float(s0), float(s1), float(sinf))
                     for key, s0, s1, sinf in zip(
                         missing["prompt_key"], nearest, order1, orderinf, strict=True
                     )],
                )
        if (start // 32) % 10 == 0 or start + len(batch) == total:
            print(f"{split}: {min(start + len(batch), total):,}/{total:,}", flush=True)


def _scores(connection, frame, split: str) -> np.ndarray:
    values = {key: (nearest, order1, orderinf) for key, nearest, order1, orderinf in connection.execute(
        "SELECT prompt_key, nearest, order1, orderinf FROM scores WHERE split=?", (split,)
    )}
    if len(values) != len(frame) or any(key not in values for key in frame["prompt_key"]):
        raise ValueError(f"Incomplete transport scores for {split}")
    return np.asarray([values[key] for key in frame["prompt_key"]], dtype=np.float64)


def _summary(values: np.ndarray, radius: float) -> dict:
    rejected = int((values > radius).sum())
    return {"count": len(values), "accepted": len(values) - rejected,
            "rejected": rejected, "acceptance_rate": (len(values) - rejected) / len(values)}


def run(
    *, corpus_path: Path = DEFAULT_CORPUS, region_path: Path = DEFAULT_REGION,
    eos_cache: Path = DEFAULT_CACHE, legacy_cache: Path | None = DEFAULT_VALIDATION_DATABASE_PATH,
    output_path: Path = DEFAULT_TRANSPORT, report_path: Path = DEFAULT_REPORT,
    device: str | None = None,
) -> dict:
    corpus = load_corpus(corpus_path)
    region = _load_region(region_path)
    corpus_hash = _source_sha256(corpus_path)
    if region.metadata.get("corpus_sha256") != corpus_hash:
        raise ValueError("Safe region does not match the prompt corpus")
    reference = corpus.loc[corpus["split"].eq("safe_reference")]
    inputs = {"version": 2, "corpus_sha256": corpus_hash,
              "region_sha256": _source_sha256(region_path),
              "model_id": region.metadata["model_id"], "task_reference_count": len(reference),
              "task_neighbors": TASK_NEIGHBORS,
              "task_selection_scope": "all_safe_reference",
              "n_cuts": 0, "miscoverage": region.metadata["miscoverage"]}
    connection = _prepare_cache(output_path, inputs)
    try:
        encoder = CloudEncoder(region.metadata["model_id"], legacy_cache, device)
        indices, clouds = _anchor_clouds(connection, reference, region, encoder)
        adaptive = AdaptiveOrderInf(region, indices, clouds)
        calibration = corpus.loc[corpus["split"].eq("safe_calibration")]
        _score_split(connection, calibration, "safe_calibration", region, eos_cache, encoder, adaptive)
        calibration_scores = _scores(connection, calibration, "safe_calibration")
        rank = math.ceil((len(calibration) + 1) * (1 - inputs["miscoverage"]))
        if rank > len(calibration):
            raise ValueError("Too few safe calibration prompts for finite transport budgets")
        radii = {"nearest_anchor": float(region.radius),
                 "order1": float(np.partition(calibration_scores[:, 1], rank - 1)[rank - 1]),
                 "orderinf": float(np.partition(calibration_scores[:, 2], rank - 1)[rank - 1])}
        report = {"inputs": inputs, "source": str(corpus_path.resolve()),
                  "region": str(region_path.resolve()), "transport": str(output_path.resolve()),
                  "calibration_rank": rank, "n_task_reference": len(indices),
                  "methods": {method: {"radius": radius, "splits": {}}
                              for method, radius in radii.items()}}
        for split in SPLITS:
            frame = corpus.loc[corpus["split"].eq(split)]
            if split != "safe_calibration":
                _score_split(connection, frame, split, region, eos_cache, encoder, adaptive)
            scores = _scores(connection, frame, split)
            for column_index, method in enumerate(METHOD_COLUMNS):
                report["methods"][method]["splits"][split] = _summary(
                    scores[:, column_index], radii[method]
                )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--region", type=Path, default=DEFAULT_REGION)
    parser.add_argument("--eos-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--legacy-cache", type=Path, default=DEFAULT_VALIDATION_DATABASE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_TRANSPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    result = run(corpus_path=args.corpus, region_path=args.region,
                 eos_cache=args.eos_cache, legacy_cache=args.legacy_cache,
                 output_path=args.output, report_path=args.report, device=args.device)
    for method, details in result["methods"].items():
        print(method, details["radius"], details["splits"])


if __name__ == "__main__":
    main()
