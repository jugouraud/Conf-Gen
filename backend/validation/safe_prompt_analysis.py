"""Read the finished T2I safe-region experiment for the analysis app."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from backend.embeddings.clip import EXPECTED_CONTEXT_SHAPE
from backend.embeddings.define_region import _load_region
from backend.embeddings.prompt_region import (
    DEFAULT_CORPUS,
    DEFAULT_REGION,
    DEFAULT_REPORT,
    _source_sha256,
    load_corpus,
)
from backend.paths import DATA_ROOT

DEFAULT_TRANSPORT = DATA_ROOT / "prompts" / "safe_prompt_transport_full_reference.sqlite3"
DEFAULT_METHOD_REPORT = DATA_ROOT / "prompts" / "safe_prompt_method_report_full_reference.json"
METHOD_COLUMNS = {"nearest_anchor": "nearest", "order1": "order1", "orderinf": "orderinf"}

EVALUATION_SPLITS = ("safe_test", "toxic_test_i2p", "toxic_test_t2i_risky")
SPLIT_LABELS = {
    "safe_test": "Held-out safe",
    "toxic_test_i2p": "I2P",
    "toxic_test_t2i_risky": "T2I-RiskyPrompt",
}


def _cached_vectors(path: Path, model_id: str, frame: pd.DataFrame) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt encoding cache not found: {path}")
    dimensions = EXPECTED_CONTEXT_SHAPE[1]
    result = np.empty((len(frame), dimensions), dtype=np.float32)
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        for start in range(0, len(frame), 400):
            keys = frame.iloc[start:start + 400]["prompt_key"].tolist()
            placeholders = ",".join("?" for _ in keys)
            values = dict(connection.execute(
                f"SELECT prompt_key, vector FROM eos WHERE model_id=? AND prompt_key IN ({placeholders})",
                (model_id, *keys),
            ))
            for offset, key in enumerate(keys):
                blob = values.get(key)
                if blob is None or len(blob) != dimensions * 4:
                    raise ValueError(
                        f"Missing cached EOS vector for prompt {key[:12]}; "
                        "rerun python -m backend.embeddings.prompt_region"
                    )
                vector = np.frombuffer(blob, dtype="<f4")
                if not np.isfinite(vector).all():
                    raise ValueError(f"Non-finite cached EOS vector for prompt {key[:12]}")
                result[start + offset] = vector
    return result


def _category(value: object, split: str) -> str:
    if split == "safe_test":
        return "Safe prompt"
    if value is None or pd.isna(value):
        return "Unspecified"
    if split == "toxic_test_t2i_risky":
        try:
            labels = json.loads(str(value))
        except (ValueError, TypeError):
            return "Unspecified"
        return ", ".join(labels) if isinstance(labels, dict) else "Unspecified"
    return str(value)


def _optional_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def load_safe_prompt_analysis(
    *, corpus_path: Path = DEFAULT_CORPUS, region_path: Path = DEFAULT_REGION,
    report_path: Path = DEFAULT_REPORT,
    transport_path: Path = DEFAULT_TRANSPORT,
    method_report_path: Path = DEFAULT_METHOD_REPORT,
) -> dict:
    """Load all three calibrated methods and verify every held-out decision."""
    corpus = load_corpus(corpus_path)
    region = _load_region(region_path)
    corpus_hash = _source_sha256(corpus_path)
    if region.metadata.get("corpus_sha256") != corpus_hash:
        raise ValueError("Saved safe region was fitted on a different corpus")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (Path(report.get("source", "")).resolve() != corpus_path.resolve() or
        Path(report.get("region", "")).resolve() != region_path.resolve() or
        not np.isclose(report.get("radius", np.nan), region.radius)):
        raise ValueError("Saved safe-region report does not match the corpus and region")
    if not method_report_path.is_file():
        raise FileNotFoundError(
            f"Full-reference method report not found: {method_report_path}. "
            "Run python -m backend.embeddings.prompt_transport to create it."
        )
    method_report = json.loads(method_report_path.read_text(encoding="utf-8"))
    expected_inputs = method_report.get("inputs", {})
    reference_count = int(corpus["split"].eq("safe_reference").sum())
    if (expected_inputs.get("version") != 2 or
        expected_inputs.get("task_selection_scope") != "all_safe_reference" or
        expected_inputs.get("task_reference_count") != reference_count or
        expected_inputs.get("task_neighbors") != 5 or
        expected_inputs.get("n_cuts") != 0 or
        expected_inputs.get("corpus_sha256") != corpus_hash or
        expected_inputs.get("region_sha256") != _source_sha256(region_path) or
        Path(method_report.get("transport", "")).resolve() != transport_path.resolve() or
        set(method_report.get("methods", {})) != set(METHOD_COLUMNS)):
        raise ValueError("Transport method report does not match the safe corpus and region")
    methods = method_report["methods"]
    if not np.isclose(methods["nearest_anchor"]["radius"], region.radius):
        raise ValueError("Nearest-anchor radius disagrees with the safe region")

    rows: list[dict] = []
    distributions: dict[str, dict[str, list[float]]] = {method: {} for method in METHOD_COLUMNS}
    with closing(sqlite3.connect(f"{transport_path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        inputs_row = connection.execute("SELECT value FROM metadata WHERE key='inputs'").fetchone()
        if inputs_row is None or json.loads(inputs_row[0]) != expected_inputs:
            raise ValueError("Transport score cache does not match its method report")
        for split in EVALUATION_SPLITS:
            frame = corpus.loc[corpus["split"].eq(split)]
            if frame.empty:
                continue
            cached = {key: (nearest, order1, orderinf)
                      for key, nearest, order1, orderinf in connection.execute(
                          "SELECT prompt_key, nearest, order1, orderinf FROM scores WHERE split=?", (split,)
                      )}
            if len(cached) != len(frame) or any(key not in cached for key in frame["prompt_key"]):
                raise ValueError(f"Incomplete transport score cache for {split}")
            values = np.asarray([cached[key] for key in frame["prompt_key"]], dtype=np.float64)
            for index, method in enumerate(METHOD_COLUMNS):
                rejected = values[:, index] > methods[method]["radius"]
                expected = methods[method]["splits"].get(split)
                if (expected is None or expected["count"] != len(frame) or
                    expected["rejected"] != int(rejected.sum())):
                    raise ValueError(f"Saved method report disagrees with cached scores for {split}/{method}")
                distributions[method][split] = values[:, index].tolist()
            nearest_rejected = values[:, 0] > region.radius
            old_expected = report["splits"].get(split)
            if (old_expected is None or old_expected["count"] != len(frame) or
                old_expected["rejected"] != int(nearest_rejected.sum())):
                raise ValueError(f"Saved safe-region report disagrees with cached scores for {split}")
            for record, scores in zip(frame.to_dict("records"), values, strict=True):
                method_scores = dict(zip(METHOD_COLUMNS, map(float, scores), strict=True))
                method_rejections = {method: score > methods[method]["radius"]
                                     for method, score in method_scores.items()}
                rows.append({
                    "row_number": len(rows) + 1,
                    "prompt": record["prompt"],
                    "split": split,
                    "source": SPLIT_LABELS[split],
                    "category": _category(record.get("category"), split),
                    "scores": method_scores,
                    "rejections": method_rejections,
                    "score": method_scores["orderinf"],
                    "rejected": method_rejections["orderinf"],
                    "prompt_toxicity": _optional_number(record.get("i2p_prompt_toxicity"))
                    if split == "toxic_test_i2p" else None,
                    "inappropriate_percentage": _optional_number(record.get("risk_score"))
                    if split == "toxic_test_i2p" else None,
                })
    if not rows:
        raise ValueError("No held-out prompts found in the corpus")
    return {
        "report": report,
        "method_report": method_report,
        "methods": methods,
        "rows": rows,
        "distributions": distributions,
        "safe_count": int(corpus["split"].str.startswith("safe_").sum()),
        "risky_count": int(corpus["split"].str.startswith("toxic_test_").sum()),
        "reference_count": int(corpus["split"].eq("safe_reference").sum()),
        "calibration_count": int(corpus["split"].eq("safe_calibration").sum()),
    }
