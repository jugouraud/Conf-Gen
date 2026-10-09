"""Fit the existing PPCA/conformal region on safe T2I prompts only.

Run after ``python -m scripts.build_prompt_dataset``. The saved region and
report live beside the centralized corpus in data/prompts/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from backend.embeddings.clip import DEFAULT_MODEL_ID, EXPECTED_CONTEXT_SHAPE
from backend.embeddings.define_region import (
    CocoRegion,
    _fit_whitener,
    _load_region,
    _nearest_distances,
)
from backend.paths import DATA_ROOT
from backend.storage.models import get_local_model_directory

DEFAULT_CORPUS = DATA_ROOT / "prompts" / "confgen_prompts.parquet"
DEFAULT_REGION = DATA_ROOT / "prompts" / "safe_prompt_region.npz"
DEFAULT_CACHE = DATA_ROOT / "prompts" / "prompt_eos.sqlite3"
DEFAULT_REPORT = DATA_ROOT / "prompts" / "safe_prompt_region_report.json"


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"prompt_key", "prompt", "split", "label", "eligible_for_region", "evaluation_only"}
    if not required.issubset(df):
        raise ValueError(f"Corpus is missing columns: {sorted(required - set(df))}")
    if df["prompt_key"].duplicated().any() and df.loc[df["split"].str.startswith("safe"), "prompt_key"].duplicated().any():
        raise ValueError("Safe prompts must be unique")
    safe_splits = {"safe_reference", "safe_calibration", "safe_test"}
    if not df.loc[df["split"].isin(safe_splits), "label"].eq("safe").all():
        raise ValueError("A safe split contains a non-safe label")
    expected_eligible = df["split"].isin({"safe_reference", "safe_calibration"})
    if not df["eligible_for_region"].eq(expected_eligible).all():
        raise ValueError("Region eligibility does not match safe splits")
    if not df["evaluation_only"].eq(~expected_eligible).all():
        raise ValueError("Evaluation-only flags do not match safe splits")
    if set(df.loc[df["split"].isin(safe_splits), "prompt_key"]) & set(
        df.loc[~df["split"].isin(safe_splits), "prompt_key"]
    ):
        raise ValueError("Safe and risky prompts overlap")
    return df


def validate_representation(representation: str) -> None:
    if representation not in {"eos_final", "mean_final"} and not re.fullmatch(r"eos_layer_[1-9][0-9]*", representation):
        raise ValueError(f"Unknown CLIP representation: {representation}")


class EosEncoder:
    def __init__(self, cache_path: Path, *, model_id: str, device: str | None,
                 representation: str = "eos_final"):
        validate_representation(representation)
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(cache_path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS eos (model_id TEXT NOT NULL, prompt_key TEXT NOT NULL, "
            "vector BLOB NOT NULL, PRIMARY KEY(model_id, prompt_key))"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS representations (model_id TEXT NOT NULL, "
            "representation TEXT NOT NULL, prompt_key TEXT NOT NULL, vector BLOB NOT NULL, "
            "PRIMARY KEY(model_id, representation, prompt_key))"
        )
        self.model_id = model_id
        self.representation = representation
        self.device_name = device
        self.model = None

    def close(self) -> None:
        self.connection.close()

    def _load_model(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPTokenizerFast

        self.torch = torch
        self.device = torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        model_path = get_local_model_directory(self.model_id)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_path, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(self.device).eval().text_model
        if self.representation.startswith("eos_layer_"):
            layer = int(self.representation.removeprefix("eos_layer_"))
            n_layers = self.model.config.num_hidden_layers
            if layer >= n_layers:
                raise ValueError(f"Intermediate layer must be between 1 and {n_layers - 1}")

    def _encode(self, prompts: list[str]) -> np.ndarray:
        self._load_model()
        tokens = self.tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=EXPECTED_CONTEXT_SHAPE[0], return_tensors="pt",
        )
        eos = tokens["input_ids"].argmax(dim=1).to(self.device)
        inputs = {name: value.to(self.device) for name, value in tokens.items()
                  if name in ("input_ids", "attention_mask")}
        with self.torch.inference_mode():
            output = self.model(
                **inputs, output_hidden_states=self.representation.startswith("eos_layer_")
            )
            context = (output.hidden_states[int(self.representation.removeprefix("eos_layer_"))]
                       if self.representation.startswith("eos_layer_") else output.last_hidden_state)
            if self.representation == "mean_final":
                mask = inputs["attention_mask"].unsqueeze(-1)
                vectors = (context * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                vectors = context[self.torch.arange(len(prompts), device=self.device), eos]
        result = vectors.float().cpu().numpy()
        if result.shape != (len(prompts), EXPECTED_CONTEXT_SHAPE[1]) or not np.isfinite(result).all():
            raise ValueError("CLIP returned invalid prompt embeddings")
        return result

    def vectors(self, frame: pd.DataFrame, *, batch_size: int = 32) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        result = np.empty((len(frame), EXPECTED_CONTEXT_SHAPE[1]), dtype=np.float32)
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start:start + batch_size]
            keys = batch["prompt_key"].tolist()
            placeholders = ",".join("?" for _ in keys)
            if self.representation == "eos_final":
                query = f"SELECT prompt_key, vector FROM eos WHERE model_id=? AND prompt_key IN ({placeholders})"
                parameters = (self.model_id, *keys)
            else:
                query = ("SELECT prompt_key, vector FROM representations WHERE model_id=? "
                         f"AND representation=? AND prompt_key IN ({placeholders})")
                parameters = (self.model_id, self.representation, *keys)
            cached = dict(self.connection.execute(query, parameters))
            missing = [index for index, key in enumerate(keys)
                       if key not in cached or len(cached[key]) != EXPECTED_CONTEXT_SHAPE[1] * 4
                       or not np.isfinite(np.frombuffer(cached[key], dtype="<f4")).all()]
            if missing:
                prompts = [batch.iloc[index]["prompt"] for index in missing]
                values = self._encode(prompts)
                with self.connection:
                    if self.representation == "eos_final":
                        self.connection.executemany(
                            "INSERT OR REPLACE INTO eos(model_id,prompt_key,vector) VALUES (?,?,?)",
                            [(self.model_id, keys[index], vector.astype("<f4").tobytes())
                             for index, vector in zip(missing, values, strict=True)],
                        )
                    else:
                        self.connection.executemany(
                            "INSERT OR REPLACE INTO representations(model_id,representation,prompt_key,vector) "
                            "VALUES (?,?,?,?)",
                            [(self.model_id, self.representation, keys[index], vector.astype("<f4").tobytes())
                             for index, vector in zip(missing, values, strict=True)],
                        )
                for index, vector in zip(missing, values, strict=True):
                    cached[keys[index]] = vector.astype("<f4").tobytes()
            for index, key in enumerate(keys):
                result[start + index] = np.frombuffer(cached[key], dtype="<f4")
            if (start // batch_size) % 20 == 0 or start + len(batch) == len(frame):
                print(f"Encoded/cached {start + len(batch):,}/{len(frame):,} prompts", flush=True)
        return result


def fit_region(
    reference: np.ndarray, calibration: np.ndarray, *, miscoverage: float,
    m_pca: int, batch_size: int, metadata: dict,
) -> CocoRegion:
    if not 0 < miscoverage < 1:
        raise ValueError("Miscoverage must be between zero and one")
    rank = math.ceil((len(calibration) + 1) * (1 - miscoverage))
    if rank > len(calibration):
        raise ValueError("Too few calibration prompts for finite conformal radius")
    mean, components, eigenvalues, gamma = _fit_whitener(reference, m_pca)
    region = CocoRegion(
        anchors=np.empty((0, reference.shape[1]), dtype=np.float32),
        mean=mean, components=components, eigenvalues=eigenvalues,
        gamma=gamma, radius=np.inf, metadata={},
    )
    region.anchors = region.whiten(reference).astype(np.float32)
    scores = _nearest_distances(region.whiten(calibration), region.anchors, batch_size=batch_size)
    region.radius = float(np.partition(scores, rank - 1)[rank - 1])
    region.metadata = {
        **metadata, "n_reference": len(reference), "n_calibration": len(calibration),
        "calibration_rank": rank,
        "score": ("nearest_mahalanobis_eos" if metadata.get("representation", "eos_final") == "eos_final"
                  else "nearest_mahalanobis_prompt"),
    }
    return region


def run(
    corpus_path: Path = DEFAULT_CORPUS, region_path: Path = DEFAULT_REGION,
    cache_path: Path = DEFAULT_CACHE, report_path: Path = DEFAULT_REPORT,
    *, model_id: str = DEFAULT_MODEL_ID, device: str | None = None,
    miscoverage: float = 0.05, m_pca: int = 32, batch_size: int = 32,
    representation: str = "eos_final", evaluate_toxic: bool = True,
) -> dict:
    validate_representation(representation)
    corpus = load_corpus(corpus_path)
    source_hash = _source_sha256(corpus_path)
    settings = {
        "version": 1, "corpus_sha256": source_hash, "model_id": model_id,
        "miscoverage": miscoverage, "m_pca": m_pca,
    }
    if representation != "eos_final":
        settings.update(version=2, representation=representation)
    encoder = EosEncoder(cache_path, model_id=model_id, device=device,
                         representation=representation)
    try:
        if region_path.is_file():
            region = _load_region(region_path)
            if any(region.metadata.get(key) != value for key, value in settings.items()):
                raise ValueError("Saved region has different corpus or settings; choose a new path or remove it")
        else:
            reference_rows = corpus.loc[corpus["split"].eq("safe_reference")]
            calibration_rows = corpus.loc[corpus["split"].eq("safe_calibration")]
            if reference_rows.empty or calibration_rows.empty:
                raise ValueError("Both reference and calibration splits must contain prompts")
            reference = encoder.vectors(reference_rows, batch_size=batch_size)
            calibration = encoder.vectors(calibration_rows, batch_size=batch_size)
            region = fit_region(
                reference, calibration, miscoverage=miscoverage, m_pca=m_pca,
                batch_size=batch_size, metadata=settings,
            )
            region_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=region_path.parent, suffix=".npz", delete=False) as temp:
                temp_path = Path(temp.name)
            try:
                np.savez_compressed(
                    temp_path, anchors=region.anchors, mean=region.mean,
                    components=region.components, eigenvalues=region.eigenvalues,
                    gamma=region.gamma, radius=region.radius,
                    metadata=json.dumps(region.metadata),
                )
                os.replace(temp_path, region_path)
            finally:
                temp_path.unlink(missing_ok=True)
            print(f"Saved safe-only region: {region_path}", flush=True)
        report = {"region": str(region_path.resolve()), "radius": region.radius,
                  "source": str(corpus_path.resolve()), "representation": representation,
                  "splits": {}}
        splits = ["safe_calibration", "safe_test"]
        if evaluate_toxic:
            splits += ["toxic_test_i2p", "toxic_test_t2i_risky"]
        for split in splits:
            rows = corpus.loc[corpus["split"].eq(split)]
            if rows.empty:
                continue
            vectors = encoder.vectors(rows, batch_size=batch_size)
            scores = region.score(vectors, batch_size=batch_size)
            report["splits"][split] = {
                "count": len(rows), "accepted": int((scores <= region.radius).sum()),
                "rejected": int((scores > region.radius).sum()),
                "acceptance_rate": float(np.mean(scores <= region.radius)),
            }
            print(split, report["splits"][split], flush=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        encoder.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--region", type=Path, default=DEFAULT_REGION)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--miscoverage", type=float, default=0.05)
    parser.add_argument("--m-pca", type=int, default=32)
    args = parser.parse_args()
    run(args.corpus, args.region, args.cache, args.report, device=args.device,
        batch_size=args.batch_size, miscoverage=args.miscoverage, m_pca=args.m_pca)


if __name__ == "__main__":
    main()
