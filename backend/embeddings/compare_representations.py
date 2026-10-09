"""Select a CLIP prompt representation using held-out safe prompts only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.embeddings.clip import DEFAULT_MODEL_ID
from backend.embeddings.define_region import _load_region
from backend.embeddings.prompt_region import (
    DEFAULT_CACHE, DEFAULT_CORPUS, EosEncoder, _source_sha256, load_corpus, run,
    validate_representation,
)
from backend.paths import DATA_ROOT

DEFAULT_OUTPUT = DATA_ROOT / "prompts" / "representations"
DEFAULT_CANDIDATES = ("eos_final", "mean_final", "eos_layer_4", "eos_layer_8")


def benign_variants(frame: pd.DataFrame) -> pd.DataFrame:
    """Create fixed style, quality and clause-order changes from safe prompts."""
    rows = []
    for record in frame.itertuples(index=False):
        prompt = record.prompt
        variants = {
            "style": f"{prompt}, cinematic lighting, concept art",
            "quality": f"{prompt}, highly detailed, 8k, dramatic composition",
        }
        clauses = [part.strip() for part in prompt.split(",")]
        if len(clauses) > 1:
            variants["clause_order"] = ", ".join(clauses[1:] + clauses[:1])
        for kind, variant in variants.items():
            rows.append({
                "prompt_key": hashlib.sha256(variant.casefold().encode()).hexdigest(),
                "prompt": variant, "source_key": record.prompt_key, "change": kind,
            })
    return pd.DataFrame(rows)


def _absolute_correlation(scores: np.ndarray, lengths: np.ndarray) -> float:
    if np.std(scores) == 0 or np.std(lengths) == 0:
        return 0.0
    return float(abs(np.corrcoef(scores, lengths)[0, 1]))


def select_candidate(metrics: dict[str, dict], target_coverage: float) -> str:
    """Precommitted safe-only rank rule; toxic results never enter this function."""
    if not metrics:
        raise ValueError("No representation candidates")
    names = sorted(metrics)
    criteria = (
        lambda m: abs(m["safe_coverage"] - target_coverage),
        lambda m: m["median_normalized_score_shift"],
        lambda m: m["absolute_length_correlation"],
        lambda m: m["radius"],
    )
    totals = {name: 0 for name in names}
    for index, criterion in enumerate(criteria):
        values = {name: criterion(metrics[name]) for name in names}
        ranks = {value: rank for rank, value in enumerate(sorted(set(values.values())))}
        for name in names:
            totals[name] += ranks[values[name]] * (4 if index == 0 else 1)
    return min(names, key=lambda name: (totals[name], name))


def compare(
    corpus_path: Path = DEFAULT_CORPUS, cache_path: Path = DEFAULT_CACHE,
    output_dir: Path = DEFAULT_OUTPUT, *, candidates: tuple[str, ...] = DEFAULT_CANDIDATES,
    model_id: str = DEFAULT_MODEL_ID, device: str | None = None, batch_size: int = 32,
    m_pca: int = 32, miscoverage: float = 0.05, variant_count: int = 256,
) -> dict:
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("Candidates must be nonempty and unique")
    for name in candidates:
        validate_representation(name)
    if variant_count < 1:
        raise ValueError("variant_count must be positive")
    corpus = load_corpus(corpus_path)
    safe_test = corpus.loc[corpus["split"].eq("safe_test")].sort_values("prompt_key")
    if safe_test.empty:
        raise ValueError("A held-out safe_test split is required for selection")
    subset = safe_test.iloc[:variant_count]
    transformed = benign_variants(subset)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name in candidates:
        region_path = output_dir / f"{name}.npz"
        report_path = output_dir / f"{name}.json"
        safe_report = run(
            corpus_path, region_path, cache_path, report_path,
            model_id=model_id, device=device, miscoverage=miscoverage, m_pca=m_pca,
            batch_size=batch_size, representation=name, evaluate_toxic=False,
        )
        region = _load_region(region_path)
        encoder = EosEncoder(cache_path, model_id=model_id, device=device, representation=name)
        try:
            safe_scores = region.score(encoder.vectors(safe_test, batch_size=batch_size), batch_size=batch_size)
            changed_scores = region.score(
                encoder.vectors(transformed, batch_size=batch_size), batch_size=batch_size
            )
        finally:
            encoder.close()
        base_by_key = dict(zip(safe_test["prompt_key"], safe_scores, strict=True))
        shifts = np.asarray(
            [abs(float(score) - float(base_by_key[key])) / region.radius
             for score, key in zip(changed_scores, transformed["source_key"], strict=True)],
            dtype=np.float64,
        )
        by_change = {}
        for change in sorted(transformed["change"].unique()):
            values = shifts[transformed["change"].eq(change).to_numpy()]
            by_change[change] = {
                "count": len(values), "median_normalized_score_shift": float(np.median(values)),
                "p95_normalized_score_shift": float(np.quantile(values, 0.95)),
            }
        lengths = safe_test["prompt"].str.split().str.len().to_numpy()
        metrics[name] = {
            "safe_coverage": safe_report["splits"]["safe_test"]["acceptance_rate"],
            "radius": region.radius,
            "median_normalized_score_shift": float(np.median(shifts)),
            "transformations": by_change,
            "absolute_length_correlation": _absolute_correlation(safe_scores, lengths),
            "safe_test_count": len(safe_test), "variant_count": len(transformed),
            "region": str(region_path.resolve()),
        }
        print(name, metrics[name], flush=True)
    chosen = select_candidate(metrics, 1 - miscoverage)
    result = {
        "selection_basis": "safe_test only; toxic splits excluded until selection is fixed",
        "selection_rule": "Rank sum: coverage error x4, median shift, absolute length correlation, radius x1; alphabetical tie break",
        "corpus_sha256": _source_sha256(corpus_path), "candidates": metrics,
        "settings": {"model_id": model_id, "miscoverage": miscoverage,
                     "m_pca": m_pca, "batch_size": batch_size,
                     "safe_variant_prompt_count": len(subset)},
        "selected": chosen,
    }
    selection_path = output_dir / "selection.json"
    selection_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    final = run(
        corpus_path, output_dir / f"{chosen}.npz", cache_path, output_dir / f"{chosen}.json",
        model_id=model_id, device=device, miscoverage=miscoverage, m_pca=m_pca,
        batch_size=batch_size, representation=chosen, evaluate_toxic=True,
    )
    result["selected_final_report"] = final
    selection_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--m-pca", type=int, default=32)
    parser.add_argument("--miscoverage", type=float, default=0.05)
    parser.add_argument("--variant-count", type=int, default=256)
    args = parser.parse_args()
    compare(args.corpus, args.cache, args.output_dir, candidates=tuple(args.candidates),
            device=args.device, batch_size=args.batch_size, m_pca=args.m_pca,
            miscoverage=args.miscoverage, variant_count=args.variant_count)


if __name__ == "__main__":
    main()
