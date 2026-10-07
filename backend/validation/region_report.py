"""Score validation CSV prompts against the saved COCO region."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.embeddings.define_region import DEFAULT_DATABASE_PATH, DEFAULT_REGION_PATH
from backend.embeddings.project_region import DEFAULT_PROJECTION_PATH
from backend.embeddings.transport_budget import DEFAULT_TRANSPORT_PATH
from backend.paths import VALIDATION_ROOT
from backend.validation.prompt_cache import DEFAULT_VALIDATION_DATABASE_PATH
from backend.validation.report_store import source_signature, toxicity_comparison, write_report_artifacts
from frontend.coco_region_validation import (
    DEFAULT_PROMPTS_PATH,
    LiveCocoRegionValidator,
    ValidationItem,
    load_validation_items,
)

DEFAULT_REPORT_PATH = VALIDATION_ROOT / "runs" / "coco_region_blocking.json"


def _statistics(rows: list[dict], score_key: str = "distance", blocked_key: str = "blocked") -> dict:
    scores = np.asarray([row[score_key] for row in rows], dtype=np.float64)
    blocked = sum(row[blocked_key] for row in rows)
    return {
        "count": len(rows), "blocked": blocked, "allowed": len(rows) - blocked,
        "block_rate": blocked / len(rows),
        "distance_min": float(scores.min()),
        "distance_p10": float(np.quantile(scores, 0.10)),
        "distance_median": float(np.median(scores)),
        "distance_p90": float(np.quantile(scores, 0.90)),
        "distance_max": float(scores.max()),
    }


def _group_statistics(
    rows: list[dict], key: str, score_key: str = "distance", blocked_key: str = "blocked",
) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return [{"category": category, **_statistics(group, score_key, blocked_key)}
            for category, group in sorted(groups.items())]


def _metric_report(
    rows: list[dict], unique_rows: list[dict], *, score_key: str,
    budget_key: str, blocked_key: str,
) -> dict:
    margin_key = "margin" if score_key == "distance" else score_key.replace("_cost", "_margin")
    return {
        "budget": rows[0][budget_key],
        "score_key": score_key,
        "blocked_key": blocked_key,
        "prompt_level": _statistics(rows, score_key, blocked_key),
        "unique_prompt_level": _statistics(unique_rows, score_key, blocked_key),
        "toxicity_comparison": toxicity_comparison(rows, blocked_key),
        "by_category": _group_statistics(rows, "category", score_key, blocked_key),
        "closest_to_boundary": sorted(rows, key=lambda row: abs(row[margin_key]))[:12],
    }


def analyze_validation_region(
    *, prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
    region_path: str | Path = DEFAULT_REGION_PATH,
    projection_path: str | Path = DEFAULT_PROJECTION_PATH,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    validation_database_path: str | Path = DEFAULT_VALIDATION_DATABASE_PATH,
    transport_path: str | Path | None = DEFAULT_TRANSPORT_PATH,
    output_path: str | Path | None = DEFAULT_REPORT_PATH,
    device: str | None = None,
    batch_size: int = 16,
    validator: LiveCocoRegionValidator | None = None,
    items: list[ValidationItem] | None = None,
) -> dict:
    """Analyze all CSV prompts with each available decision method."""
    prompts_path = Path(prompts_path)
    items = items if items is not None else load_validation_items(prompts_path)
    unique_prompts = list(dict.fromkeys(item.prompt for item in items))
    print(f"Loading cached or new encodings for {len(unique_prompts)} distinct prompts "
          f"across {len(items)} rows...", flush=True)
    validator = validator or LiveCocoRegionValidator(
        region_path, projection_path, transport_path=transport_path,
        database_path=database_path, validation_database_path=validation_database_path,
        device=device,
    )
    vectors, clouds = validator.encode_prompts_with_clouds(unique_prompts, batch_size=batch_size)
    scores = validator.region.score(vectors)
    score_by_prompt = dict(zip(unique_prompts, map(float, scores), strict=True))
    transport_by_prompt = {}
    if validator.transport is not None:
        whitened = validator.region.whiten(vectors)
        for prompt, vector, cloud in zip(unique_prompts, whitened, clouds, strict=True):
            transport_by_prompt[prompt] = validator.transport.score(vector, cloud)
    radius = float(validator.region.radius)
    rows = []
    for item in items:
        score = score_by_prompt[item.prompt]
        row = {
            "row_number": item.row_number,
            "prompt": item.prompt,
            "category": item.categories,
            "prompt_toxicity": item.prompt_toxicity,
            "distance": score,
            "radius": radius,
            "margin": score - radius,
            "blocked": score > radius,
        }
        if transport_by_prompt:
            order1, orderinf = transport_by_prompt[item.prompt]
            order1_budget = validator.transport.summary["order1_radius"]
            orderinf_budget = validator.transport.summary["orderinf_radius"]
            row.update({
                "order1_cost": order1, "order1_budget": order1_budget,
                "order1_margin": order1 - order1_budget, "order1_blocked": order1 > order1_budget,
                "orderinf_cost": orderinf, "orderinf_budget": orderinf_budget,
                "orderinf_margin": orderinf - orderinf_budget, "orderinf_blocked": orderinf > orderinf_budget,
            })
        rows.append(row)
    first_row_by_prompt = {}
    for row in rows:
        first_row_by_prompt.setdefault(row["prompt"], row)
    unique_rows = list(first_row_by_prompt.values())
    metrics = {"nearest_anchor": _metric_report(
        rows, unique_rows, score_key="distance", budget_key="radius", blocked_key="blocked",
    )}
    metrics["nearest_anchor"]["reference_count"] = len(validator.region.anchors)
    if transport_by_prompt:
        metrics["order1"] = _metric_report(
            rows, unique_rows, score_key="order1_cost", budget_key="order1_budget",
            blocked_key="order1_blocked",
        )
        metrics["orderinf"] = _metric_report(
            rows, unique_rows, score_key="orderinf_cost", budget_key="orderinf_budget",
            blocked_key="orderinf_blocked",
        )
        metrics["order1"]["reference_count"] = len(validator.region.anchors)
        metrics["orderinf"]["reference_count"] = validator.transport.summary["n_task_reference"]
    report = {
        "metric": "full_768d_nearest_anchor_distance_of_clip_text_eos",
        "decision_input": "prompt_text_not_image_pixels",
        "region": str(Path(region_path).resolve()),
        "prompt_source": str(prompts_path.resolve()),
        "radius": radius,
        "metrics": metrics,
        "prompt_level": _statistics(rows),
        "unique_prompt_level": _statistics(unique_rows),
        "by_category": _group_statistics(rows, "category"),
        "closest_to_boundary": sorted(rows, key=lambda row: abs(row["margin"]))[:12],
        "rows": rows,
    }
    if output_path is not None:
        output = Path(output_path)
        write_report_artifacts(report, output, source_signature(
            prompts_path, region_path, transport_path,
        ))
    for method, metric in metrics.items():
        summary = metric["prompt_level"]
        print(f"{method}: {summary['blocked']}/{len(rows)} prompts blocked "
              f"({summary['block_rate']:.1%})")
    print(f"Unique prompts: {report['unique_prompt_level']['blocked']}/{len(unique_rows)} blocked "
          f"({report['unique_prompt_level']['block_rate']:.1%})")
    if output_path is not None:
        print(f"Saved {output}, {output.with_suffix('.csv')}, and compact summary")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze COCO-region blocking of validation CSV prompts.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--region", type=Path, default=DEFAULT_REGION_PATH)
    parser.add_argument("--projections", type=Path, default=DEFAULT_PROJECTION_PATH)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--validation-database", type=Path, default=DEFAULT_VALIDATION_DATABASE_PATH)
    parser.add_argument("--transport", type=Path, default=DEFAULT_TRANSPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    analyze_validation_region(
        prompts_path=args.prompts,
        region_path=args.region, projection_path=args.projections,
        database_path=args.database, validation_database_path=args.validation_database,
        transport_path=args.transport,
        output_path=args.output, device=args.device, batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
