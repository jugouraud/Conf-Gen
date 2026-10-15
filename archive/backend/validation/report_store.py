"""Saved validation summaries and row-level CSV results for fast page loads."""

import csv
import json
import os
import tempfile
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from frontend.coco_region_validation import ValidationItem


_FLOAT_FIELDS = {
    "prompt_toxicity", "distance", "radius", "margin",
    "order1_cost", "order1_budget", "order1_margin",
    "orderinf_cost", "orderinf_budget", "orderinf_margin",
}
_BOOL_FIELDS = {"blocked", "order1_blocked", "orderinf_blocked"}


def toxicity_comparison(rows: list[dict], blocked_key: str) -> dict:
    """Summarize CSV toxicity scores for blocked and allowed rows."""
    blocked = [row["prompt_toxicity"] for row in rows if row[blocked_key]]
    allowed = [row["prompt_toxicity"] for row in rows if not row[blocked_key]]

    def summarize(values: list[float]) -> dict:
        return {"count": len(values), "mean": mean(values) if values else None,
                "median": median(values) if values else None}

    return {"blocked": summarize(blocked), "allowed": summarize(allowed),
            "mean_difference": mean(blocked) - mean(allowed) if blocked and allowed else None}


def source_signature(
    prompts_path: str | Path, region_path: str | Path, transport_path: str | Path | None,
) -> dict:
    """Invalidate saved decisions when their source data or decision artifacts change."""
    def signature(path: str | Path) -> dict:
        path = Path(path).resolve()
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    return {"prompts": signature(prompts_path), "region": signature(region_path),
            "transport": signature(transport_path) if transport_path is not None else None}


def summary_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_name(f"{output.stem}_summary.json")


def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         prefix=f".{path.stem}_", suffix=".tmp",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            write(handle)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def write_report_artifacts(report: dict, output_path: str | Path, signature: dict) -> None:
    """Write rows to CSV and the compact summary last as the completion marker."""
    output = Path(output_path)
    report["source_signature"] = signature
    rows = report["rows"]

    def write_csv(handle) -> None:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _atomic_write(output.with_suffix(".csv"), write_csv)
    _atomic_write(output, lambda handle: json.dump(report, handle, indent=2, ensure_ascii=False))
    compact = {key: value for key, value in report.items() if key != "rows"}
    _atomic_write(summary_path(output), lambda handle: json.dump(compact, handle, ensure_ascii=False))


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["row_number"] = int(row["row_number"])
        for key in _FLOAT_FIELDS & row.keys():
            row[key] = float(row[key])
        for key in _BOOL_FIELDS & row.keys():
            if row[key] not in {"True", "False"}:
                raise ValueError(f"Invalid saved decision {key} in {path}")
            row[key] = row[key] == "True"
    return rows


def _rows_match_items(rows: list[dict], items: list["ValidationItem"]) -> bool:
    return len(rows) == len(items) and all(
        row["row_number"] == item.row_number and row["prompt"] == item.prompt and
        row["category"] == item.categories and row.get("prompt_toxicity") == item.prompt_toxicity
        for row, item in zip(rows, items, strict=True)
    )


def load_saved_report(
    output_path: str | Path, items: list["ValidationItem"], *,
    prompts_path: str | Path, region_path: str | Path,
    transport_path: str | Path | None,
) -> dict | None:
    """Read a current summary and CSV; upgrade a compatible legacy full report."""
    output = Path(output_path)
    csv_path = output.with_suffix(".csv")
    current_signature = source_signature(prompts_path, region_path, transport_path)
    compact_path = summary_path(output)
    if compact_path.is_file() and csv_path.is_file():
        try:
            compact = json.loads(compact_path.read_text(encoding="utf-8"))
            if compact.get("source_signature") == current_signature:
                rows = _read_csv_rows(csv_path)
                if _rows_match_items(rows, items) and all(
                    "toxicity_comparison" in metric for metric in compact["metrics"].values()
                ):
                    return {**compact, "rows": rows}
        except (OSError, ValueError, KeyError, TypeError, csv.Error):
            pass

    if not output.is_file():
        return None
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("source_signature") is not None:
            return None
        input_mtimes = [part["mtime_ns"] for part in current_signature.values() if part is not None]
        if (output.stat().st_mtime_ns < max(input_mtimes) or
            report.get("prompt_source") != str(Path(prompts_path).resolve()) or
            report.get("region") != str(Path(region_path).resolve())):
            return None
        rows = report["rows"]
        if len(rows) != len(items) or any(
            row["row_number"] != item.row_number or row["prompt"] != item.prompt or
            row["category"] != item.categories
            for row, item in zip(rows, items, strict=True)
        ):
            return None
        if transport_path is not None and not {"order1", "orderinf"}.issubset(report["metrics"]):
            return None
        for row, item in zip(rows, items, strict=True):
            row["prompt_toxicity"] = item.prompt_toxicity
        by_number = {row["row_number"]: row["prompt_toxicity"] for row in rows}
        for collection in [report.get("closest_to_boundary", [])] + [
            metric.get("closest_to_boundary", []) for metric in report["metrics"].values()
        ]:
            for row in collection:
                row["prompt_toxicity"] = by_number[row["row_number"]]
        for metric in report["metrics"].values():
            metric["toxicity_comparison"] = toxicity_comparison(rows, metric["blocked_key"])
        write_report_artifacts(report, output, current_signature)
        return report
    except (OSError, ValueError, KeyError, TypeError, csv.Error):
        return None
