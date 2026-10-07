"""Launch the COCO validation analysis dashboard."""

import argparse
from pathlib import Path
from threading import Lock

from nicegui import ui

from backend.embeddings.define_region import DEFAULT_DATABASE_PATH, DEFAULT_REGION_PATH
from backend.embeddings.project_region import DEFAULT_PROJECTION_PATH
from backend.embeddings.transport_budget import DEFAULT_TRANSPORT_PATH
from backend.validation.prompt_cache import DEFAULT_VALIDATION_DATABASE_PATH
from backend.validation.region_report import DEFAULT_REPORT_PATH, analyze_validation_region
from backend.validation.report_store import load_saved_report
from frontend.coco_region_analysis_page import create_coco_region_analysis_page
from frontend.coco_region_validation import DEFAULT_PROMPTS_PATH, load_validation_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze validation prompts against the COCO region.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--region", type=Path, default=DEFAULT_REGION_PATH)
    parser.add_argument("--projections", type=Path, default=DEFAULT_PROJECTION_PATH)
    parser.add_argument("--transport", type=Path, default=DEFAULT_TRANSPORT_PATH,
                        help="Cache for the calibrated transportation scores.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH,
                        help="CSV file containing a prompt column and optional categories column.")
    parser.add_argument("--validation-database", type=Path, default=DEFAULT_VALIDATION_DATABASE_PATH,
                        help="SQLite cache for validation prompt encodings.")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device for the live CLIP text model.")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    items = load_validation_items(args.prompts)
    report_lock = Lock()
    report: dict | None = None

    def saved_report() -> dict | None:
        nonlocal report
        with report_lock:
            if report is None:
                report = load_saved_report(
                    DEFAULT_REPORT_PATH, items, prompts_path=args.prompts,
                    region_path=args.region, transport_path=args.transport,
                )
            return report

    def rebuild_report() -> dict:
        nonlocal report
        with report_lock:
            report = analyze_validation_region(
                prompts_path=args.prompts, region_path=args.region,
                projection_path=args.projections, database_path=args.database,
                validation_database_path=args.validation_database,
                transport_path=args.transport, device=args.device, items=items,
            )
            return report

    @ui.page("/")
    def create_analysis_page() -> None:
        create_coco_region_analysis_page(rebuild_report, initial_report=saved_report())

    ui.run(title="Validation prompt analysis", port=args.port, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
