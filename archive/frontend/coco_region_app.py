"""Launch the T2I safe-region dashboard and the COCO baseline view."""

import argparse
from pathlib import Path
from threading import Lock

from nicegui import ui

from backend.embeddings.define_region import DEFAULT_DATABASE_PATH, DEFAULT_REGION_PATH
from backend.embeddings.project_region import DEFAULT_PROJECTION_PATH
from backend.embeddings.prompt_region import (
    DEFAULT_CORPUS as DEFAULT_SAFE_CORPUS,
)
from backend.embeddings.prompt_region import (
    DEFAULT_REGION as DEFAULT_SAFE_REGION,
)
from backend.embeddings.prompt_region import (
    DEFAULT_REPORT as DEFAULT_SAFE_REPORT,
)
from backend.embeddings.transport_budget import DEFAULT_TRANSPORT_PATH
from backend.validation.prompt_cache import DEFAULT_VALIDATION_DATABASE_PATH
from backend.validation.region_report import (
    DEFAULT_REPORT_PATH,
    analyze_validation_region,
)
from backend.validation.report_store import load_saved_report
from backend.validation.safe_prompt_analysis import (
    DEFAULT_METHOD_REPORT,
    load_safe_prompt_analysis,
)
from backend.validation.safe_prompt_analysis import (
    DEFAULT_TRANSPORT as DEFAULT_SAFE_TRANSPORT,
)
from frontend.coco_region_analysis_page import create_coco_region_analysis_page
from frontend.coco_region_validation import DEFAULT_PROMPTS_PATH, load_validation_items
from frontend.safe_prompt_analysis_page import create_safe_prompt_analysis_page


def main() -> None:
    parser = argparse.ArgumentParser(description="View the T2I safe region and COCO baseline analysis.")
    parser.add_argument("--safe-corpus", type=Path, default=DEFAULT_SAFE_CORPUS)
    parser.add_argument("--safe-region", type=Path, default=DEFAULT_SAFE_REGION)
    parser.add_argument("--safe-report", type=Path, default=DEFAULT_SAFE_REPORT)
    parser.add_argument("--safe-transport", type=Path, default=DEFAULT_SAFE_TRANSPORT)
    parser.add_argument("--safe-method-report", type=Path, default=DEFAULT_METHOD_REPORT)
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

    safe_analysis = load_safe_prompt_analysis(
        corpus_path=args.safe_corpus, region_path=args.safe_region,
        report_path=args.safe_report, transport_path=args.safe_transport,
        method_report_path=args.safe_method_report,
    )
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
    def create_safe_analysis_page() -> None:
        create_safe_prompt_analysis_page(safe_analysis)

    @ui.page("/coco")
    def create_baseline_page() -> None:
        create_coco_region_analysis_page(
            rebuild_report, initial_report=saved_report(), back_path="/",
        )

    ui.run(title="T2I safe-region analysis", port=args.port, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
