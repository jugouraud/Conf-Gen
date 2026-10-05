"""Launch the separate NiceGUI COCO region explorer."""

import argparse
from pathlib import Path

from nicegui import ui

from backend.embeddings.define_region import DEFAULT_DATABASE_PATH, DEFAULT_REGION_PATH
from backend.embeddings.project_region import DEFAULT_PROJECTION_PATH
from frontend.coco_region_page import create_coco_region_page
from frontend.coco_region_plots import CocoRegionVisualizationController, load_region_visualization_data
from frontend.coco_region_validation import DEFAULT_VALIDATION_DIRECTORY, LiveCocoRegionValidator, load_validation_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore the saved COCO safe region in 3D.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--region", type=Path, default=DEFAULT_REGION_PATH)
    parser.add_argument("--projections", type=Path, default=DEFAULT_PROJECTION_PATH)
    parser.add_argument("--anchors", type=int, default=1200, help="Reference anchors to show (default: 1200).")
    parser.add_argument("--calibration", type=int, default=200, help="Held-out captions to show (default: 200).")
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIRECTORY)
    parser.add_argument("--prompts", type=Path, help="JSON mapping of validation image names to prompts or dataset records.")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device for the live CLIP text model.")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    data = load_region_visualization_data(
        args.database, args.region, args.projections,
        max_anchors=args.anchors, max_calibration=args.calibration,
    )
    items = load_validation_items(args.validation_dir, args.prompts)
    validator = LiveCocoRegionValidator(args.region, args.projections, device=args.device)

    def create_page() -> None:
        create_coco_region_page(CocoRegionVisualizationController(data), validator, items)

    ui.run(root=create_page, title="COCO safe region", port=args.port, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
