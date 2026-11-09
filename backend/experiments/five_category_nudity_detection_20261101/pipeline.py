"""Compose the five-category nudity detection stages through their public H1 APIs."""

import argparse
import traceback
from pathlib import Path

from backend.experiments.five_category_nudity_detection_20261101.package import (
    run_detection_packaging,
)
from backend.experiments.five_category_nudity_detection_20261101.prepare import (
    run_nudity_detection_preparation,
)
from backend.runtime.colab import run_colab_operation
from backend.runtime.detection import run_detection_worker
from backend.src.common.storage import write_json
from backend.src.detection.pipeline import main as rescore_detection
from backend.src.detection.score import run_detection_scoring


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "package",
            "detect",
            "score",
            "max",
            "colab-preflight",
            "colab-stage",
            "colab-launch",
            "colab-status",
            "colab-collect",
        ),
        default="prepare",
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.toml")
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "run")
    parser.add_argument("--working-directory", type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.stage == "prepare":
        return run_nudity_detection_preparation(args.config, output)
    if args.stage == "package":
        return run_detection_packaging(output)
    if args.stage.startswith("colab-"):
        return run_colab_operation(output, args.stage.removeprefix("colab-"))
    try:
        if args.stage == "detect":
            return run_detection_worker(["--output", str(output)])
        if args.stage == "score":
            return run_detection_scoring(output)
        return rescore_detection(
            [
                "--source",
                str(output),
                "--source-slice-source",
                str(output),
                "--output",
                str(output / "max_deviation"),
                "--probe-aggregation",
                "max",
                "--device",
                "cuda",
                "--working-directory",
                str(args.working_directory or output.parent / "scratch"),
                "--threads",
                "4",
            ]
        )
    except Exception:
        write_json(
            output / "failure.json",
            {"status": "failed", "error": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    print(main())
