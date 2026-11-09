"""CUDA worker entry point for a prepared resident paired-detection experiment."""

import argparse
import traceback
from pathlib import Path

from backend.src.common.storage import read_json, write_json
from backend.src.detection.pipeline import main as rescore_detection
from backend.src.detection.pipeline import run_detection_pipeline


def run_detection_worker(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = read_json(args.output / "experiment.json")
        aggregations = (
            manifest["config"].get("execution", {}).get("probe_aggregations", ["rms"])
        )
        if aggregations not in (["rms"], ["rms", "max"]):
            raise ValueError("Resident worker supports rms or rms followed by max")
        result = run_detection_pipeline(args.output)
        if "max" in aggregations:
            write_json(
                args.output / "status.json",
                {"status": "running", "phase": "max_deviation"},
            )
            rescore_detection(
                [
                    "--source",
                    str(args.output),
                    "--source-slice-source",
                    str(args.output),
                    "--output",
                    str(args.output / "max_deviation"),
                    "--probe-aggregation",
                    "max",
                    "--device",
                    "cuda",
                    "--working-directory",
                    "/var/tmp",
                    "--threads",
                    str(manifest["config"]["experiment"].get("threads", 4)),
                ]
            )
            write_json(
                args.output / "status.json",
                {"status": "complete", "phase": "rms_and_max_complete"},
            )
        return result
    except BaseException:
        write_json(
            args.output / "failure.json",
            {"status": "failed", "error": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    run_detection_worker()
