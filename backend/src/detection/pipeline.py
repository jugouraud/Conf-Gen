"""Compose detection stage APIs for resident inference or frozen-field rescoring."""

import argparse
import json
from pathlib import Path

import torch

from backend.src.detection.extract import run_detection_extraction
from backend.src.detection.measure import run_detection_measurement
from backend.src.detection.score import run_detection_scoring
from backend.src.fields.distances import PROBE_AGGREGATIONS


def run_detection_pipeline(
    output: Path,
    *,
    mode: str = "resident",
    source: Path | None = None,
    source_slice_source: Path | None = None,
    probe_aggregation: str = "rms",
    device: str = "cpu",
    working_directory: Path | str = "/var/tmp",
    wait_for_source: bool = False,
    report_path: Path | None = None,
    threads: int = 4,
) -> dict | list:
    """Compose owning stages without importing their private helpers."""
    if mode not in {"resident", "rescore", "controls"}:
        raise ValueError("Unknown detection pipeline mode")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be positive")
    output = Path(output).resolve()
    torch.set_num_threads(threads)
    if mode == "resident":
        run_detection_extraction(output, mode="resident", options={"threads": threads})
        return run_detection_scoring(output, mode="resident")
    if mode == "controls":
        run_detection_measurement(output, mode="controls", device=device)
        return run_detection_scoring(output, mode="controls")
    if source is None:
        raise ValueError("Rescoring requires a frozen source")
    sources = {
        "trajectory": Path(source),
        "source_slice": Path(source_slice_source)
        if source_slice_source is not None
        else Path(source) / "source_methods",
    }
    for space, origin in sources.items():
        origin = origin.resolve()
        if (
            output == origin
            or output in origin.parents
            or output.is_relative_to(origin / space)
        ):
            raise ValueError(
                "Rescoring output must be separate from the original run and signatures"
            )
    for space, origin in sources.items():
        run_detection_measurement(
            output / space,
            mode="probes",
            source=origin,
            space=space,
            device=device,
            options={
                "probe_aggregation": probe_aggregation,
                "working_directory": working_directory,
                "wait_for_source": wait_for_source and space == "source_slice",
            },
        )
    return run_detection_scoring(output, mode="probes", report_path=report_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-slice-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--probe-aggregation", choices=PROBE_AGGREGATIONS, default="rms"
    )
    parser.add_argument("--working-directory", type=Path, default=Path("/var/tmp"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--wait-for-source", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    result = run_detection_pipeline(
        args.output,
        mode="rescore",
        source=args.source,
        source_slice_source=args.source_slice_source,
        probe_aggregation=args.probe_aggregation,
        device=args.device,
        working_directory=args.working_directory,
        wait_for_source=args.wait_for_source,
        report_path=args.report,
        threads=args.threads,
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
