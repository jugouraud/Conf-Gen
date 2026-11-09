"""Prepare, launch, monitor, or execute the twenty-pair repair experiment."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "launch", "monitor", "worker"), default="prepare"
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.toml")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("paired", "controls"), default="paired")
    args = parser.parse_args(argv)
    if args.output is None:
        from backend.src.common.storage import read_toml_config, relocate_path

        args.output = relocate_path(read_toml_config(args.config)["paths"]["output"])
    if args.stage == "prepare":
        from backend.src.repair.prepare import run_repair_preparation

        return run_repair_preparation(
            args.config, args.output.resolve(), mode=args.mode
        ).summary
    if args.stage == "worker":
        from backend.src.common.storage import read_json

        optimized = read_json(args.output / "experiment.json")["fingerprint"][
            "settings"
        ]["execution"].get("optimized_gpu", False)
        if optimized:
            from backend.runtime.repair_cuda import run_worker
        else:
            from backend.runtime.repair_worker import run_worker

        return run_worker(args.output.resolve())
    from backend.runtime.repair_session import launch, monitor

    return (launch if args.stage == "launch" else monitor)(args.output.resolve())


if __name__ == "__main__":
    print(main())
