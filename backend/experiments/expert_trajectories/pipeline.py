"""Generate, independently validate, or summarize an offline expert dataset."""

import argparse
import json
import tomllib
from pathlib import Path

from backend.experiments.expert_trajectories.comparison import run_generation_comparison
from backend.experiments.expert_trajectories.scenes import run_scene_sampling
from backend.src.expert.audit import run_dataset_validation
from backend.src.expert.config import ExpertConfig
from backend.src.expert.pipeline import run_expert_pipeline
from backend.src.expert.storage import dataset_lock, read_record


def read_configuration(path: Path) -> tuple[ExpertConfig, dict]:
    with path.open("rb") as stream:
        data = stream.read(128 * 1024 + 1)
    if len(data) > 128 * 1024:
        raise ValueError("Configuration exceeds 128 KiB")
    record = tomllib.loads(data.decode("utf-8"))
    scenes = record.pop("scenes")
    return ExpertConfig.from_record(record), scenes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.toml")
    )
    generate.add_argument("--scenes", type=Path)
    generate.add_argument("--workers", type=int)
    generate.add_argument("--resume", action="store_true")
    compare = commands.add_parser("compare")
    compare.add_argument("--procedural", type=Path, required=True)
    compare.add_argument("--optimization", type=Path, required=True)
    for subcommand in (
        generate,
        commands.add_parser("validate"),
        commands.add_parser("summary"),
    ):
        subcommand.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        config, settings = read_configuration(args.config)
        scenes, source = run_scene_sampling(settings, config, args.scenes)
        report = run_expert_pipeline(
            config,
            scenes,
            args.output,
            resume=args.resume,
            workers=args.workers,
            scene_source=source,
        )
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "complete" else 1
    if args.command == "compare":
        print(json.dumps(run_generation_comparison(args.procedural, args.optimization), indent=2))
        return 0
    if not args.output.is_dir():
        raise ValueError("Dataset directory does not exist")
    with dataset_lock(args.output):
        validation = run_dataset_validation(args.output)
        result = (
            validation
            if args.command == "validate"
            else read_record(args.output / "summary.json")
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
