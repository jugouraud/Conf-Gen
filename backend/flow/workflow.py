"""Run the full safe-flow experiment or one resumable Colab stage."""

import argparse
from pathlib import Path

from backend.paths import VALIDATION_ROOT
from scripts.build_prompt_dataset import build_corpus
from .calibration import calibrate
from .common import ROOT, load_config
from .corpus import prepare
from .evaluation import evaluate
from .features import extract
from .probes import freeze_probes
from .report import report

STAGES = ("all", "probes", "reference", "calibration", "safe-test", "unsafe-test", "report")


# Hierarchy 3: atomic actions.

def _default_dataset_options(cfg):
    corpus = Path(cfg["paths"]["corpus"])
    if corpus.name != "confgen_prompts.parquet":
        raise ValueError("Dataset builder requires a corpus named confgen_prompts.parquet")
    output_dir = corpus.parent if corpus.is_absolute() else ROOT / corpus.parent
    return argparse.Namespace(
        output_dir=output_dir, diffusiondb_metadata=None,
        i2p_csv=VALIDATION_ROOT / "i2p_benchmark.csv",
        t2i_risky_json=None, safe_size=10_000, eval_size=3_000,
        prompt_nsfw_max=0.01, image_nsfw_max=0.05,
        min_words=3, max_words=60, skip_t2i_risky=False,
    )


# Hierarchy 1: workflow orchestration.

def run_complete_workflow(cfg):
    """Run every experiment step in dependency order from a prompt corpus."""
    prepare(cfg)
    freeze_probes(cfg)
    extract(cfg, "safe_reference")
    extract(cfg, "safe_validation")
    calibrate(cfg)
    extract(cfg, "safe_test")
    evaluate(cfg, "safe_test")
    extract(cfg, "unsafe_test")
    evaluate(cfg, "unsafe_test")
    report(cfg)


def run_from_sources(cfg, dataset_options):
    """Build the prompt corpus, then run every experiment step."""
    build_corpus(dataset_options)
    run_complete_workflow(cfg)


def run_stage(cfg, stage):
    """Run a complete workflow or one resumable stage."""
    if stage not in STAGES:
        raise ValueError(f"Unknown workflow stage: {stage}")
    if stage == "all":
        run_complete_workflow(cfg)
    elif stage == "probes":
        prepare(cfg)
        freeze_probes(cfg)
    elif stage == "reference":
        extract(cfg, "safe_reference")
    elif stage == "calibration":
        extract(cfg, "safe_validation")
        calibrate(cfg)
    elif stage == "safe-test":
        extract(cfg, "safe_test")
        evaluate(cfg, "safe_test")
    elif stage == "unsafe-test":
        extract(cfg, "unsafe_test")
        evaluate(cfg, "unsafe_test")
    else:
        report(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "safe_flow.toml")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--build-dataset", action="store_true",
                        help="Build the default prompt corpus before the full workflow")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.corpus is not None:
        cfg["paths"]["corpus"] = str(args.corpus.resolve())
    if args.output is not None:
        cfg["paths"]["output"] = str(args.output.resolve())
    if args.build_dataset:
        if args.stage != "all":
            parser.error("--build-dataset requires --stage all")
        run_from_sources(cfg, _default_dataset_options(cfg))
    else:
        run_stage(cfg, args.stage)


if __name__ == "__main__":
    main()
