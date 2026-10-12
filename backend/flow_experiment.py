"""CLI and compatibility imports for the fixed-probe Sana experiment."""

import argparse
from pathlib import Path

from backend.flow.common import ROOT, SPLITS, load_config, resolved_path, sha256, write_json
from backend.flow.corpus import canonical, limited_rows, prepare, read_rows, split_corpus
from backend.flow.features import extract, feature_file
from backend.flow.calibration import calibrate
from backend.flow.evaluation import evaluate
from backend.flow.metrics import flat_feature, nearest
from backend.flow.model import SanaAdapter
from backend.flow.probes import bank_file, freeze_probes, read_bank
from backend.flow.report import report
from backend.flow.workflow import run_complete_workflow


# Hierarchy 1: workflow orchestration.

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "prepare", "probes", "extract", "calibrate", "evaluate", "report"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "safe_flow.toml")
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--limit", type=int, help="Temporary extraction cap, must match configured scoring caps")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare":
        prepare(cfg)
    elif args.command == "probes":
        freeze_probes(cfg)
    elif args.command == "extract":
        if args.split is None:
            parser.error("extract requires --split")
        extract(cfg, args.split, args.limit)
    elif args.command == "calibrate":
        calibrate(cfg)
    elif args.command == "evaluate":
        if args.split not in ("safe_test", "unsafe_test"):
            parser.error("evaluate requires --split safe_test or unsafe_test")
        evaluate(cfg, args.split)
    else:
        report(cfg)


if __name__ == "__main__":
    main()
