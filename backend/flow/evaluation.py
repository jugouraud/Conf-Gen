"""Score a held-out safe or risky split against the frozen threshold."""

import json

from .common import resolved_path
from .metrics import reference_history, score_rows, write_scores

EVALUATION_SPLITS = ("safe_test", "unsafe_test")


# Hierarchy 3: atomic actions.

def _validate_split(split):
    if split not in EVALUATION_SPLITS:
        raise ValueError("Evaluation requires safe_test or unsafe_test")


def _load_threshold(cfg):
    path = resolved_path(cfg, "output") / "scores" / "threshold.json"
    if not path.exists():
        raise FileNotFoundError("Calibrate before evaluating")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_threshold(threshold, probe_hash, reference_ids):
    if threshold["bank_sha256"] != probe_hash or threshold["reference_ids"] != reference_ids:
        raise ValueError("Calibration was frozen with different probes/reference history")


# Hierarchy 2: subworkflows.

def _load_reference(cfg):
    return reference_history(cfg)


def _score_split(cfg, split, bank, probe_hash, reference):
    return score_rows(cfg, split, bank, probe_hash, reference)


def _save_scores(cfg, split, rows, threshold):
    write_scores(cfg, split, rows, threshold)


# Hierarchy 1: workflow orchestration.

def evaluate(cfg, split):
    """Measure and save one held-out split without changing calibration."""
    _validate_split(split)
    bank, probe_hash, reference_ids, reference = _load_reference(cfg)
    threshold = _load_threshold(cfg)
    _verify_threshold(threshold, probe_hash, reference_ids)
    rows = _score_split(cfg, split, bank, probe_hash, reference)
    _save_scores(cfg, split, rows, threshold)
