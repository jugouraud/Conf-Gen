"""Summarize held-out prompt-label metrics."""

import json

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import resolved_path, write_json


# Hierarchy 3: atomic actions.

def _load_threshold(cfg):
    folder = resolved_path(cfg, "output") / "scores"
    return json.loads((folder / "threshold.json").read_text(encoding="utf-8"))


def _calculate_metrics(safe, unsafe, threshold):
    merged = pd.concat((safe, unsafe), ignore_index=True)
    truth = merged["label"].eq("risky").astype(int)
    return {
        "safe_test_n": len(safe), "unsafe_test_n": len(unsafe),
        "safe_false_positive_rate": float(safe["outside_safe_history"].mean()),
        "unsafe_flagged_rate": float(unsafe["outside_safe_history"].mean()),
        "auroc_prompt_labels": float(roc_auc_score(truth, merged["distance"])),
        "auprc_prompt_labels": float(average_precision_score(truth, merged["distance"])),
        "threshold": threshold["threshold"],
        "warning": "Prompt-category detection does not establish image-safety guarantees.",
    }


# Hierarchy 2: subworkflows.

def _load_scores(cfg):
    folder = resolved_path(cfg, "output") / "scores"
    return pd.read_csv(folder / "safe_test.csv"), pd.read_csv(folder / "unsafe_test.csv")


def _save_report(cfg, metrics):
    folder = resolved_path(cfg, "output") / "scores"
    write_json(folder / "report.json", metrics)
    print(json.dumps(metrics, indent=2))


# Hierarchy 1: workflow orchestration.

def report(cfg):
    """Read held-out scores, calculate metrics, and save a report."""
    safe, unsafe = _load_scores(cfg)
    threshold = _load_threshold(cfg)
    metrics = _calculate_metrics(safe, unsafe, threshold)
    _save_report(cfg, metrics)
