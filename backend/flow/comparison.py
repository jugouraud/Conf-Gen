"""Frozen method definitions and artifact reuse for the T2I experiment."""

import copy
import json
import os
import shutil

from .common import ROOT, sha256

VARIANTS = {"trajectory": ("euclidean", "anchor_static", "anchor_adaptive"),
            "source_slice": ("anchor_static", "anchor_adaptive")}


def safe_source_path(source):
    """Resolve the retained safe cache while preserving historical config paths."""
    retained = ROOT / "data/flow_t2i_two_mode_1000/reused_safe_source"
    original = ROOT / "data/flow_method_comparison_1000"
    if not (source / "experiment.json").exists() and source.resolve() == original.resolve():
        return retained
    return source


def link_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.samefile(target):
            return
        if sha256(source) != sha256(target):
            raise ValueError(f"Different frozen artifact: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def space_config(output, space):
    manifest = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(manifest["config"])
    cfg["paths"]["output"] = str(output / space)
    cfg["scoring"] = {"method": "anchor_static"}
    if space == "source_slice":
        cfg["probes"]["time_points"] = 1
    return cfg
