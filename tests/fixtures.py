"""Small cached field fixture for CPU distance and provenance checks."""

import json
from types import SimpleNamespace

import torch

from backend.flow.common import sha256
from backend.flow.feature_store import feature_file, save_feature
from backend.flow.model import model_signature
from backend.flow.probe_bank import bank_file, build_bank


def cached_experiment(root):
    cfg = {"paths": {"output": str(root)},
           "experiment": {"alpha": 0.25, "reference_limit": 0,
                          "validation_limit": 0, "test_limit": 0},
           "probes": {"trajectories": 1, "time_points": 1, "integration_steps": 2}}
    values = {"safe_reference": [0., 10.], "safe_validation": [1., 2., 4.],
              "safe_test": [5.], "unsafe_test": [15.]}
    groups = {}
    for split, fields in values.items():
        groups[split] = [{"id": f"{split}-{i}", "prompt": f"Prompt {split} {i}",
                          "label": "risky" if split == "unsafe_test" else "safe",
                          "source": "i2p" if split == "unsafe_test" else "diffusiondb"}
                         for i in range(len(fields))]
        path = root / "splits" / f"{split}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in groups[split]))
    metadata = {"model_id": "toy", "scheduler": {}}
    adapter = SimpleNamespace(metadata=metadata, signature=model_signature(metadata))
    provenance = [{"time_index": 0, "trajectory_id": i, "seed": i,
                   "prompt_id": groups["safe_reference"][i]["id"]} for i in range(1)]
    bank = build_bank(cfg, adapter, groups["safe_reference"],
                       [[torch.zeros(1, 1, 1)]], provenance, [0.5])
    path = bank_file(cfg)
    path.parent.mkdir(exist_ok=True)
    torch.save(bank, path)
    bank_hash = sha256(path)
    for split, rows in groups.items():
        for row, value in zip(rows, values[split]):
            save_feature(feature_file(cfg, split, row["id"]), row["id"],
                          torch.full((1, 1, 1, 1), value), bank_hash, adapter.signature)
    return cfg
