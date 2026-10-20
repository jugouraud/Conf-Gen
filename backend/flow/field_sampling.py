"""Evaluate the prompt field on a fixed bank in bounded tensor batches."""

import torch


def evaluate_probe_field(cfg, adapter, bank, embeddings):
    chunk_size = int(cfg["experiment"].get("probe_batch_size", 1))
    if chunk_size < 1:
        raise ValueError("probe_batch_size must be positive")
    outputs = []
    count = bank["trajectories"]
    for time_id, midpoint in enumerate(bank["midpoints"]):
        for offset in range(0, count, chunk_size):
            start = time_id * count + offset
            end = min(start + chunk_size, (time_id + 1) * count)
            states = bank["states"][start:end].to(adapter.device)
            effect = adapter.prompt_effect(states, float(midpoint), embeddings)
            outputs.extend(effect.cpu().unbind(0))
    return torch.stack(outputs)
