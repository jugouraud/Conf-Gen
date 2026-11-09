"""CPU resume uses recovered fields and exact, bounded-memory distance arithmetic."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.runtime.repair_cpu import (
    CPUProgress,
    build_weighted_cache,
    compute_field,
    recover_native_baseline_fields,
    tiled_distance_products,
)
from backend.src.common.storage import calculate_file_sha256, read_json, write_json
from backend.src.fields.resident import RebuiltSignatureStore
from backend.src.fields.signatures import save_field_signature


@pytest.mark.integration
def test_cpu_gram_tiles_match_direct_distances_for_all_three_tables():
    rng = np.random.default_rng(1729)
    refs = rng.normal(size=(7, 19)).astype(np.float32)
    query = rng.normal(size=(3, 19)).astype(np.float32)
    observed = tiled_distance_products(
        refs, query, 4, lambda **kwargs: None, tile_size=5
    )
    expected = [
        np.linalg.norm(
            query.astype(float)[:, None] - refs[:4].astype(float)[None], axis=2
        ),
        np.linalg.norm(
            query.astype(float)[:, None] - refs[4:].astype(float)[None], axis=2
        ),
        np.linalg.norm(
            refs[:4].astype(float)[:, None] - refs[4:].astype(float)[None], axis=2
        ),
    ]
    for actual, wanted in zip(observed, expected, strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=1e-12, atol=1e-12)


@pytest.mark.integration
def test_weighted_disk_cache_reuses_inputs_and_invalidates_changed_fields(tmp_path):
    bank = {"states": torch.zeros(2, 1, 1, 1), "weights": torch.full((2,), 0.5)}
    root = tmp_path / "source"
    path = root / "trajectory/features/safe_reference/one.pt"
    row = {"id": "one"}

    def save(value):
        save_field_signature(
            path, "one", torch.full((2, 1, 1, 1), value), "bank", "model"
        )
        write_json(
            path.with_suffix(".sha256.json"), {"sha256": calculate_file_sha256(path)}
        )

    store = RebuiltSignatureStore(root, bank, "bank", "model")
    cache = tmp_path / "weighted.npy"
    save(2.0)
    first = build_weighted_cache(
        cache, [("safe_reference", row)], store, bank, lambda **kwargs: None
    )
    np.testing.assert_allclose(first, [[2 * np.sqrt(0.5)] * 2])
    del first
    before = cache.stat().st_mtime_ns
    reused = build_weighted_cache(
        cache, [("safe_reference", row)], store, bank, lambda **kwargs: None
    )
    assert cache.stat().st_mtime_ns == before
    del reused
    save(4.0)
    changed = build_weighted_cache(
        cache, [("safe_reference", row)], store, bank, lambda **kwargs: None
    )
    np.testing.assert_allclose(changed, [[4 * np.sqrt(0.5)] * 2])


@pytest.mark.unit
def test_cpu_field_uses_each_frozen_probe_and_reports_partial_progress():
    calls = []
    adapter = SimpleNamespace(
        encode_prompt=lambda text: text,
        evaluate_prompt_effect=lambda state, time, embedding: state + time,
    )
    bank = {
        "states": torch.tensor([[[[1.0]]], [[[2.0]]]]),
        "times": torch.tensor([0.25, 0.75]),
    }
    result = compute_field(
        adapter,
        bank,
        {"id": "one", "prompt": "test"},
        lambda **kw: calls.append(kw),
        18,
        1504,
    )
    torch.testing.assert_close(result.flatten(), torch.tensor([1.25, 2.75]))
    assert [call["completed_probes"] for call in calls] == [1, 2]
    assert all(call["completed_fields"] == 18 for call in calls)


@pytest.mark.integration
def test_cpu_progress_preserves_explicit_failure_and_completion(tmp_path):
    progress = CPUProgress(tmp_path, 4)
    progress("rebuild_reference_fields_cpu", completed_fields=18, total_fields=1504)
    assert read_json(tmp_path / "progress.json")["device"] == "cpu"
    progress("failed", status="error", error="measured mismatch")
    state = read_json(tmp_path / "progress.json")
    assert state["status"] == "error" and state["error"] == "measured mismatch"
    progress("completed", status="completed")
    assert read_json(tmp_path / "progress.json")["status"] == "completed"


@pytest.mark.unit
def test_cpu_field_reuses_frozen_null_path_when_available():
    calls = []

    def evaluate(state, flow_time, embeddings, index):
        calls.append(index)
        return torch.full_like(state, 3.0)

    adapter = SimpleNamespace(
        encode_prompt=lambda text: text, evaluate_recovered_probe_effect=evaluate
    )
    bank = {"states": torch.zeros(2, 1, 1, 1), "times": torch.tensor([0.25, 0.75])}
    result = compute_field(
        adapter, bank, {"id": "cached", "prompt": "x"}, lambda **kw: None, 19, 1504
    )
    assert calls == [0, 1]
    assert result.flatten().tolist() == [3.0, 3.0]


@pytest.mark.integration
def test_verified_native_baseline_supplies_missing_reference_without_inference(
    tmp_path,
):
    bank = {"states": torch.zeros(2, 1, 1, 1), "adapter_signature": "model"}
    groups = {"safe_reference": [{"id": "s"}], "unsafe_reference": [{"id": "u"}]}
    records = [
        {"row": row, "fields": {"trajectory": torch.full((2, 1, 1, 1), float(index))}}
        for index, row in enumerate(
            [groups["safe_reference"][0], groups["unsafe_reference"][0]]
        )
    ]
    folder = tmp_path / "optimization"
    folder.mkdir()
    torch.save(
        {
            "bank_hashes": {"trajectory": "bank"},
            "adapter_signature": "model",
            "records": records,
        },
        folder / "baseline.pt",
    )
    write_json(
        folder / "trajectory.compatibility.json",
        {
            "passed": True,
            "bank_sha256": "bank",
            "records": [
                {"id": name, "bitwise_equal": True, "embedding_bitwise_equal": True}
                for name in ["s", "u"]
            ],
        },
    )
    assert recover_native_baseline_fields(tmp_path, bank, "bank", groups) == 2
    assert recover_native_baseline_fields(tmp_path, bank, "bank", groups) == 0
    store = RebuiltSignatureStore(tmp_path, bank, "bank", "model")
    torch.testing.assert_close(
        store.load_signature("unsafe_reference", {"id": "u"}), torch.ones(2, 1, 1, 1)
    )
