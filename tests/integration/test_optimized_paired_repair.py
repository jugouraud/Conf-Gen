"""Unsafe-only cohort and numerically equivalent GPU acceleration boundaries."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.runtime.repair_colab import RepairColabConnection
from backend.runtime.repair_cuda import weighted_distance_tables
from backend.runtime.repair_fields import (
    GPUReferenceStore,
    GPURepairAdapter,
    null_cache_path,
    read_native_gpu_null,
)
from backend.runtime.repair_session import verify_completed_cases
from backend.src.common.storage import calculate_file_sha256, write_json
from backend.src.inference.flow import FlowModelAdapter
from backend.src.repair.prepare import select_matched_pairs, select_repair_cases
from backend.src.repair.steering import (
    RepairedVelocityField,
    build_constant_steering_schedule,
    build_temporal_steering_geometry,
    sample_repaired_trajectory,
)


@pytest.mark.unit
def test_gpu_negative_condition_rejects_cpu_origin_and_cache_tracks_encoding(tmp_path):
    path = tmp_path / "condition.pt"
    record = {
        "device": "cpu",
        "adapter_signature": "model",
        "embeddings": [torch.ones(1)],
    }
    torch.save(record, path)
    with pytest.raises(ValueError, match="CUDA-encoded"):
        read_native_gpu_null(path, "model")
    record["device"] = "cuda"
    torch.save(record, path)
    assert read_native_gpu_null(path, "model")["device"] == "cuda"
    with pytest.raises(ValueError):
        read_native_gpu_null(path, "different")
    assert null_cache_path(tmp_path, 16, "a" * 64) != null_cache_path(
        tmp_path, 16, "b" * 64
    )


@pytest.mark.integration
def test_shift_correction_preserves_stopped_attempt_and_restores_native_fields(
    tmp_path,
):
    from backend.runtime.repair_recovery import clone_and_correct_source
    from backend.src.fields.signatures import save_field_signature

    previous = tmp_path / "previous/source"
    output = tmp_path / "corrected"
    (previous / "optimization").mkdir(parents=True)
    bank_path = previous / "trajectory/probes/probe_bank.pt"
    bank_path.parent.mkdir(parents=True)
    torch.save(
        {"adapter_signature": "model", "metadata": {"guidance_scale": 7.5}}, bank_path
    )
    bank_hash = calculate_file_sha256(bank_path)
    native = previous / "optimization/native_gpu_null_embeddings.pt"
    torch.save({"adapter_signature": "model", "embeddings": [torch.zeros(1)]}, native)
    original = torch.tensor([[1.0, 2.0]])
    wrong = torch.tensor([[1.2, 2.1]])
    for name, values in (
        ("trajectory.null.pt", original),
        ("optimized-null-16.pt", wrong),
    ):
        torch.save(
            {"adapter_signature": "model", "bank_sha256": bank_hash, "values": values},
            previous / "optimization" / name,
        )
    fields = []
    for name, values in (
        ("old", torch.tensor([[3.0, 4.0]])),
        ("new", torch.tensor([[5.0, 6.0]])),
    ):
        path = previous / f"trajectory/features/safe_reference/{name}.pt"
        save_field_signature(path, name, values, bank_hash, "model")
        write_json(
            path.with_suffix(".sha256.json"), {"sha256": calculate_file_sha256(path)}
        )
        fields.append(path)
    write_json(
        previous.parent / "native_null_recovery.json",
        {
            "status": "complete",
            "comparison_to_original_frozen_null_velocity": {"bitwise_equal": True},
        },
    )
    old = fields[0]
    allowed = {
        "source/trajectory/probes/probe_bank.pt": bank_hash,
        "source/" + old.relative_to(previous).as_posix(): calculate_file_sha256(old),
    }
    write_json(
        output / "experiment.json",
        {"reuse": {"files": allowed, "bank_sha256": bank_hash}},
    )
    hashes = {p: calculate_file_sha256(p) for p in previous.rglob("*") if p.is_file()}
    result = clone_and_correct_source(previous, output)
    assert result["corrected_ids"] == ["new"]
    corrected = torch.load(
        output / "source/trajectory/features/safe_reference/new.pt", weights_only=True
    )["values"]
    assert torch.equal(corrected, torch.tensor([[5.0, 6.0]]) + 7.5 * (wrong - original))
    assert all(calculate_file_sha256(p) == digest for p, digest in hashes.items())


def groups():
    result = {"safe_test": [], "unsafe_test": []}
    for category in range(5):
        for scenario in range(8):
            for variant in range(2):
                pair = f"{category}-{scenario}-{variant}"
                for label in ("safe", "unsafe"):
                    result[f"{label}_test"].append(
                        {
                            "id": f"{pair}_{label}",
                            "pair_id": pair,
                            "scenario_id": f"{category}-{scenario}",
                            "category": str(category),
                            "variant_id": str(variant),
                            "prompt": f"{pair}-{label}",
                            "label": label,
                        }
                    )
    return result


@pytest.mark.unit
def test_unsafe_only_uses_exact_previous_unsafe_members_without_safe_variants():
    data = groups()
    options = {
        "cohort_selection": "unsafe_test_only",
        "seed": 20261102,
        "pairs_per_category": 4,
    }
    cases = select_repair_cases(data, options)
    previous = select_matched_pairs(data, 4, 20261102)
    assert cases == [row for row in previous if row["split"] == "unsafe_test"]
    assert len(cases) == 20
    assert len({row["scenario_id"] for row in cases}) == 20
    assert all(row["label"] == "unsafe" for row in cases)
    assert all(
        sum(row["category"] == str(category) for row in cases) == 4
        for category in range(5)
    )
    assert all(
        data[row["split"]][row["source_index"]]["id"] == row["id"] for row in cases
    )


@pytest.mark.unit
def test_gpu_distance_tiles_preserve_native_fp32_weighting_and_fp64_distances():
    torch.manual_seed(1729)
    refs = torch.randn(7, 4, 1, 2, 2)
    queries = torch.randn(3, 4, 1, 2, 2)
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
    actual, cross = weighted_distance_tables(
        refs, queries, weights, 4, "cpu", lambda **kw: None, tile_size=5
    )
    weighted_refs = (
        (refs * weights.sqrt()[None, :, None, None, None]).flatten(1).double()
    )
    weighted_queries = (
        (queries * weights.sqrt()[None, :, None, None, None]).flatten(1).double()
    )
    expected = torch.cdist(weighted_queries, weighted_refs).numpy()
    expected_cross = torch.cdist(weighted_refs[:4], weighted_refs[4:]).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cross, expected_cross, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_cached_geometry_matches_streamed_geometry_for_nonuniform_weights():
    torch.manual_seed(1729)
    bank = {
        "states": torch.zeros(4, 1, 2, 2),
        "weights": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "time_points": 2,
        "trajectories": 2,
        "midpoints": torch.tensor([0.25, 0.75]),
    }
    refs = torch.randn(5, 4, 1, 2, 2)
    data = {
        "safe_reference": [{"id": str(i)} for i in range(3)],
        "unsafe_reference": [{"id": str(i)} for i in range(3, 5)],
    }
    fast = GPUReferenceStore.__new__(GPUReferenceStore)
    fast.bank, fast.values, fast.device, fast.batch_size = bank, refs, "cpu", 2
    fast.lookup = {
        (split, row["id"]): int(row["id"])
        for split, rows in data.items()
        for row in rows
    }
    slow = SimpleNamespace(load_signature=lambda split, row: refs[int(row["id"])])
    q, a = torch.randn(4, 1, 2, 2), torch.randn(4, 1, 2, 2)
    experiment = SimpleNamespace(
        bank=bank,
        groups=data,
        calibration={"safe_cutoff": 3, "unsafe_cutoff": 1},
        features=slow,
    )
    expected = build_temporal_steering_geometry(q, a, experiment)
    experiment.features = fast
    actual = build_temporal_steering_geometry(q, a, experiment)
    for label in ["safe", "unsafe"]:
        for left, right in zip(
            actual.coefficients[label], expected.coefficients[label], strict=True
        ):
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)


class AnalyticTransformer:
    dtype = torch.float32

    def __init__(self):
        self.calls = 0

    def __call__(self, *, hidden_states, encoder_hidden_states, **kwargs):
        self.calls += 1
        return (hidden_states + encoder_hidden_states[:, :1, :1].reshape(-1, 1, 1, 1),)


def analytic_adapter():
    adapter = GPURepairAdapter.__new__(GPURepairAdapter)
    (
        adapter.device,
        adapter.shape,
        adapter.guidance_scale,
        adapter.num_train_timesteps,
    ) = "cpu", (1, 1, 1), 2.0, 1000
    adapter.pipe = SimpleNamespace(transformer=AnalyticTransformer())
    adapter.null_embeddings = (torch.zeros(1, 1, 1), torch.zeros(1, 1))
    adapter._diagnostic = None
    return adapter


@pytest.mark.unit
@pytest.mark.parametrize("strength", [0, 0.25, 1])
def test_packed_conditions_evaluate_same_velocity_at_same_current_state(strength):
    adapter = analytic_adapter()
    state = torch.ones(1, 1, 1, 1)
    original = (torch.full((1, 1, 1), 4.0), torch.zeros(1, 1))
    anchor = (torch.full((1, 1, 1), 1.0), torch.zeros(1, 1))
    expected_original = FlowModelAdapter.evaluate_guided_velocity(
        adapter, state, 0.4, original
    )
    expected_anchor = FlowModelAdapter.evaluate_guided_velocity(
        adapter, state, 0.4, anchor
    )
    before = adapter.pipe.transformer.calls
    actual = adapter.evaluate_repaired_guided_velocity(
        state, 0.4, original, anchor, strength
    )
    torch.testing.assert_close(
        actual, (1 - strength) * expected_original + strength * expected_anchor
    )
    assert adapter.pipe.transformer.calls == before + 1
    if strength:
        torch.testing.assert_close(
            adapter.cached_diagnostic_velocity(state, 0.4, original), expected_original
        )
        assert adapter.cached_diagnostic_velocity(state, 0.4, original) is None


@pytest.mark.unit
def test_sampling_diagnostic_reuses_actual_original_velocity_without_extra_forward():
    adapter = analytic_adapter()
    original = (torch.full((1, 1, 1), 4.0), torch.zeros(1, 1))
    anchor = (torch.full((1, 1, 1), 1.0), torch.zeros(1, 1))
    field = RepairedVelocityField(
        adapter, original, anchor, build_constant_steering_schedule(0.25)
    )
    _, stats = sample_repaired_trajectory(field, 1729, 4)
    assert adapter.pipe.transformer.calls == 4
    assert stats["model_evaluations"] == 4
    assert stats["diagnostic_model_evaluations"] == 0
    assert stats["integrated_field_perturbation"] > 0


@pytest.mark.integration
def test_high_memory_allocation_falls_back_only_to_approved_t4(tmp_path, monkeypatch):
    write_json(
        tmp_path / "experiment.json",
        {
            "config": {
                "colab": {
                    "command": ["colab"],
                    "session": "test",
                    "gpu": "A100",
                    "fallback_gpu": "T4",
                    "remote_root": "/content/test",
                }
            }
        },
    )
    conn = RepairColabConnection(tmp_path)
    calls = []

    def invoke(*args, **kwargs):
        calls.append(args)
        if args[args.index("--gpu") + 1] == "A100":
            raise RuntimeError("A100 unavailable")
        return "created"

    monkeypatch.setattr(conn, "invoke", invoke)
    conn.create()
    assert calls == [
        ("new", "-s", "test", "--gpu", "A100", "--high-mem"),
        ("new", "-s", "test", "--gpu", "T4", "--high-mem"),
    ]


@pytest.mark.integration
def test_completion_rejects_safe_variant_and_missing_method_even_with_correct_counts(
    tmp_path,
):
    options = {
        "cohort_selection": "unsafe_test_only",
        "methods": [f"R{i}" for i in range(6)],
        "modes": ["safe_attraction_unsafe_avoidance"],
        "generation_seeds": [1729],
    }
    manifest = {
        "expected_prompts": 20,
        "expected_generation_records": 120,
        "selected_cohort": [{"id": str(i)} for i in range(20)],
        "fingerprint": {"settings": {"experiment": options}},
    }
    write_json(tmp_path / "experiment.json", manifest)
    digest = calculate_file_sha256(tmp_path / "experiment.json")
    for i in range(20):
        rows = [
            {
                "id": str(i),
                "method": method,
                "mode": options["modes"][0],
                "seed": 1729,
                "split": "unsafe_test",
            }
            for method in options["methods"]
        ]
        write_json(
            tmp_path / f"pilot/records/{i}.json",
            {"experiment_sha256": digest, "rows": rows, "artifacts": {}},
        )
    write_json(
        tmp_path / "image_scoring/shieldgemma2/pilot/status.json",
        {"status": "complete"},
    )
    assert verify_completed_cases(tmp_path) == 120
    path = tmp_path / "pilot/records/0.json"
    receipt = json.loads(path.read_text())
    receipt["rows"][0]["split"] = "safe_test"
    write_json(path, receipt)
    with pytest.raises(ValueError, match="Safe test variants"):
        verify_completed_cases(tmp_path)
    receipt["rows"][0]["split"] = "unsafe_test"
    receipt["rows"][0]["method"] = "R1"
    write_json(path, receipt)
    with pytest.raises(ValueError, match="duplicate"):
        verify_completed_cases(tmp_path)


@pytest.mark.integration
def test_uploaded_cache_scripts_compile_and_reject_unsafe_paths_before_writing(
    tmp_path, monkeypatch
):
    import ast

    from backend.src.common.storage import read_json

    package = tmp_path / "input.zip"
    package.write_bytes(b"source bundle")
    part = tmp_path / "part0000"
    part.write_bytes(b"recovered cache")
    digest = calculate_file_sha256(part)
    config = {
        "command": ["colab"],
        "session": "test",
        "gpu": "A100",
        "fallback_gpu": "T4",
        "remote_root": "/content/test",
        "hf_token_path": "/content/.test-token",
    }
    write_json(
        tmp_path / "experiment.json",
        {"config": {"colab": config}, "reuse": {"archive_sha256": digest}},
    )
    write_json(
        tmp_path / "validation.json",
        {
            "package_path": str(package),
            "package_sha256": calculate_file_sha256(package),
        },
    )
    metadata = tmp_path / "metadata.json"
    write_json(
        metadata, {"sha256": digest, "parts": [{"name": part.name, "sha256": digest}]}
    )
    write_json(
        tmp_path / "gpu/reuse_upload.json",
        {"metadata": str(metadata), "parts_directory": str(tmp_path)},
    )
    conn = RepairColabConnection(tmp_path)
    sources = []

    def execute(source):
        tree = ast.parse(source)
        compile(tree, "remote.py", "exec")
        sources.append(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith(
                    "import hashlib,json,shutil,traceback,zipfile"
                )
            ):
                compile(ast.parse(node.value), "stage_cache.py", "exec")
                assert "is_relative_to" in node.value and "sha256" in node.value
        return 'CACHE_STAGE={"status":"complete","files":1}'

    monkeypatch.setattr(conn, "verify_hardware", lambda: None)
    monkeypatch.setattr(conn, "stage_model_auth", lambda: None)
    monkeypatch.setattr(conn, "invoke", lambda *args, **kwargs: "ok")
    monkeypatch.setattr(conn, "execute", execute)
    assert conn.stage()["status"] == "complete"
    assert any(
        "expected_hashes=manifest['reuse']['files']" in source for source in sources
    )
    assert read_json(tmp_path / "progress.json")["status"] == "starting"


@pytest.mark.unit
def test_native_sampling_gate_falls_back_without_changing_the_velocity(tmp_path):
    adapter = analytic_adapter()
    transformer = adapter.pipe.transformer

    class IncompatiblePackedTransformer:
        dtype = torch.float32

        def __call__(self, **kwargs):
            prediction = transformer(**kwargs)[0]
            return (prediction + 0.2 if len(prediction) == 3 else prediction,)

    adapter.pipe.transformer = IncompatiblePackedTransformer()
    original = (torch.full((1, 1, 1), 4.0), torch.zeros(1, 1))
    anchor = (torch.full((1, 1, 1), 1.0), torch.zeros(1, 1))
    adapter.encode_prompt = lambda text: original if text == "original" else anchor
    bank = {"states": torch.ones(1, 1, 1, 1), "times": torch.tensor([0.25])}
    adapter.check_packed_sampling(tmp_path, bank, ["original", "anchor"])
    assert adapter.packed_sampling_enabled is False
    state = bank["states"]
    expected = 0.75 * FlowModelAdapter.evaluate_guided_velocity(
        adapter, state, 0.25, original
    ) + 0.25 * FlowModelAdapter.evaluate_guided_velocity(adapter, state, 0.25, anchor)
    field = RepairedVelocityField(
        adapter, original, anchor, build_constant_steering_schedule(0.25)
    )
    torch.testing.assert_close(field.evaluate_guided_velocity(state, 0.25), expected)
    assert field.model_evaluations == 2


@pytest.mark.unit
def test_full_run_estimate_uses_only_unsafe_test_scope():
    from backend.src.repair.report import _estimate_full_run

    experiment = SimpleNamespace(
        settings={
            "experiment": {
                "cohort_selection": "unsafe_test_only",
                "methods": ["R0", "R1"],
                "modes": ["safe_attraction_unsafe_avoidance"],
                "generation_seeds": [1729],
                "inference_steps": 32,
            },
            "steering": {"window_widths": [0.25], "matched_budgets": []},
        },
        groups={
            "safe_test": [{}] * 200,
            "unsafe_test": [{}] * 200,
            "additional_safe_test": [],
            "safe_reference": [{}],
            "unsafe_reference": [{}],
        },
        bank={"states": torch.zeros(4, 1, 1, 1)},
    )
    estimate = _estimate_full_run(experiment, [])
    assert estimate["held_out_prompts"] == 200
    assert estimate["held_out_splits"] == ["unsafe_test"]
