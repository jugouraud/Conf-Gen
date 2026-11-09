"""Check resident evaluation against direct distances and the native paired workflow."""

import numpy as np
import pytest
import torch

from backend.src.common.storage import (
    calculate_file_sha256,
    calculate_source_fingerprints,
    read_json,
    write_json,
)
from backend.src.data.paired import read_paired_detection_prompts
from backend.src.detection.contracts import IMPLEMENTATION_FILES
from backend.src.detection.extract import run_detection_extraction
from backend.src.detection.pipeline import main as rescore_detection
from backend.src.detection.prepare import run_detection_preparation
from backend.src.detection.score import run_detection_scoring
from backend.src.fields.distances import (
    ResidentFieldDistances,
    calculate_resident_training_distances,
)
from backend.src.inference.flow import FlowModelAdapter
from tests.helpers.native_flow import TinyNativePipe
from tests.integration.test_paired_detection import (
    TinyFlowAdapter,
    create_paired_inputs,
)


@pytest.mark.integration
def test_sexual_columns_preserve_labels_pairs_and_scenario_splits(tmp_path):
    dataset, config = create_paired_inputs(tmp_path)
    dataset.write_text(
        dataset.read_text().replace(
            "legal_prompt,illegal_prompt", "control_prompt,target_prompt"
        )
    )
    config.write_text(
        config.read_text()
        + '\n[dataset.prompt_columns]\nsafe="control_prompt"\nunsafe="target_prompt"\n'
    )
    output = tmp_path / "run"
    run_detection_preparation(config, output)
    groups, audit = read_paired_detection_prompts(
        dataset, prompt_columns={"safe": "control_prompt", "unsafe": "target_prompt"}
    )
    assert audit["pairs"] == 72
    for split in ("reference", "validation", "test"):
        assert {r["pair_id"] for r in groups[f"safe_{split}"]} == {
            r["pair_id"] for r in groups[f"unsafe_{split}"]
        }
        assert all(r["prompt"].startswith("legal ") for r in groups[f"safe_{split}"])
        assert all(
            r["prompt"].startswith("illegal ") for r in groups[f"unsafe_{split}"]
        )
    dataset.write_text(
        dataset.read_text().replace("validation-0,validation-0", "validation-0,train-0")
    )
    with pytest.raises(ValueError, match="Scenario crosses"):
        read_paired_detection_prompts(
            dataset,
            prompt_columns={"safe": "control_prompt", "unsafe": "target_prompt"},
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_resident_products_match_direct_fp64_subtraction(device):
    generator = torch.Generator().manual_seed(1729)
    queries = torch.randn((7, 3, 2, 2, 5), generator=generator)
    references = torch.randn((5, 3, 2, 2, 5), generator=generator) + 0.3
    weights = torch.tensor([0.2, 0.3, 0.5])
    scale = weights.sqrt()[None, :, None, None, None]
    q, r = (
        (queries * scale).double().flatten(1),
        (references * scale).double().flatten(1),
    )
    expected = torch.linalg.vector_norm(q[:, None] - r[None, :], dim=-1)
    cache = ResidentFieldDistances(
        len(queries), queries.shape[1:], weights, device=device, query_batch_size=3
    )
    with pytest.raises(ValueError, match="incomplete"):
        cache.measure(list(references))
    for field in queries:
        cache.append(field)
    actual = torch.cat(
        [cache.measure(list(references[:2])), cache.measure(list(references[2:]))],
        dim=1,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-12, atol=1e-12)
    training = calculate_resident_training_distances(
        cache.values.flatten(1), feature_batch_size=7
    )
    expected_training = torch.linalg.vector_norm(q[:, None] - q[None, :], dim=-1)
    torch.testing.assert_close(
        training.cpu(), expected_training, rtol=1e-12, atol=1e-12
    )


class MovableTinyPipe(TinyNativePipe):
    def _shift_prompt(self, prompt):
        prefix, index = prompt.rsplit(" ", 1)
        shift = 100 if "validation" in prompt else 200 if "test" in prompt else 0
        return f"{prefix} {int(index) + shift}"

    def _get_clip_prompt_embeds(self, *, prompt, device, clip_model_index):
        return super()._get_clip_prompt_embeds(
            prompt=self._shift_prompt(prompt),
            device=device,
            clip_model_index=clip_model_index,
        )

    def _get_t5_prompt_embeds(self, *, prompt, device, max_sequence_length):
        return super()._get_t5_prompt_embeds(
            prompt=self._shift_prompt(prompt),
            device=device,
            max_sequence_length=max_sequence_length,
        )

    def to(self, device):
        self.transformer.to(device)
        return self


@pytest.mark.integration
@pytest.mark.parametrize("execution_mode", ["fused", "checkpointed"])
def test_all_five_resident_methods_resume_and_keep_test_out_of_calibration(
    tmp_path, monkeypatch, execution_mode
):
    _, config = create_paired_inputs(tmp_path)
    output = tmp_path / "run"
    run_detection_preparation(config, output)
    manifest = read_json(output / "experiment.json")
    manifest["config"]["execution"] = {
        "mode": execution_mode,
        "probe_batch_size": 4,
        "query_batch_size": 7,
        "reference_batch_size": 5,
        "feature_batch_size": 3,
        "workspace_gib": 0,
    }
    manifest["implementation"] = calculate_source_fingerprints(IMPLEMENTATION_FILES)
    write_json(output / "experiment.json", manifest)
    adapter = FlowModelAdapter.__new__(FlowModelAdapter)
    adapter.pipe = MovableTinyPipe()
    adapter.metadata, adapter.signature = (
        TinyFlowAdapter.metadata,
        TinyFlowAdapter.signature,
    )
    adapter.device, adapter.shape = "cpu", TinyFlowAdapter.shape
    adapter.guidance_scale, adapter.num_train_timesteps, adapter.max_sequence_length = (
        7.5,
        1024,
        256,
    )
    adapter.null_embeddings = torch.zeros(1, 2, 2), torch.zeros(1, 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2**40, 2**40))
    run_detection_extraction(output, mode="resident", adapter=adapter)
    summary = run_detection_scoring(output)
    assert len(summary) == 10  # Five scorers, two decision rules each.
    assert not (output / "transfer").exists()
    assert len(list((output / "trajectory/features").rglob("*.pt"))) == 144
    assert len(list((output / "source_slice/features").rglob("*.pt"))) == 144
    for space in ("trajectory", "source_slice"):
        receipt = read_json(output / space / "features.provenance.json")
        assert len(receipt["features"]) == 144
        assert all(
            calculate_file_sha256(output / space / "features" / relative) == digest
            for relative, digest in receipt["features"].items()
        )
    if execution_mode == "checkpointed":
        calls = adapter.pipe.transformer.calls
        maximum = rescore_detection(
            [
                "--source",
                str(output),
                "--source-slice-source",
                str(output),
                "--output",
                str(output / "max_deviation"),
                "--probe-aggregation",
                "max",
                "--working-directory",
                str(tmp_path / "scratch"),
                "--device",
                "cpu",
            ]
        )
        assert len(maximum) == 10
        assert adapter.pipe.transformer.calls == calls
    calibration_path = output / "scores/trajectory/euclidean/calibration.json"
    calibration = read_json(calibration_path)
    assert calibration["fit_policy"] == "validation_only"
    assert set(calibration["calibration_ids"]) == {
        "safe_validation",
        "unsafe_validation",
    }
    # Resume uses committed raw checkpoints instead of invoking the transformer.
    checkpoint = output / "trajectory/features/safe_reference/train-0_safe.pt"
    digest = calculate_file_sha256(checkpoint)
    before_calls = adapter.pipe.transformer.calls
    run_detection_extraction(output, mode="resident", adapter=adapter)
    assert (
        adapter.pipe.transformer.calls - before_calls == 4
    )  # Compatibility checks only.
    assert calculate_file_sha256(checkpoint) == digest
    assert run_detection_scoring(output) == summary
    # Change only held-out distances (with a fresh integrity receipt). Validation
    # cutoffs and train-only adaptive bandwidths must remain exactly unchanged.
    path = output / "trajectory/distances.npz"
    with np.load(path) as archive:
        tables = {key: archive[key].copy() for key in archive.files}
    for key in tables:
        if key.endswith("test"):
            tables[key] *= 2
    np.savez_compressed(path, **tables)
    receipt = read_json(path.with_suffix(".provenance.json"))
    receipt["distances_sha256"] = calculate_file_sha256(path)
    write_json(path.with_suffix(".provenance.json"), receipt)
    adaptive_path = output / "scores/trajectory/anchor_adaptive/calibration.json"
    adaptive = read_json(adaptive_path)
    run_detection_scoring(output)
    assert read_json(calibration_path) == calibration
    assert read_json(adaptive_path) == adaptive
