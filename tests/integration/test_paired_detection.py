"""Exercise fresh paired detection artifacts with deterministic tiny flow inference."""

import csv
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.src.common.storage import (
    calculate_file_sha256,
    link_or_copy_frozen_file,
    read_json,
    write_json,
)
from backend.src.data.paired import read_paired_detection_prompts
from backend.src.detection.calibration import calibrate_paired_detection_thresholds
from backend.src.detection.extract import run_detection_extraction
from backend.src.detection.measure import run_detection_measurement
from backend.src.detection.prepare import run_detection_preparation
from backend.src.detection.score import run_detection_scoring
from backend.src.fields.checkpoints import reuse_paired_detection_checkpoints
from backend.src.fields.signatures import (
    calculate_model_signature,
    load_signature_space_config,
    read_probe_bank,
)
from backend.src.inference.flow import FlowModelAdapter


class TinyFlowAdapter:
    device = "cpu"
    shape = (1, 1, 1)
    metadata = {"scheduler": {}, "model": "tiny deterministic integration fixture"}
    signature = calculate_model_signature(metadata)

    def encode_prompt(self, prompt):
        return (0.0 if prompt.startswith("legal") else 10.0) + int(
            prompt.rsplit(" ", 1)[1]
        ) / 100

    def evaluate_guided_velocity(self, states, time, embeddings):
        return torch.zeros_like(states) + embeddings

    def evaluate_prompt_effect(self, states, time, embeddings):
        return states * 0.01 + embeddings


@pytest.mark.unit
def test_direct_prompt_encoding_releases_previous_offloaded_components():
    calls = []

    def release_components():
        calls.append("release")

    def encode_prompt(**kwargs):
        assert calls == ["release"]
        calls.append(kwargs["prompt"])
        return "tokens", None, "pooled", None

    adapter = FlowModelAdapter.__new__(FlowModelAdapter)
    adapter.device = "cpu"
    adapter.max_sequence_length = 256
    adapter.pipe = SimpleNamespace(
        maybe_free_model_hooks=release_components, encode_prompt=encode_prompt
    )
    assert adapter.encode_prompt("next training prompt") == ("tokens", "pooled")
    assert calls == ["release", "next training prompt"]


def create_paired_inputs(tmp_path):
    dataset = tmp_path / "pairs.csv"
    with dataset.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "pair_id",
                "scenario_id",
                "split",
                "category",
                "variant_id",
                "legal_prompt",
                "illegal_prompt",
            ],
        )
        writer.writeheader()
        for split in ("train", "validation", "test"):
            for i in range(24):
                writer.writerow(
                    dict(
                        pair_id=f"{split}-{i}",
                        scenario_id=f"{split}-{i}",
                        split=split,
                        category="synthetic",
                        variant_id=1,
                        legal_prompt=f"legal {split} {i}",
                        illegal_prompt=f"illegal {split} {i}",
                    )
                )
    config = tmp_path / "config.toml"
    config.write_text(
        f'[paths]\ndataset = "{dataset.as_posix()}"\n'
        "[experiment]\nseed = 17\nguidance_scale = 7.5\nalpha = 0.05\nunsafe_alpha = 0.05\ncap_alpha = 0.05\n"
        '[model]\nid = "tiny-injected-test-model"\n'
        "[probes]\ntime_points = 2\ntrajectories = 2\nintegration_steps = 4\n",
        encoding="utf-8",
    )
    return dataset, config


@pytest.mark.integration
def test_fresh_paired_detection_only_workflow(tmp_path):
    dataset, config = create_paired_inputs(tmp_path)
    output = tmp_path / "run"
    run_detection_preparation(config, output)
    manifest_bytes = (output / "experiment.json").read_bytes()
    run_detection_extraction(output, mode="native", adapter=TinyFlowAdapter())
    run_detection_measurement(output, device="cpu", mode="paired")
    reports = run_detection_scoring(output, mode="paired")
    assert reports["safe_attraction"]["safe_false_positive_rate"] == 0
    assert reports["safe_attraction"]["unsafe_flagged_rate"] == 1
    assert read_json(output / "calibration.json")["fit_policy"] == "validation_only"
    assert read_json(output / "status.json")["completed_features"] == 144
    assert (output / "experiment.json").read_bytes() == manifest_bytes
    assert not (output / "source_slice").exists()
    assert not list(output.rglob("*.png"))
    assert len(list((output / "trajectory/features").rglob("*.pt"))) == 144
    bank = torch.load(output / "trajectory/probes/probe_bank.pt", weights_only=True)
    assert all(item["prompt_id"].startswith("train-") for item in bank["provenance"])
    with np.load(output / "trajectory/distances.npz") as archive:
        distances = {key: archive[key] for key in archive.files}
    changed = deepcopy(distances)
    for key in changed:
        if key.endswith("test"):
            changed[key] = changed[key] * 1000
    settings = read_json(output / "experiment.json")["config"]["experiment"]
    assert calibrate_paired_detection_thresholds(
        changed, settings
    ) == calibrate_paired_detection_thresholds(distances, settings)
    with pytest.raises(ValueError, match="new output"):
        run_detection_preparation(config, output)
    dataset.write_text(dataset.read_text().replace("test-0,test-0", "test-0,train-0"))
    with pytest.raises(ValueError, match="Scenario crosses"):
        read_paired_detection_prompts(dataset)


@pytest.mark.integration
def test_paired_score_rejects_modified_split(tmp_path):
    _, config = create_paired_inputs(tmp_path)
    output = tmp_path / "run"
    run_detection_preparation(config, output)
    run_detection_extraction(output, mode="native", adapter=TinyFlowAdapter())
    run_detection_measurement(output, device="cpu", mode="paired")
    path = output / "trajectory/splits/safe_test.jsonl"
    path.write_text(path.read_text().replace("legal test", "changed test"))
    with pytest.raises(ValueError, match="Split bytes"):
        run_detection_scoring(output, mode="paired")


@pytest.mark.integration
def test_expansion_reuses_identical_bank_and_prompt_signatures(tmp_path):
    dataset, config = create_paired_inputs(tmp_path)
    source, output = tmp_path / "original", tmp_path / "expanded"
    run_detection_preparation(config, source)
    run_detection_extraction(source, mode="native", adapter=TinyFlowAdapter())
    with dataset.open("a") as stream:
        stream.write("extra,extra,train,synthetic,1,legal extra 99,illegal extra 99\n")
    config.write_text(
        config.read_text().replace("[probes]", '[probes]\nfit_split = "safe_probe_fit"')
    )
    run_detection_preparation(config, output)
    fit = output / "trajectory/splits/safe_probe_fit.jsonl"
    link_or_copy_frozen_file(source / "trajectory/splits/safe_reference.jsonl", fit)
    manifest = read_json(output / "experiment.json")
    manifest["probe_fit_sha256"] = calculate_file_sha256(fit)
    manifest["continuation"] = {
        "source_experiment_sha256": calculate_file_sha256(source / "experiment.json")
    }
    write_json(output / "experiment.json", manifest)
    receipt = reuse_paired_detection_checkpoints(source, output)
    assert receipt["reused_features"] == 144
    assert calculate_file_sha256(
        source / "trajectory/probes/probe_bank.pt"
    ) == calculate_file_sha256(output / "trajectory/probes/probe_bank.pt")

    class CountingAdapter(TinyFlowAdapter):
        def __init__(self):
            self.prompts = []

        def encode_prompt(self, prompt):
            self.prompts.append(prompt)
            return super().encode_prompt(prompt)

    adapter = CountingAdapter()
    run_detection_extraction(output, mode="native", adapter=adapter)
    assert set(adapter.prompts) == {"legal extra 99", "illegal extra 99"}
    run_detection_measurement(output, device="cpu", mode="paired")
    run_detection_scoring(output, mode="paired")
    assert read_json(output / "status.json")["completed_features"] == 146
    # A bank-fitting prompt may never become a held-out or modified record.
    training = output / "trajectory/splits/safe_reference.jsonl"
    training.write_text(training.read_text().replace("legal train", "modified train"))
    with pytest.raises(ValueError, match="unchanged legal training"):
        read_probe_bank(load_signature_space_config(output, "trajectory"))
