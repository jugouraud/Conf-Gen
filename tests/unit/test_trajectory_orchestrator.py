from pathlib import Path

from backend.experiments.trajectory_flow import orchestrator


def test_orchestrator_composes_generation_and_training_without_browser(monkeypatch):
    calls = []

    def generate(config, output, *, workers):
        calls.append(("generate", config, output, workers))
        return {"status": "complete", "dataset_id": "dataset"}

    def train(dataset, config, output):
        calls.append(("train", dataset, config, output))
        return {"sha256": "a" * 64, "checkpoint": str(output / "weights.pt")}

    monkeypatch.setattr(orchestrator, "run_dataset_generation", generate)
    monkeypatch.setattr(orchestrator, "run_trajectory_flow_pipeline", train)

    report = orchestrator.run_generation_training_visualization(
        Path("expert.toml"), Path("flow.toml"), Path("dataset"), Path("training"),
        workers=2, launch_browser=False,
    )

    assert report["dataset"]["dataset_id"] == "dataset"
    assert report["training"]["sha256"] == "a" * 64
    assert calls == [
        ("generate", Path("expert.toml"), Path("dataset"), 2),
        ("train", Path("dataset"), Path("flow.toml"), Path("training")),
    ]
