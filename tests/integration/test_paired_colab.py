"""Colab CLI integration stays on its named T4 and transfers compact artifacts."""

import ast
import subprocess
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from backend.runtime.colab import COMPACT_PATTERNS, ColabConnection
from backend.src.common.storage import read_json, write_json


def create_connection(tmp_path):
    output = tmp_path / "prepared"
    write_json(
        output / "experiment.json",
        {
            "config": {
                "colab": {
                    "command": [
                        "wsl",
                        "-d",
                        "Ubuntu",
                        "--",
                        "/home/julien/.local/bin/colab",
                    ],
                    "auth": "oauth2",
                    "session": "isolated-sexual-t4",
                    "gpu": "T4",
                    "high_mem": True,
                    "remote_root": "/content/isolated-sexual-t4",
                }
            }
        },
    )
    (tmp_path / "gpu-inputs.zip").write_bytes(b"test upload archive")
    return ColabConnection(output)


@pytest.mark.integration
def test_cli_preflight_uses_authenticated_wsl_without_allocating_gpu(
    tmp_path, monkeypatch
):
    connection = create_connection(tmp_path)
    calls = []

    def invoke(command, **kwargs):
        assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
        calls.append(command)
        return SimpleNamespace(
            returncode=0, stdout="[other-experiment] Hardware: A100", stderr=""
        )

    monkeypatch.setattr("backend.runtime.colab.subprocess.run", invoke)
    receipt = connection.preflight()
    assert receipt["authenticated"] is True
    assert receipt["session_exists"] is False
    assert receipt["gpu_started_by_preflight"] is False
    assert len(calls) == 1
    assert calls[0][:5] == [
        "wsl",
        "-d",
        "Ubuntu",
        "--",
        "/home/julien/.local/bin/colab",
    ]
    assert calls[0][-3:] == ["--auth", "oauth2", "sessions"]
    if PureWindowsPath(str(tmp_path)).drive:
        assert connection.local_argument(tmp_path).startswith("/mnt/")


@pytest.mark.integration
def test_named_non_t4_session_is_rejected_before_upload_or_launch(
    tmp_path, monkeypatch
):
    connection = create_connection(tmp_path)
    calls = []

    def invoke(*arguments, **kwargs):
        calls.append(arguments)
        return "[isolated-sexual-t4] Hardware: A100"

    monkeypatch.setattr(connection, "invoke", invoke)
    with pytest.raises(RuntimeError, match="other than T4"):
        connection.launch()
    assert calls == [("sessions",)]


@pytest.mark.integration
def test_launch_uses_cli_t4_high_ram_and_compiles_detached_worker(
    tmp_path, monkeypatch
):
    connection = create_connection(tmp_path)
    calls, sources = [], []
    exists = False

    def invoke(*arguments, **kwargs):
        nonlocal exists
        calls.append(arguments)
        if arguments[0] == "sessions":
            return "[isolated-sexual-t4] Hardware: T4" if exists else ""
        if arguments[0] == "new":
            exists = True
        return "ok"

    def execute(source):
        tree = ast.parse(source)
        compile(tree, "remote.py", "exec")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("import importlib.metadata")
            ):
                compile(node.value, "detached_worker.py", "exec")
        sources.append(source)
        return "launched"

    monkeypatch.setattr(connection, "invoke", invoke)
    monkeypatch.setattr(connection, "execute", execute)
    assert connection.launch() == "launched"
    assert ("new", "-s", connection.session, "--gpu", "T4", "--high-mem") in calls
    assert any(c[0] == "upload" for c in calls)
    assert len(sources) == 3  # Hardware check, checksummed staging, detached launch.
    assert "Expected exactly one T4" in sources[0]
    assert "Upload checksum mismatch" in sources[1]
    assert "start_new_session=True" in sources[2]
    assert "refusing a CUDA stack replacement" in sources[2]
    assert all(c[0] not in {"stop", "restart-kernel"} for c in calls)


@pytest.mark.unit
def test_compact_collection_patterns_exclude_all_large_tensor_caches():
    large = [
        "trajectory/features/safe_reference/prompt.pt",
        "trajectory/probes/probe_bank.pt",
        "optimization/embeddings/t5/prompt.pt",
        "optimization/baseline.pt",
        "optimization/trajectory.null.pt",
    ]
    assert all(
        not any(Path(name).match(pattern) for pattern in COMPACT_PATTERNS)
        for name in large
    )
    assert any(
        Path("trajectory/distances.npz").match(pattern) for pattern in COMPACT_PATTERNS
    )


@pytest.mark.integration
@pytest.mark.parametrize("include_max", [False, True])
def test_colab_worker_runs_configured_aggregations_on_remote_cuda(
    tmp_path, monkeypatch, include_max
):
    from backend.runtime import detection as worker

    output = tmp_path / "results"
    write_json(
        output / "experiment.json",
        {
            "config": {
                "experiment": {"threads": 4},
                "execution": {"probe_aggregations": ["rms", "max"]}
                if include_max
                else {},
            }
        },
    )
    calls = []
    monkeypatch.setattr(
        worker,
        "run_detection_pipeline",
        lambda path: calls.append(("rms", path)) or ["RMS results"],
    )
    monkeypatch.setattr(
        worker, "rescore_detection", lambda argv: calls.append(("max", argv))
    )
    assert worker.run_detection_worker(["--output", str(output)]) == ["RMS results"]
    assert calls[0] == ("rms", output)
    assert len(calls) == (2 if include_max else 1)
    if include_max:
        options = dict(zip(calls[1][1][::2], calls[1][1][1::2], strict=True))
        assert options["--device"] == "cuda"
        assert options["--source"] == options["--source-slice-source"] == str(output)
        assert options["--output"] == str(output / "max_deviation")
        assert read_json(output / "status.json")["phase"] == "rms_and_max_complete"


@pytest.mark.integration
def test_compact_collection_includes_nested_max_results_without_raw_caches(tmp_path):
    expected = [
        "max_deviation/comparison.json",
        "max_deviation/trajectory/distances.npz",
        "max_deviation/source_slice/distances.provenance.json",
        "max_deviation/trajectory/scores/anchor_static/report.json",
        "max_deviation/source_slice/scores/anchor_adaptive/test_decisions.csv",
    ]
    excluded = [
        "trajectory/features/safe_reference/prompt.pt",
        "optimization/embeddings/t5/prompt.pt",
        "max_deviation/trajectory/checkpoint.npz",
        "max_deviation/source_slice/scratch/fields.dat",
    ]
    for name in expected + excluded:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    collected = {
        path.relative_to(tmp_path).as_posix()
        for pattern in COMPACT_PATTERNS
        for path in tmp_path.glob(pattern)
        if path.is_file()
    }
    assert collected == set(expected)


@pytest.mark.integration
def test_five_category_pipeline_routes_status_through_colab_cli(tmp_path, monkeypatch):
    from backend.experiments.five_category_nudity_detection_20261101 import pipeline

    calls = []
    monkeypatch.setattr(
        pipeline,
        "run_colab_operation",
        lambda output, operation: calls.append((output, operation)) or "status",
    )
    assert (
        pipeline.main(["--stage", "colab-status", "--output", str(tmp_path)])
        == "status"
    )
    assert calls == [(tmp_path.resolve(), "status")]
