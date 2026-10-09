"""Colab CLI launcher for the fixed-probe Sana experiment.

Mirrors backend.gpu.prompt_representation_client: zip -> Colab T4 -> verified zip.
No GPU dependencies are needed on the client-side WSL Python installation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from backend.flow_experiment import load_config, resolved_path
from backend.gpu.client import (
    _colab_prefix, _find_colab_command, _run_command, _start_session, _stop_session,
)
from backend.paths import PROJECT_ROOT

REMOTE_INPUT = "/content/conf-gen-safe-flow-job.zip"
REMOTE_OUTPUT = "/content/conf-gen-safe-flow-result.zip"
STAGES = ("all", "probes", "reference", "calibration", "safe-test", "unsafe-test", "report")
DEPENDENCIES = (
    "numpy>=2.0.0", "pandas>=2.0.0", "pyarrow>=15.0.0",
    "scikit-learn>=1.4", "huggingface-hub>=0.24.0",
    "diffusers>=0.32.0", "transformers>=4.44,<5",
    "accelerate>=0.33", "sentencepiece>=0.2", "protobuf>=4.25",
)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1048576), b""):
            digest.update(data)
    return digest.hexdigest()


def snapshot_files(root: Path) -> dict[str, str]:
    """Collect only experiment artifacts, never arbitrary files or credentials."""
    allowed = {"splits", "probes", "features", "scores"}
    result = {}
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.relative_to(root).parts[0] in allowed:
                result[path.relative_to(root).as_posix()] = file_sha(path)
    return result


def _check_path(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or name.startswith("/") or "\\" in name
            or ".." in path.parts or "." in path.parts
            or not path.parts or path.parts[0] not in
            {"splits", "probes", "features", "scores"}):
        raise ValueError(f"Invalid result artifact path: {name!r}")
    return name


def create_job(archive: Path, cfg_path: Path, corpus: Path, root: Path,
               stage: str, forward_hf_token: bool) -> dict:
    state = snapshot_files(root)
    if stage in ("all", "probes") and state:
        raise FileExistsError(
            "A fresh 'all' or 'probes' run needs a new empty output directory; "
            "use staged commands to resume an existing experiment"
        )
    if stage not in ("all", "probes") and not state:
        raise FileNotFoundError("Staged Colab run requires local artifacts; run 'probes' first")
    manifest = {"stage": stage, "corpus_sha256": file_sha(corpus),
                "config_sha256": file_sha(cfg_path), "inputs": state,
                "format_version": 1}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1) as result:
        result.write(cfg_path, "config.toml")
        result.write(corpus, "corpus.parquet")
        result.writestr("manifest.json", json.dumps(manifest))
        for relative in ("backend/__init__.py", "backend/flow_experiment.py"):
            path = PROJECT_ROOT / relative
            result.write(path, relative)
        for name in state:
            result.write(root / name, f"results/{name}")
        if forward_hf_token:
            from huggingface_hub import get_token
            token = get_token()
            if not token:
                raise RuntimeError("--forward-hf-token needs HF_TOKEN or a local HF login")
            result.writestr(".hf-token", token)
    return manifest


def merge_result(archive: Path, root: Path, original: dict) -> int:
    """Verify every path and checksum before any file is added or replaced."""
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or "result_manifest.json" not in names:
            raise ValueError("Missing or duplicated Colab result manifest")
        result = json.loads(z.read("result_manifest.json"))
        for field in ("stage", "corpus_sha256", "config_sha256", "inputs", "format_version"):
            if result.get(field) != original[field]:
                raise ValueError(f"Returned Colab result mismatches submitted {field}")
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("Colab returned no experiment artifacts")
        expected = {"result_manifest.json"} | {f"results/{_check_path(key)}" for key in artifacts}
        if set(names) != expected:
            raise ValueError("Colab returned extra or missing result paths")
        # Abort if any uploaded state changed while the remote job was running.
        for name, value in original["inputs"].items():
            path = root / _check_path(name)
            if not path.is_file() or file_sha(path) != value:
                raise RuntimeError(f"Local experiment inputs changed during Colab job: {name}")
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="safe_flow_merge_", dir=root.parent) as d:
            temp = Path(d)
            for name, checksum in artifacts.items():
                target = temp / _check_path(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(f"results/{name}") as reader, target.open("wb") as writer:
                    shutil.copyfileobj(reader, writer)
                if file_sha(target) != checksum:
                    raise ValueError(f"Invalid Colab result SHA-256: {name}")
                current = root / name
                if current.exists() and file_sha(current) != checksum:
                    raise FileExistsError(f"Conflicting local artifact: {current}")
            count = 0
            for name in artifacts:
                destination = root / name
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temp / name, destination)
                    count += 1
    return count


def run_colab_flow(config: Path, *, stage: str = "all", gpu: str = "T4",
                   auth: str = "oauth2", session: str | None = None,
                   high_mem: bool = False, forward_hf_token: bool = False) -> Path:
    if stage not in STAGES:
        raise ValueError(f"Invalid stage {stage}")
    config = config.resolve()
    cfg = load_config(config)
    corpus = resolved_path(cfg, "corpus")
    root = resolved_path(cfg, "output")
    if not corpus.is_file():
        raise FileNotFoundError(
            f"Corpus absent: {corpus}; build it using scripts.build_prompt_dataset")
    colab = _find_colab_command()
    session_name = session or f"conf-gen-flow-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(prefix="conf_gen_flow_") as directory:
        temporary = Path(directory)
        input_file = temporary / "job.zip"
        output_file = temporary / "result.zip"
        submitted = create_job(input_file, config, corpus, root, stage, forward_hf_token)
        started = False
        try:
            _start_session(colab, auth, session_name, gpu, high_mem)
            started = True
            _run_command(_colab_prefix(colab, auth) +
                         ["install", "-s", session_name, *DEPENDENCIES])
            _run_command(_colab_prefix(colab, auth) +
                         ["upload", "-s", session_name, str(input_file), REMOTE_INPUT])
            _run_command(_colab_prefix(colab, auth) +
                         ["exec", "-s", session_name, "--timeout", "86400", "-f",
                          str(PROJECT_ROOT / "backend" / "gpu" / "safe_flow_worker.py")])
            _run_command(_colab_prefix(colab, auth) +
                         ["download", "-s", session_name, REMOTE_OUTPUT, str(output_file)])
        finally:
            if started:
                _stop_session(colab, auth, session_name)
        count = merge_result(output_file, root, submitted)
        print(f"Colab {stage}: verified and merged {count} new experiment files into {root}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "safe_flow.toml")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--gpu", choices=("T4", "L4", "G4", "H100", "A100"), default="T4")
    parser.add_argument("--auth", choices=("oauth2", "adc"), default="oauth2")
    parser.add_argument("--session")
    parser.add_argument("--high-mem", action="store_true")
    parser.add_argument("--forward-hf-token", action="store_true",
                        help="Explicitly upload your cached Hugging Face token (gated Gemma access)")
    args = parser.parse_args()
    run_colab_flow(args.config, stage=args.stage, gpu=args.gpu, auth=args.auth,
                   session=args.session, high_mem=args.high_mem,
                   forward_hf_token=args.forward_hf_token)


if __name__ == "__main__":
    main()
