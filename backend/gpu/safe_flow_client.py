"""Colab CLI launcher for the fixed-probe flow experiment.

Packages the job, runs it on Colab, and verifies the returned archive.
No GPU dependencies are needed on the client-side WSL Python installation.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from huggingface_hub import get_token

from backend.flow.common import load_config, resolved_path
from backend.gpu.colab import (
    _colab_prefix, _find_colab_command, _run_command, _start_session, _stop_session,
)
from backend.gpu.safe_flow_result_transfer import assemble_result
from backend.paths import PROJECT_ROOT

REMOTE_INPUT = "/content/conf-gen-safe-flow-job.zip"
REMOTE_OUTPUT = "/content/conf-gen-safe-flow-result.zip"
UPLOAD_PART_BYTES = 32 * 1024 * 1024
STAGES = ("all", "probes", "reference", "calibration", "safe-test", "unsafe-test", "report")
DEPENDENCIES = (
    "numpy>=2.0.0", "pandas>=2.0.0", "pyarrow>=15.0.0",
    "scikit-learn>=1.4", "huggingface-hub>=0.24,<1",
    "requests>=2.32",
    "diffusers==0.32.0", "transformers>=4.44,<4.50",
    "accelerate>=0.33", "sentencepiece>=0.2",
)


# Hierarchy 3: atomic actions.

def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1048576), b""):
            digest.update(data)
    return digest.hexdigest()


def split_job_archive(archive: Path, folder: Path,
                      part_bytes: int = UPLOAD_PART_BYTES) -> tuple[list[Path], Path]:
    """Split a large job into bounded uploads with a whole-archive checksum."""
    if part_bytes < 1:
        raise ValueError("Upload part size must be positive")
    parts = []
    with archive.open("rb") as source:
        while data := source.read(part_bytes):
            part = folder / f"{Path(REMOTE_INPUT).name}.part{len(parts):04d}"
            part.write_bytes(data)
            parts.append(part)
    if not parts:
        raise ValueError("Empty job archive")
    manifest = folder / "input_parts.json"
    manifest.write_text(json.dumps({
        "part_count": len(parts), "archive_size": archive.stat().st_size,
        "archive_sha256": file_sha(archive),
    }), encoding="utf-8")
    return parts, manifest


def _check_path(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or name.startswith("/") or "\\" in name
            or any(part in (".", "..") for part in name.split("/"))
            or not path.parts or path.parts[0] not in
            {"splits", "probes", "features", "scores"}):
        raise ValueError(f"Invalid result artifact path: {name!r}")
    return name


# Hierarchy 2: subworkflows.

def _needed_for_stage(stage: str, name: str) -> bool:
    """Send only the frozen state read by a held-out scoring stage."""
    if stage in ("safe-test", "unsafe-test"):
        split = "safe_test" if stage == "safe-test" else "unsafe_test"
        return (name.startswith(("splits/", "probes/", "features/safe_reference/",
                                 f"features/{split}/"))
                or name == "scores/threshold.json")
    if stage == "report":
        return name in {"scores/threshold.json", "scores/safe_test.csv",
                        "scores/unsafe_test.csv"}
    return True


def snapshot_files(root: Path, stage: str = "all") -> dict[str, str]:
    """Collect stage inputs only, never arbitrary files or credentials."""
    allowed = {"splits", "probes", "features", "scores"}
    result = {}
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.relative_to(root).parts[0] in allowed:
                name = path.relative_to(root).as_posix()
                if _needed_for_stage(stage, name):
                    result[name] = file_sha(path)
    return result


def create_job(archive: Path, cfg_path: Path, corpus: Path, root: Path,
               stage: str, forward_hf_token: bool) -> dict:
    state = snapshot_files(root, stage)
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
        source_files = [
            PROJECT_ROOT / "backend" / "__init__.py",
            PROJECT_ROOT / "backend" / "paths.py",
            PROJECT_ROOT / "backend" / "flow_experiment.py",
            PROJECT_ROOT / "scripts" / "build_prompt_dataset.py",
            *(PROJECT_ROOT / "backend" / "flow").glob("*.py"),
        ]
        for path in sorted(source_files):
            result.write(path, path.relative_to(PROJECT_ROOT).as_posix())
        for name in state:
            result.write(root / name, f"results/{name}")
        if forward_hf_token:
            token = get_token()
            if not token:
                raise RuntimeError("--forward-hf-token needs HF_TOKEN or a local HF login")
            result.writestr(".hf-token", token)
    return manifest


def merge_result(archive: Path, root: Path, original: dict) -> int:
    """Verify every path and checksum before any file is added or replaced."""
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if names == ["error.json"]:
            error = json.loads(z.read("error.json"))["error"]
            raise RuntimeError(f"Colab flow stage failed:\n{error}")
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


def upload_job(colab: str, auth: str, session: str, archive: Path,
               temporary: Path) -> None:
    """Use small uploads when an archive exceeds the Colab contents limit."""
    prefix = _colab_prefix(colab, auth) + ["upload", "-s", session]
    if archive.stat().st_size <= UPLOAD_PART_BYTES:
        _run_command(prefix + [str(archive), REMOTE_INPUT])
        return
    parts, manifest = split_job_archive(archive, temporary)
    for index, part in enumerate(parts):
        _run_command(prefix + [str(part), f"{REMOTE_INPUT}.part{index:04d}"])
        part.unlink()
    _run_command(prefix + [str(manifest), f"{REMOTE_INPUT}.parts.json"])


def download_result(colab: str, auth: str, session: str, archive: Path,
                    *, split_remote: bool = True) -> None:
    """Download verified, bounded parts instead of one large API response."""
    prefix = _colab_prefix(colab, auth)
    if split_remote:
        _run_command(prefix + ["exec", "-s", session, "--timeout", "3600", "-f",
                                str(PROJECT_ROOT / "backend" / "gpu" /
                                    "safe_flow_result_transfer.py")])
    manifest = Path(f"{archive}.parts.json")
    _run_command(prefix + ["download", "-s", session,
                            f"{REMOTE_OUTPUT}.parts.json", str(manifest)])
    details = json.loads(manifest.read_text(encoding="utf-8"))
    parts = details["parts"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 10000:
        raise ValueError("Invalid result part count")
    for index, expected in enumerate(parts):
        local_part = Path(f"{archive}.part{index:04d}")
        if (local_part.is_file() and local_part.stat().st_size == expected["size"]
                and file_sha(local_part) == expected["sha256"]):
            continue
        _run_command(prefix + ["download", "-s", session,
                                f"{REMOTE_OUTPUT}.part{index:04d}",
                                str(local_part)])
    assemble_result(manifest, archive)


# Hierarchy 1: workflow orchestration.

def run_colab_flow(config: Path, *, stage: str = "all", gpu: str = "T4",
                   auth: str = "oauth2", session: str | None = None,
                   high_mem: bool = False, forward_hf_token: bool = False,
                   reuse_session: bool = False, keep_session: bool = False) -> Path:
    if stage not in STAGES:
        raise ValueError(f"Invalid stage {stage}")
    if reuse_session and keep_session:
        raise ValueError("Choose either --reuse-session or --keep-session")
    if (reuse_session or keep_session) and not session:
        raise ValueError("Persistent session mode requires --session NAME")
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
            if not reuse_session:
                _start_session(colab, auth, session_name, gpu, high_mem)
                started = True
            _run_command(_colab_prefix(colab, auth) +
                         ["install", "-s", session_name, *DEPENDENCIES])
            upload_job(colab, auth, session_name, input_file, temporary)
            _run_command(_colab_prefix(colab, auth) +
                         ["exec", "-s", session_name, "--timeout", "86400", "-f",
                          str(PROJECT_ROOT / "backend" / "gpu" / "safe_flow_worker.py")])
            download_result(colab, auth, session_name, output_file)
        finally:
            if started and not keep_session:
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
    parser.add_argument("--reuse-session", action="store_true",
                        help="Use an existing named session and leave it running")
    parser.add_argument("--keep-session", action="store_true",
                        help="Create a named session and leave it running")
    parser.add_argument("--high-mem", action="store_true")
    parser.add_argument("--forward-hf-token", action="store_true",
                        help="Explicitly upload your cached Hugging Face token (gated model access)")
    args = parser.parse_args()
    run_colab_flow(args.config, stage=args.stage, gpu=args.gpu, auth=args.auth,
                   session=args.session, high_mem=args.high_mem,
                   forward_hf_token=args.forward_hf_token,
                   reuse_session=args.reuse_session, keep_session=args.keep_session)


if __name__ == "__main__":
    main()
