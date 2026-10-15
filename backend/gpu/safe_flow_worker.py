"""Remote Colab T4 worker for the safe-flow extraction protocol."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

ARCHIVE = Path("/content/conf-gen-safe-flow-job.zip")
PARTS_MANIFEST = Path(str(ARCHIVE) + ".parts.json")
RESULT = Path("/content/conf-gen-safe-flow-result.zip")
WORK = Path("/content/conf-gen-safe-flow-work")


# Hierarchy 3: atomic actions.

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def assemble_job_archive():
    """Rebuild a chunked upload before reading or extracting the job."""
    if not PARTS_MANIFEST.exists():
        if not ARCHIVE.is_file():
            raise FileNotFoundError(f"Missing Colab job archive: {ARCHIVE}")
        return
    details = json.loads(PARTS_MANIFEST.read_text(encoding="utf-8"))
    count = details["part_count"]
    if not isinstance(count, int) or not 1 <= count <= 10000:
        raise ValueError("Invalid upload part count")
    temporary = Path(str(ARCHIVE) + ".assembling")
    parts = [Path(f"{ARCHIVE}.part{index:04d}") for index in range(count)]
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output)
    if (temporary.stat().st_size != details["archive_size"]
            or sha(temporary) != details["archive_sha256"]):
        temporary.unlink()
        raise ValueError("Uploaded job parts failed integrity check")
    os.replace(temporary, ARCHIVE)
    for part in parts:
        part.unlink()
    PARTS_MANIFEST.unlink()


def run_workflow(config_path, corpus, root, stage, token=None):
    command = [
        sys.executable, "-m", "backend.flow.workflow",
        "--config", str(config_path), "--corpus", str(corpus),
        "--output", str(root), "--stage", stage,
    ]
    with (WORK / "workflow.log").open("w+b") as log:
        environment = os.environ.copy()
        if token:
            environment["HF_TOKEN"] = token
        completed = subprocess.run(command, cwd=WORK, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            log.seek(max(0, log.tell() - 12000))
            detail = log.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Flow workflow exited with code {completed.returncode}. "
                f"Last workflow output:\n{detail}"
            )


# Hierarchy 1: workflow orchestration.

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Colab did not allocate a CUDA GPU")
    WORK.mkdir(parents=True, exist_ok=True)
    assemble_job_archive()
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in z.namelist():
            p = Path(name)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"Unsafe archive path: {name}")
        z.extractall(WORK)
    manifest = json.loads((WORK / "manifest.json").read_text())
    corpus = WORK / "corpus.parquet"
    config_path = WORK / "config.toml"
    if sha(corpus) != manifest["corpus_sha256"] or sha(config_path) != manifest["config_sha256"]:
        raise ValueError("Input archive integrity mismatch")
    token_path = WORK / ".hf-token"
    token = None
    if token_path.is_file():
        token = token_path.read_text().strip()
        token_path.unlink()
    root = WORK / "results"
    stage = manifest["stage"]
    # Verify all pre-existing files before calculating, including run manifests.
    for name, checksum in manifest["inputs"].items():
        target = root / name
        if not target.is_file() or sha(target) != checksum:
            raise ValueError(f"Uploaded state mismatch: {name}")
    run_workflow(config_path, corpus, root, stage, token)
    artifacts = {}
    with zipfile.ZipFile(RESULT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for file in sorted(root.rglob("*")):
            if not file.is_file():
                continue
            relative = file.relative_to(root).as_posix()
            if relative.split("/")[0] not in ("splits", "probes", "features", "scores"):
                raise ValueError(f"Unexpected artifact: {relative}")
            artifacts[relative] = sha(file)
            z.write(file, "results/" + relative)
        result = dict(manifest)
        result["artifacts"] = artifacts
        z.writestr("result_manifest.json", json.dumps(result))
    print(f"Completed {stage}: {len(artifacts)} artifacts.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        with zipfile.ZipFile(RESULT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("error.json", json.dumps({"error": str(error)}))
        raise
