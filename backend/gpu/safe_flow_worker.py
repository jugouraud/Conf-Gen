"""Remote Colab T4 worker for the safe-flow extraction protocol."""

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import torch
from huggingface_hub import login

ARCHIVE = Path("/content/conf-gen-safe-flow-job.zip")
RESULT = Path("/content/conf-gen-safe-flow-result.zip")
WORK = Path("/content/conf-gen-safe-flow-work")


# Hierarchy 3: atomic actions.

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


# Hierarchy 1: workflow orchestration.

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Colab did not allocate a CUDA GPU")
    WORK.mkdir(parents=True, exist_ok=True)
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
    if token_path.is_file():
        token = token_path.read_text().strip()
        token_path.unlink()
        login(token=token, add_to_git_credential=False)
    root = WORK / "results"
    stage = manifest["stage"]
    # Verify all pre-existing files before calculating, including run manifests.
    for name, checksum in manifest["inputs"].items():
        target = root / name
        if not target.is_file() or sha(target) != checksum:
            raise ValueError(f"Uploaded state mismatch: {name}")
    subprocess.run([
        sys.executable, "-m", "backend.flow.workflow",
        "--config", str(config_path), "--corpus", str(corpus),
        "--output", str(root), "--stage", stage,
    ], cwd=WORK, check=True)
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
    main()
