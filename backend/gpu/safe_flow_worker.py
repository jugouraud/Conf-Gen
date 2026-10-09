"""Remote Colab T4 worker for the safe-flow extraction protocol."""
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

ARCHIVE = Path("/content/conf-gen-safe-flow-job.zip")
RESULT = Path("/content/conf-gen-safe-flow-result.zip")
WORK = Path("/content/conf-gen-safe-flow-work")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import torch
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
        from huggingface_hub import login
        token = token_path.read_text().strip()
        token_path.unlink()
        login(token=token, add_to_git_credential=False)
    sys.path.insert(0, str(WORK))
    from backend.flow_experiment import (
        extract, freeze_probes, load_config, prepare, report, resolved_path, score,
    )
    cfg = load_config(config_path)
    cfg["paths"]["corpus"] = str(corpus)
    cfg["paths"]["output"] = str(WORK / "results")
    root = resolved_path(cfg, "output")
    stage = manifest["stage"]
    # Verify all pre-existing files before calculating, including run manifests.
    for name, checksum in manifest["inputs"].items():
        target = root / name
        if not target.is_file() or sha(target) != checksum:
            raise ValueError(f"Uploaded state mismatch: {name}")
    if stage in ("all", "probes"):
        prepare(cfg)
        freeze_probes(cfg)
    if stage in ("all", "reference"):
        extract(cfg, "safe_reference")
    if stage in ("all", "calibration"):
        extract(cfg, "safe_validation")
        score(cfg, "safe_validation", calibrate=True)
    if stage in ("all", "safe-test"):
        extract(cfg, "safe_test")
        score(cfg, "safe_test")
    if stage in ("all", "unsafe-test"):
        extract(cfg, "unsafe_test")
        score(cfg, "unsafe_test")
    if stage in ("all", "report"):
        report(cfg)
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
