"""Run the T2I comparison on Colab, transferring only existing safe caches and new unsafe work."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SESSION = "conf-gen-t2i-1000"
REMOTE = "/content/conf-gen-t2i-1000"
OUTPUT = ROOT / "data/flow_t2i_two_mode_1000"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def colab(*args, capture=True):
    timeout = int(args[args.index("--timeout") + 1]) + 120 if "--timeout" in args else 600
    command = [str(Path.home() / ".local/bin/colab"), "--auth", "oauth2", *args]
    result = subprocess.run(command, capture_output=capture, text=True, timeout=timeout)
    if result.returncode:
        # CLI errors can contain signed transfer URLs. Preserve a short error
        # without echoing credentials or the full subprocess command.
        raise RuntimeError(f"Colab {args[0]} failed with exit code {result.returncode}")
    return result


def launch():
    from huggingface_hub import get_token
    token = get_token()
    if not token:
        raise RuntimeError("Cached Hugging Face authentication is required")
    colab("install", "-s", SESSION, "numpy>=2", "pandas>=2", "pyarrow>=15",
          "scikit-learn>=1.4", "huggingface-hub>=0.24,<1", "requests>=2.32",
          "diffusers==0.32.0", "transformers>=4.44,<4.50", "accelerate>=0.33", "sentencepiece>=0.2")
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        job = temporary / "job.zip"
        with zipfile.ZipFile(job, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in [ROOT / "backend/__init__.py", ROOT / "backend/paths.py",
                         *sorted((ROOT / "backend/flow").glob("*.py")),
                         *sorted((ROOT / "scripts").glob("*.py"))]:
                archive.write(path, path.relative_to(ROOT).as_posix())
            archive.writestr("scripts/__init__.py", "")
            archive.write(OUTPUT / "experiment.json", "results/experiment.json")
            for space in ("trajectory", "source_slice"):
                for path in (OUTPUT / space / "splits").glob("*.jsonl"):
                    archive.write(path, "results/" + path.relative_to(OUTPUT).as_posix())
                for relative in ("probes/probe_bank.pt", "safe_distances.npz", "safe_distances.provenance.json"):
                    archive.write(OUTPUT / space / relative, f"results/{space}/{relative}")
        part_count = 0
        with job.open("rb") as reader:
            while block := reader.read(24 * 1024**2):
                part = temporary / f"job.part{part_count:04d}"
                part.write_bytes(block)
                for attempt in range(5):
                    try:
                        colab("upload", "-s", SESSION, str(part), f"/content/t2i-job.part{part_count:04d}")
                        break
                    except (RuntimeError, subprocess.TimeoutExpired):
                        if attempt == 4:
                            raise
                        time.sleep(5)
                part_count += 1
        credentials = temporary / "token"
        credentials.write_text(token)
        colab("upload", "-s", SESSION, str(credentials), "/content/.t2i-hf-token")
        worker = temporary / "launch.py"
        worker.write_text('''import os,subprocess,sys,zipfile,shutil
from pathlib import Path
import torch
root=Path('/content/conf-gen-t2i-1000');root.mkdir(exist_ok=True)
with Path('/content/t2i-job.zip').open('wb') as writer:
 for part in sorted(Path('/content').glob('t2i-job.part*')):
  with part.open('rb') as reader:shutil.copyfileobj(reader,writer)
with zipfile.ZipFile('/content/t2i-job.zip') as archive:archive.extractall(root)
if torch.cuda.get_device_properties(0).total_memory < 75*1024**3:raise RuntimeError('A100 80GB required')
print('GPU:',torch.cuda.get_device_name(0),'free disk GiB:',round(shutil.disk_usage('/content').free/1024**3),flush=True)
credentials=Path('/content/.t2i-hf-token');env=os.environ.copy();env['HF_TOKEN']=credentials.read_text().strip();credentials.unlink()
env['PYTHONPATH']=str(root);env['PYTHONUNBUFFERED']='1';env['USE_TF']='0'
with Path('/content/t2i-worker.log').open('wb') as log:
 process=subprocess.Popen([sys.executable,'-u','-m','scripts.extract_t2i_comparison','--output',str(root/'results')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
print('Unsafe extraction PID:',process.pid,flush=True)
''')
        colab("exec", "-s", SESSION, "--timeout", "86400", "-f", str(worker), capture=False)


def receive_bundle(item):
    name, metadata = item
    if Path(name).name != name or not name.startswith("part-") or not name.endswith(".zip"):
        raise ValueError("Invalid transfer bundle name")
    folder = ROOT / ".tools/t2i-transfer"
    folder.mkdir(exist_ok=True)
    archive = folder / name
    for attempt in range(5):
        try:
            colab("download", "-s", SESSION, f"{REMOTE}/results/transfer/{name}", str(archive))
            if sha(archive) != metadata["sha256"]:
                raise ValueError("Transfer checksum mismatch")
            break
        except (RuntimeError, subprocess.TimeoutExpired, ValueError):
            if attempt == 4:
                raise
            time.sleep(5)
    with zipfile.ZipFile(archive) as source:
        files = json.loads(source.read("bundle_manifest.json"))
        if set(source.namelist()) != set(files) | {"bundle_manifest.json"} or len(files) != metadata["files"]:
            raise ValueError("Invalid transfer contents")
        for relative, digest in files.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
                raise ValueError("Invalid result path")
            target = OUTPUT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha(target) != digest:
                    raise ValueError("Conflicting existing result")
                continue
            temporary = target.with_suffix(".transfer")
            with source.open(relative) as reader, temporary.open("wb") as writer:
                import shutil
                shutil.copyfileobj(reader, writer)
            if sha(temporary) != digest:
                raise ValueError("Invalid extracted result checksum")
            temporary.replace(target)
    archive.unlink()
    return name, metadata["sha256"], files


def watch():
    ledger_path = OUTPUT / "download_status.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"bundles": {}, "files": {}}
    failures = 0
    while True:
        with tempfile.TemporaryDirectory() as directory:
            status = Path(directory) / "status.json"
            try:
                colab("download", "-s", SESSION, f"{REMOTE}/results/status.json", str(status))
                state = json.loads(status.read_text())
                failures = 0
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError):
                failures += 1
                if failures >= 30:
                    raise RuntimeError("Worker unavailable; verified results remain local")
                time.sleep(20)
                continue
        pending = [(name, metadata) for name, metadata in state["bundles"].items()
                   if ledger["bundles"].get(name) != metadata["sha256"]]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for name, digest, files in pool.map(receive_bundle, pending):
                ledger["bundles"][name] = digest
                ledger["files"].update(files)
                temporary = ledger_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(ledger))
                temporary.replace(ledger_path)
        state["locally_verified_features"] = sum("/features/" in p for p in ledger["files"])
        (OUTPUT / "status.json").write_text(json.dumps(state, indent=2) + "\n")
        print(f"{state['phase']}: {state['completed_features']}/{state['total_features']} extracted, "
              f"{state['locally_verified_features']} verified locally; {state['status']}", flush=True)
        if state["status"] in ("complete", "failed"):
            colab("download", "-s", SESSION, "/content/t2i-worker.log", str(OUTPUT / "worker.log"))
            if state["status"] == "failed":
                raise RuntimeError(state["error"])
            if state["locally_verified_features"] != state["total_features"]:
                raise ValueError("Incomplete feature transfer")
            return
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "watch"))
    args = parser.parse_args()
    launch() if args.command == "launch" else watch()


if __name__ == "__main__":
    main()
