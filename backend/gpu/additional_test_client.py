"""Run only an additional held-out set against the completed experiment's frozen state."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import tempfile
import time
import threading
import zipfile

from backend.gpu import t2i_comparison_client as client
from backend.gpu import t2i_comparison_finish as finishing
from backend.gpu import t2i_safe_transfer as transfer

ROOT = client.ROOT
OUTPUT = ROOT / "data/flow_t2i_two_mode_1000/additional_tests/style_matched_safe_prompts_1000"
SESSION = "conf-gen-t2i-controls-1000"
REMOTE = "/content/conf-gen-t2i-controls-1000"


def configure():
    for module in (client, finishing, transfer):
        module.OUTPUT, module.SESSION = OUTPUT, SESSION
    client.REMOTE = finishing.REMOTE = REMOTE
    finishing.verify_safe_reuse = verify_frozen_state
    original_status = finishing.record_status
    finishing.record_status = lambda values: original_status(dict(values, additional_safe_test=1000))


def launch():
    from huggingface_hub import get_token
    token = get_token()
    if not token:
        raise RuntimeError("Cached Hugging Face login required")
    client.colab("install", "-s", SESSION, "numpy>=2", "pandas>=2", "pyarrow>=15",
                 "scikit-learn>=1.4", "huggingface-hub>=0.24,<1", "requests>=2.32",
                 "diffusers==0.32.0", "transformers>=4.44,<4.50", "accelerate>=0.33", "sentencepiece>=0.2")
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        archive = temporary / "job.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for path in [ROOT / "backend/__init__.py", ROOT / "backend/paths.py",
                         *sorted((ROOT / "backend/flow").glob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))]:
                z.write(path, path.relative_to(ROOT).as_posix())
            z.writestr("scripts/__init__.py", "")
            for name in ("experiment.json", "frozen_calibration.json", "parent_comparison.json"):
                z.write(OUTPUT / name, f"results/{name}")
            if (OUTPUT / "resume_features.json").exists():
                z.write(OUTPUT / "resume_features.json", "results/resume_features.json")
            for space in ("trajectory", "source_slice"):
                for path in (OUTPUT / space / "splits").glob("*.jsonl"):
                    z.write(path, "results/" + path.relative_to(OUTPUT).as_posix())
                for name in ("probes/probe_bank.pt", "reference_features.provenance.json", "frozen_geometry.npz"):
                    z.write(OUTPUT / space / name, f"results/{space}/{name}")
        with archive.open("rb") as reader:
            index = 0
            while data := reader.read(24 * 1024**2):
                path = temporary / f"job.part{index:04d}"
                path.write_bytes(data)
                for attempt in range(5):
                    try:
                        client.colab("upload", "-s", SESSION, str(path), f"/content/controls-job.part{index:04d}")
                        break
                    except RuntimeError:
                        if attempt == 4:
                            raise
                        time.sleep(5)
                index += 1
        credential = temporary / "token"
        credential.write_text(token)
        client.colab("upload", "-s", SESSION, str(credential), "/content/.controls-hf-token")
        script = temporary / "launch.py"
        script.write_text('''import os,subprocess,sys,zipfile,shutil
from pathlib import Path
import torch
root=Path('/content/conf-gen-t2i-controls-1000');root.mkdir(exist_ok=True)
with Path('/content/controls-job.zip').open('wb') as writer:
 for part in sorted(Path('/content').glob('controls-job.part*')):
  with part.open('rb') as reader:shutil.copyfileobj(reader,writer)
with zipfile.ZipFile('/content/controls-job.zip') as archive:archive.extractall(root)
if torch.cuda.get_device_properties(0).total_memory<39*1024**3:raise RuntimeError('A100 with at least 40GB required')
credentials=Path('/content/.controls-hf-token');env=os.environ.copy();env['HF_TOKEN']=credentials.read_text().strip();credentials.unlink()
env.update(PYTHONPATH=str(root),PYTHONUNBUFFERED='1',USE_TF='0')
command=[sys.executable,'-u','-m','scripts.extract_t2i_comparison','--output',str(root/'results'),'--split','additional_safe_test']
if (root/'results/resume_features.json').exists():command+=['--precomputed',str(root/'results/resume_features.json')]
with Path('/content/t2i-worker.log').open('wb') as log:
 process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
print('Additional-test extraction PID:',process.pid)
''')
        client.colab("exec", "-s", SESSION, "--timeout", "60", "-f", str(script), capture=False)


def upload_references():
    manifest = json.loads((OUTPUT / "experiment.json").read_text())
    paths, hashes = [], {}
    for space in ("trajectory", "source_slice"):
        receipt = json.loads((OUTPUT / space / "reference_features.provenance.json").read_text())
        hashes.update({f"{space}/features/{p}": h for p, h in receipt["features"].items()})
    for split in ("safe_reference", "unsafe_reference"):
        for identifier in manifest["reference_ids"][split]:
            paths.extend(f"{s}/features/{split}/{identifier}.pt" for s in ("trajectory", "source_slice"))
    resume_path = OUTPUT / "resume_features.json"
    if resume_path.exists():
        resume = json.loads(resume_path.read_text())
        if resume["experiment_sha256"] != client.sha(OUTPUT / "experiment.json"):
            raise ValueError("Resume checkpoint belongs to a different experiment")
        hashes.update(resume["features"])
        for identifier in manifest["split_ids"]["additional_safe_test"]:
            paths.extend(p for s in ("trajectory", "source_slice")
                         if (p := f"{s}/features/additional_safe_test/{identifier}.pt") in resume["features"])
    batches = [(i // 4, paths[i:i+4], hashes) for i in range(0, len(paths), 4)]
    ledger_path = OUTPUT / "safe_upload_status.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"uploaded_batches": []}
    done = set(ledger["uploaded_batches"])
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(transfer.upload_batch, item) for item in batches if item[0] not in done]
        for future in as_completed(futures):
            done.add(future.result())
            ledger.update(uploaded_batches=sorted(done), total_batches=len(batches))
            temporary = ledger_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(ledger)); temporary.replace(ledger_path)
            if len(done) % 25 == 0:
                print(f"Frozen reference transfer: {len(done)}/{len(batches)}", flush=True)
    with tempfile.TemporaryDirectory() as directory:
        marker = Path(directory) / "complete.json"
        marker.write_text(json.dumps({"batches": len(batches)}))
        client.colab("upload", "-s", SESSION, str(marker), "/content/t2i-safe-upload-complete.json")
    ledger["status"] = "complete"
    ledger_path.write_text(json.dumps(ledger))


def start_receiver():
    client.colab("upload", "-s", SESSION, str(ROOT / "backend/gpu/t2i_safe_receiver.py"), "/content/t2i-safe-receiver.py")
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "receiver.py"
        script.write_text(f'''import subprocess,sys
from pathlib import Path
with Path('/content/t2i-safe-receiver.log').open('wb') as log:
 process=subprocess.Popen([sys.executable,'/content/t2i-safe-receiver.py','--output','{REMOTE}/results'],stdout=log,stderr=subprocess.STDOUT)
print('Frozen reference receiver PID:',process.pid)
''')
        client.colab("exec", "-s", SESSION, "--timeout", "60", "-f", str(script), capture=False)


def launch_evaluation():
    finishing.launch_analysis(modules=("scripts.evaluate_additional_test",),
                              additional_prefixes=("additional_distances.",))


def verify_frozen_state():
    manifest = json.loads((OUTPUT / "experiment.json").read_text())
    parent = ROOT / "data/flow_t2i_two_mode_1000"
    if (client.sha(parent / "experiment.json") != manifest["parent_experiment_sha256"]
            or client.sha(parent / "comparison.json") != manifest["parent_comparison_sha256"]):
        raise ValueError("Parent experiment changed during additional-test evaluation")
    frozen = json.loads((OUTPUT / "frozen_calibration.json").read_text())
    report = json.loads((OUTPUT / "comparison.json").read_text())
    if len(report["methods"]) != 10:
        raise ValueError("Expected five methods in both modes")
    for row in report["methods"].values():
        name = f"{row['space']}_{row['method']}"
        original = json.loads((parent / "variants" / name / "calibration.json").read_text())
        if row["calibration"] != original or frozen[name] != original:
            raise ValueError("Additional tests altered a frozen cutoff or bandwidth")
    (OUTPUT / "frozen_state_validation.json").write_text(json.dumps({"status": "verified", "test_prompts": 1000,
                                                                   "calibration_unchanged": True}, indent=2))


def finish():
    while True:
        state = finishing.read_state(OUTPUT / "status.json")
        ledger = OUTPUT / "safe_upload_status.json"
        uploaded = finishing.read_state(ledger) if ledger.exists() else {}
        if state.get("status") == "failed":
            raise RuntimeError(state.get("error", "Extraction failed"))
        finishing.record_status({"status": "running", "phase": "extract_and_transfer",
                                "completed_features": state.get("completed_features", 0), "total_features": 2000,
                                "reference_batches_uploaded": len(uploaded.get("uploaded_batches", [])),
                                "reference_batches_total": uploaded.get("total_batches", 1000)})
        if state.get("status") == "complete" and uploaded.get("status") == "complete":
            break
        time.sleep(30)
    start_receiver()
    launch_evaluation()
    finishing.collect_analysis()


def keep_alive(done):
    while not done.wait(180):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "health.py"
            script.write_text("import torch\nprint('GPU runtime active:',torch.cuda.is_available())\n")
            client.colab("exec", "-s", SESSION, "--timeout", "30", "-f", str(script))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "launch", "watch", "references", "finish"))
    args = parser.parse_args()
    configure()
    if args.command == "all":
        done = threading.Event()
        with ThreadPoolExecutor(max_workers=4) as pool:
            heartbeat = pool.submit(keep_alive, done)
            try:
                launch()
                for future in as_completed([pool.submit(f) for f in (client.watch, upload_references, finish)]):
                    future.result()
            finally:
                done.set()
            heartbeat.result()
    else:
        {"launch": launch, "watch": client.watch, "references": upload_references, "finish": finish}[args.command]()


if __name__ == "__main__":
    main()
