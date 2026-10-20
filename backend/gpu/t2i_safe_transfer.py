"""Transfer frozen safe signatures without recomputing them on the GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import tempfile
import time
import zipfile

from backend.gpu.t2i_comparison_client import ROOT, OUTPUT, SESSION, colab


def upload_batch(item):
    index, paths, hashes = item
    name = f"safe-{index:05d}.zip"
    folder = ROOT / ".tools/t2i-safe-transfer"
    folder.mkdir(exist_ok=True)
    archive = folder / name
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as target:
        for relative in paths:
            target.write(OUTPUT / relative, relative)
        target.writestr("safe_manifest.json", json.dumps({p: hashes[p] for p in paths}))
    if archive.stat().st_size > 32 * 1024**2:
        raise ValueError("Safe upload exceeds Colab's 32MiB limit")
    for attempt in range(5):
        try:
            colab("upload", "-s", SESSION, str(archive), f"/content/t2i-{name}")
            break
        except RuntimeError:
            if attempt == 4:
                raise
            time.sleep(5)
    archive.unlink()
    return index


def start_receiver():
    colab("upload", "-s", SESSION, str(ROOT / "backend/gpu/t2i_safe_receiver.py"),
          "/content/t2i-safe-receiver.py")
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "receiver.py"
        script.write_text("""import subprocess,sys
from pathlib import Path
with Path('/content/t2i-safe-receiver.log').open('wb') as log:
 process=subprocess.Popen([sys.executable,'/content/t2i-safe-receiver.py'],stdout=log,stderr=subprocess.STDOUT)
print('Parallel safe cache receiver PID:',process.pid)
""")
        colab("exec", "-s", SESSION, "--timeout", "60", "-f", str(script), capture=False)


def upload_safe():
    hashes, paths = {}, []
    manifest = json.loads((OUTPUT / "experiment.json").read_text())
    for split in ("safe_reference", "safe_validation", "safe_test"):
        for identifier in manifest["split_ids"][split]:
            for space in ("trajectory", "source_slice"):
                paths.append(f"{space}/features/{split}/{identifier}.pt")
    for space in ("trajectory", "source_slice"):
        receipt = json.loads((OUTPUT / space / "safe_distances.provenance.json").read_text())
        hashes.update({f"{space}/features/{p}": h for p, h in receipt["features"].items() if p.startswith("safe_")})
    batches = [(i // 4, paths[i:i+4], hashes) for i in range(0, len(paths), 4)]
    ledger_path = OUTPUT / "safe_upload_status.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"uploaded_batches": []}
    done = set(ledger["uploaded_batches"])
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(upload_batch, item) for item in batches if item[0] not in done]
        for future in as_completed(futures):
            done.add(future.result())
            ledger["uploaded_batches"] = sorted(done)
            ledger["total_batches"] = len(batches)
            ledger_path.write_text(json.dumps(ledger))
            if len(done) % 25 == 0:
                print(f"Safe cache upload: {len(done)}/{len(batches)} batches", flush=True)
    with tempfile.TemporaryDirectory() as directory:
        sentinel = Path(directory) / "complete.json"
        sentinel.write_text(json.dumps({"batches": len(batches)}))
        colab("upload", "-s", SESSION, str(sentinel), "/content/t2i-safe-upload-complete.json")
    ledger["status"] = "complete"
    ledger_path.write_text(json.dumps(ledger))
    print("All frozen safe signatures uploaded.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("upload", "receive"), default="upload", nargs="?")
    args = parser.parse_args()
    upload_safe() if args.command == "upload" else start_receiver()
