"""Unpack independent safe-cache archives concurrently, committing verified batches."""

import hashlib
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import zipfile

ROOT = Path("/content/conf-gen-t2i-1000/results")
LEDGER = ROOT / "safe_received.json"
STATUS = ROOT / "safe_transfer_status.json"


def write_atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def unpack_verified(archive, root=None):
    root = ROOT if root is None else root
    with zipfile.ZipFile(archive) as source:
        manifest = json.loads(source.read("safe_manifest.json"))
        if set(source.namelist()) != set(manifest) | {"safe_manifest.json"}:
            raise ValueError("Unexpected safe-cache archive contents")
        for relative, digest in manifest.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
                raise ValueError("Invalid safe-cache artifact path")
            content = source.read(relative)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("Safe feature checksum differs from the original receipt")
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    return archive


def receive(root=ROOT):
    ledger, status = root / "safe_received.json", root / "safe_transfer_status.json"
    expected = json.loads(Path("/content/t2i-safe-upload-complete.json").read_text())["batches"]
    done = set(json.loads(ledger.read_text())) if ledger.exists() else set()
    archives = sorted(Path("/content").glob("t2i-safe-*.zip"))
    pending = [path for path in archives if path.name not in done]
    write_atomic(status, {"status": "receiving", "batches": len(done), "workers": 6})
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(unpack_verified, path, root) for path in pending]
            for future in as_completed(futures):
                archive = future.result()
                done.add(archive.name)
                write_atomic(ledger, sorted(done))
                archive.unlink()
                write_atomic(status, {"status": "receiving", "batches": len(done), "workers": 6})
        if len(done) != expected:
            raise ValueError(f"Only {len(done)} of {expected} safe-cache batches were received")
        write_atomic(status, {"status": "complete", "batches": len(done), "workers": 6})
    except BaseException as error:
        write_atomic(status, {"status": "failed", "batches": len(done), "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    receive(parser.parse_args().output)
