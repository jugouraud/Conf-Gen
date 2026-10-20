"""Checkpoint and verified transfer bundles for T2I feature extraction."""

import json
import time
import zipfile

from .common import sha256, write_json
from .comparison import VARIANTS


class Run:
    def __init__(self, output):
        self.output = output.resolve()
        self.status_path = self.output / "status.json"
        self.state = {"status": "running", "phase": "loading_model", "completed_features": 0,
                      "counts": {}, "bundles": {}, "started_unix": time.time()}
        manifest_path = self.output / "experiment.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.state["total_features"] = len(VARIANTS) * sum(manifest["counts"].values())
        self.pending = []
        self.hashes = {space: {} for space in VARIANTS}
        self.tables = {space: {} for space in VARIANTS}
        self.counter = 0
        self.update()

    def update(self, **values):
        self.state.update(values, elapsed_seconds=time.time() - self.state["started_unix"])
        if self.state.get("total_features"):
            self.state["feature_progress_percent"] = (
                100 * self.state["completed_features"] / self.state["total_features"])
        temporary = self.status_path.with_suffix(".tmp")
        write_json(temporary, self.state)
        temporary.replace(self.status_path)

    def bundle(self, force=False):
        if not self.pending or (not force and len(self.pending) < 16
                                and sum(p.stat().st_size for p in self.pending) < 32 * 1024**2):
            return
        folder = self.output / "transfer"
        folder.mkdir(exist_ok=True)
        name = f"part-{self.counter:05d}.zip"
        path = folder / name
        files = {p.relative_to(self.output).as_posix(): sha256(p) for p in self.pending}
        with zipfile.ZipFile(path.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for p in self.pending:
                archive.write(p, p.relative_to(self.output).as_posix())
            archive.writestr("bundle_manifest.json", json.dumps(files))
        path.with_suffix(".tmp").replace(path)
        self.state["bundles"][name] = {"sha256": sha256(path), "files": len(files), "bytes": path.stat().st_size}
        self.counter += 1
        self.pending.clear()
        self.update()

    def register(self, path, space=None, split=None, prompt_id=None):
        self.pending.append(path)
        if space is not None:
            relative = f"{split}/{prompt_id}.pt"
            self.hashes[space][relative] = sha256(path)
            self.state["completed_features"] += 1
            key = f"{space}/{split}"
            self.state["counts"][key] = self.state["counts"].get(key, 0) + 1
        self.bundle()
