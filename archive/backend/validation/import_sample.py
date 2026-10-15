"""Score a recorded random sample on Colab and import it with toxic source IDs."""

import hashlib
import json
import math
import random
import secrets
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.gpu.client import _validate_completed_database, run_colab_embeddings
from backend.storage.fairness import Base, _create_engine, ensure_source_id_column


def main():
    directory = ROOT / "data" / "validation" / "toxic_source"
    runs = ROOT / "data" / "validation" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    manifest_path = runs / "sample_manifest.json"
    source = ROOT / "data" / "fairness" / "fairness.sqlite3"
    stage = runs / "sample_input.sqlite3"
    result = runs / "sample_scored.sqlite3"
    if not manifest_path.exists():
        images = sorted(path for path in directory.rglob("*") if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
        seed = secrets.randbits(64)
        selected = random.Random(seed).sample(images, 10)
        manifest = {"seed": seed, "population": len(images), "samples": [
            {"source_id": f"toxic_{index}", "image": str(image.relative_to(directory)),
             "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
            for index, image in enumerate(selected, 1)
        ]}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from PIL import Image
    for item in manifest["samples"]:
        path = directory / item["image"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        with Image.open(path) as image:
            image.verify()

    if not stage.exists():
        engine = _create_engine(stage)
        Base.metadata.create_all(engine)
        engine.dispose()
        with closing(sqlite3.connect(stage)) as connection, connection:
            connection.executemany(
                "INSERT INTO entries (source_id, caption, image_path) VALUES (?, ?, ?)",
                [(item["source_id"], f"Imported image: {item['image']}", str((directory / item["image"]).resolve()))
                 for item in manifest["samples"]],
            )
    print(f"Selected {len(manifest['samples'])} images from {manifest['population']}; seed={manifest['seed']}", flush=True)
    if not result.exists():
        run_colab_embeddings(stage, result, task="image-toxicity", gpu="T4", forward_hf_token=True)
    _validate_completed_database(result, task="image-toxicity", expected_count=10)
    with closing(sqlite3.connect(result)) as connection:
        rows = connection.execute("SELECT source_id, caption, image_path, image_harmfulness FROM entries ORDER BY id").fetchall()
    assert [row[0] for row in rows] == [item["source_id"] for item in manifest["samples"]]
    for row, item in zip(rows, manifest["samples"]):
        assert row[2] == str((directory / item["image"]).resolve())
        assert math.isfinite(row[3]) and 0 <= row[3] <= 1

    backup = runs / "fairness_before_toxic_import.sqlite3"
    if not backup.exists():
        with closing(sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)) as original, closing(sqlite3.connect(backup)) as saved:
            original.backup(saved)
    engine = _create_engine(source)
    ensure_source_id_column(engine)
    engine.dispose()
    with closing(sqlite3.connect(source)) as connection, connection:
        for row in rows:
            existing = connection.execute("SELECT caption, image_path, image_harmfulness FROM entries WHERE source_id = ?", (row[0],)).fetchone()
            if existing is not None:
                if existing != row[1:]:
                    raise RuntimeError(f"Source ID already exists with different data: {row[0]}")
                continue
            connection.execute("INSERT INTO entries (source_id, caption, image_path, image_harmfulness) VALUES (?, ?, ?, ?)", row)
        inserted = connection.execute(f"SELECT id, source_id, image_harmfulness FROM entries WHERE source_id IN ({','.join('?' for _ in rows)}) ORDER BY id", [row[0] for row in rows]).fetchall()
    report = {"device": "Colab T4", "model": "google/shieldgemma-2-4b-it", "records": [
        {"id": row[0], "source_id": row[1], "image_harmfulness": row[2]} for row in inserted
    ]}
    (runs / "import_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
