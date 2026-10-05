"""Remote worker executed inside a Google Colab GPU runtime."""

import json
import shutil
import sqlite3
import sys
import zipfile
from contextlib import closing
from pathlib import Path


ARCHIVE_PATH = Path("/content/conf-gen-embedding-job.zip")
WORK_DIRECTORY = Path("/content/conf-gen-embedding-job")
OUTPUT_DATABASE_PATH = Path("/content/conf-gen-embedding-output.sqlite3")


def main() -> None:
    _require_cuda()
    _extract_job()
    manifest = json.loads((WORK_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    database_path = WORK_DIRECTORY / "database.sqlite3"
    original_image_paths = _read_image_paths(database_path)
    _set_remote_image_paths(database_path, manifest["images"])
    _fill_embeddings(database_path, manifest["model_id"])
    _write_image_paths(database_path, original_image_paths)
    shutil.copy2(database_path, OUTPUT_DATABASE_PATH)
    print(f"Embedding database is ready: {OUTPUT_DATABASE_PATH}")


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The Colab runtime does not expose CUDA. Check the requested GPU allocation.")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")


def _extract_job() -> None:
    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(f"Uploaded job archive not found: {ARCHIVE_PATH}")
    if WORK_DIRECTORY.exists():
        shutil.rmtree(WORK_DIRECTORY)
    WORK_DIRECTORY.mkdir(parents=True)
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        archive.extractall(WORK_DIRECTORY)


def _set_remote_image_paths(database_path: Path, image_paths: dict[str, str]) -> None:
    updates = [
        (str((WORK_DIRECTORY / relative_path).resolve()), int(record_id))
        for record_id, relative_path in image_paths.items()
    ]
    _write_image_paths(database_path, updates)


def _read_image_paths(database_path: Path) -> list[tuple[str, int]]:
    with closing(sqlite3.connect(database_path)) as connection:
        rows = connection.execute("SELECT image_path, id FROM entries ORDER BY id").fetchall()
    return [(str(image_path), int(record_id)) for image_path, record_id in rows]


def _write_image_paths(database_path: Path, updates: list[tuple[str, int]]) -> None:
    with closing(sqlite3.connect(database_path)) as connection:
        connection.executemany("UPDATE entries SET image_path = ? WHERE id = ?", updates)
        connection.commit()


def _fill_embeddings(database_path: Path, model_id: str) -> None:
    sys.path.insert(0, str(WORK_DIRECTORY))
    sys.path.insert(0, str(WORK_DIRECTORY / "backend"))
    from filling_database import fill_database

    record_count = fill_database(
        database_path,
        embedding_model_id=model_id,
        device="cuda",
        fill_inner_embeddings=True,
    )
    print(f"Filled CUDA embeddings for {record_count} records.")


if __name__ == "__main__":
    main()
