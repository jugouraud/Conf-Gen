"""COCO caption schema and annotation storage."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from backend.paths import COCO_ROOT

DEFAULT_DATABASE_PATH = COCO_ROOT / "coco.sqlite3"


def create_empty_coco_database(database_path: str | Path = DEFAULT_DATABASE_PATH) -> Path:
    """Create the empty COCO image/caption schema if it does not already exist."""
    database = Path(database_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                image_id INTEGER PRIMARY KEY,
                file_name TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS captions (
                caption_id INTEGER PRIMARY KEY,
                image_id INTEGER NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
                caption TEXT NOT NULL,
                text_context BLOB,
                text_context_shape TEXT,
                embedding_model TEXT
            );

            CREATE INDEX IF NOT EXISTS ix_captions_image_id ON captions(image_id);
            """
        )
    return database


def _read_annotations(annotations_path: Path) -> dict:
    try:
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"COCO captions file not found: {annotations_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"COCO captions file is not valid JSON: {annotations_path}") from error

    images = payload.get("images") if isinstance(payload, dict) else None
    captions = payload.get("annotations") if isinstance(payload, dict) else None
    if not isinstance(images, list) or not isinstance(captions, list):
        raise ValueError("COCO captions JSON must contain 'images' and 'annotations' lists.")  # noqa: TRY004
    return {"images": images, "captions": captions}


def _upsert_annotations(connection: sqlite3.Connection, annotations: dict) -> None:
    """Synchronize annotation metadata while preserving already computed vectors."""
    images = []
    for image in annotations["images"]:
        image_id, file_name = image.get("id"), image.get("file_name")
        if not isinstance(image_id, int) or isinstance(image_id, bool) or not isinstance(file_name, str) or not file_name:
            raise ValueError("Each COCO image requires an integer id and a file_name.")
        images.append((image_id, file_name))

    captions = []
    for annotation in annotations["captions"]:
        caption_id, image_id, caption = annotation.get("id"), annotation.get("image_id"), annotation.get("caption")
        if (
            not isinstance(caption_id, int)
            or isinstance(caption_id, bool)
            or not isinstance(image_id, int)
            or isinstance(image_id, bool)
            or not isinstance(caption, str)
            or not caption.strip()
        ):
            raise ValueError("Each COCO annotation requires integer id/image_id and a non-empty caption.")
        captions.append((caption_id, image_id, caption))

    connection.executemany(
        "INSERT INTO images(image_id, file_name) VALUES (?, ?) "
        "ON CONFLICT(image_id) DO UPDATE SET file_name = excluded.file_name "
        "WHERE images.file_name IS NOT excluded.file_name",
        images,
    )
    connection.executemany(
        "INSERT INTO captions(caption_id, image_id, caption) VALUES (?, ?, ?) "
        "ON CONFLICT(caption_id) DO UPDATE SET image_id = excluded.image_id, caption = excluded.caption "
        "WHERE captions.image_id IS NOT excluded.image_id OR captions.caption IS NOT excluded.caption",
        captions,
    )
    connection.commit()


