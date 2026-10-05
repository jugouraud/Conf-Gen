"""Populate COCO caption embeddings in a SQLite database."""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
import zipfile
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tqdm import tqdm

from backend.embeddings.clip import DEFAULT_MODEL_ID, extract_inner_embeddings
from backend.paths import COCO_ROOT, PROJECT_ROOT
from backend.storage.coco import (
    _read_annotations,
    _upsert_annotations,
    create_empty_coco_database,
)

DEFAULT_ANNOTATIONS_PATH = COCO_ROOT / "annotations" / "captions_val2017.json"
DEFAULT_IMAGES_DIRECTORY = COCO_ROOT / "images" / "val2017"
DEFAULT_DATABASE_PATH = COCO_ROOT / "coco.sqlite3"
REMOTE_ARCHIVE_PATH = "/content/conf-gen-coco-job.zip"
REMOTE_DATABASE_PATH = "/content/conf-gen-coco-output.sqlite3"
COCO_EXEC_TIMEOUT_SECONDS = 600
COCO_PROCESS_TIMEOUT_SECONDS = 660


def fill_coco_database(
    *,
    annotations_path: str | Path = DEFAULT_ANNOTATIONS_PATH,
    images_directory: str | Path = DEFAULT_IMAGES_DIRECTORY,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
) -> int:
    """Insert COCO captions and fill missing 77x768 text-context embeddings.

    Each annotation is stored once, under its COCO caption ID, and linked to
    the ``images`` row whose primary key is the COCO image ID. Re-running this
    function resumes from captions whose embedding is still NULL.
    """
    annotations = _read_annotations(Path(annotations_path))
    image_directory = Path(images_directory)
    database = create_empty_coco_database(database_path)

    with closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _upsert_annotations(connection, annotations)
        pending = connection.execute(
            """
            SELECT captions.caption_id, captions.caption, images.file_name
            FROM captions JOIN images USING (image_id)
            WHERE captions.text_context IS NULL
            ORDER BY captions.caption_id
            """
        ).fetchall()

        with TemporaryDirectory(prefix="coco_caption_embeddings_") as temporary_directory:
            for caption_id, caption, file_name in tqdm(pending, desc="Embedding COCO captions", unit="caption"):
                image_path = image_directory / file_name
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"COCO image for caption {caption_id} not found: {image_path}. "
                        "Pass --images-dir pointing to the val2017 image directory."
                    )
                embedding = _extract_text_context(
                    caption, image_path, Path(temporary_directory) / str(caption_id), model_id, device
                )
                connection.execute(
                    """
                    UPDATE captions
                    SET text_context = ?, text_context_shape = ?, embedding_model = ?
                    WHERE caption_id = ?
                    """,
                    (sqlite3.Binary(embedding.tobytes()), json.dumps(list(embedding.shape)), model_id, caption_id),
                )
                connection.commit()
    return len(pending)


def fill_coco_database_on_gpu(
    *,
    annotations_path: str | Path = DEFAULT_ANNOTATIONS_PATH,
    images_directory: str | Path = DEFAULT_IMAGES_DIRECTORY,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    model_id: str = DEFAULT_MODEL_ID,
    limit: int | None = None,
    batch_size: int = 250,
) -> int:
    """Fill missing COCO text contexts in batches in one Colab T4 session."""
    if os.name == "nt":
        raise RuntimeError(
            "Google Colab CLI 0.7.4 requires POSIX termios and cannot run natively on Windows. "
            "Run .\\scripts\\coco.ps1 --gpu from PowerShell after setting up WSL."
        )
    sys.path.insert(0, str(PROJECT_ROOT))
    from backend.gpu.client import (
        _colab_prefix,
        _find_colab_command,
        _install_remote_dependencies,
        _run_command,
        _start_session,
        _stop_session,
    )

    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive.")
    if batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    annotation_file = Path(annotations_path).resolve()
    image_directory = Path(images_directory).resolve()
    database = create_empty_coco_database(database_path).resolve()
    annotations = _read_annotations(annotation_file)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _upsert_annotations(connection, annotations)
        annotation_ids = {caption["id"] for caption in annotations["captions"]}
        pending = [
            caption_id for (caption_id,) in connection.execute(
                "SELECT caption_id FROM captions WHERE text_context IS NULL ORDER BY caption_id"
            ) if caption_id in annotation_ids
        ]
        if limit is not None:
            pending = pending[:limit]
    if not pending:
        return 0

    selected_annotations = _annotations_for_pending(annotations, set(pending))
    image_names = sorted({image["file_name"] for image in selected_annotations["images"]})
    missing_image = next((image_directory / name for name in image_names if not (image_directory / name).is_file()), None)
    if missing_image is not None:
        raise FileNotFoundError(
            f"COCO image not found: {missing_image}. Pass --images-dir pointing to the val2017 image directory."
        )

    batches = _coco_caption_batches(annotations, pending, batch_size)
    colab_command = _find_colab_command()
    session_name: str | None = None
    session_started = False
    completed = 0
    with TemporaryDirectory(prefix="conf_gen_coco_colab_") as temporary_directory:
        temporary = Path(temporary_directory)
        archive = temporary / "coco-job.zip"
        returned_database = temporary / "coco.sqlite3"
        try:
            for batch_number, batch_ids in enumerate(batches, start=1):
                batch_annotations = _annotations_for_pending(annotations, set(batch_ids))
                batch_image_names = sorted({image["file_name"] for image in batch_annotations["images"]})
                _build_coco_job_archive(batch_annotations, image_directory, batch_image_names, archive, model_id)
                print(f"COCO GPU batch {batch_number}/{len(batches)}: {len(batch_ids)} captions", flush=True)
                for attempt in range(1, 4):
                    if session_name is None:
                        session_name = f"conf-gen-coco-{uuid.uuid4().hex[:8]}"
                    try:
                        if not session_started:
                            _start_session(colab_command, "oauth2", session_name, "T4", False)
                            session_started = True
                            _install_remote_dependencies(colab_command, "oauth2", session_name)
                        returned_database.unlink(missing_ok=True)
                        _run_command(_colab_prefix(colab_command, "oauth2") + [
                            "upload", "-s", session_name, str(archive), REMOTE_ARCHIVE_PATH,
                        ])
                        _run_command(
                            _colab_prefix(colab_command, "oauth2")
                            + ["exec", "-s", session_name, "--timeout", str(COCO_EXEC_TIMEOUT_SECONDS),
                               "-f", str(PROJECT_ROOT / "backend" / "gpu" / "coco_worker.py")],
                            timeout=COCO_PROCESS_TIMEOUT_SECONDS,
                        )
                        _run_command(_colab_prefix(colab_command, "oauth2") + [
                            "download", "-s", session_name, REMOTE_DATABASE_PATH, str(returned_database),
                        ])
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                        _stop_session(colab_command, "oauth2", session_name)
                        session_name = None
                        session_started = False
                        if attempt == 3:
                            raise RuntimeError(
                                f"COCO GPU batch {batch_number}/{len(batches)} failed after three "
                                f"Colab attempts; {completed} captions were saved locally. Rerun to resume."
                            ) from error
                        print(
                            f"Colab command failed; retrying batch {batch_number}/{len(batches)} "
                            f"in a new session (attempt {attempt + 1}/3).",
                            flush=True,
                        )
                        time.sleep(5 * attempt)
                        continue
                    _merge_coco_gpu_results(database, returned_database, batch_annotations, model_id)
                    completed += len(batch_ids)
                    break
        finally:
            if session_started:
                _stop_session(colab_command, "oauth2", session_name)
    return len(pending)


def _coco_caption_batches(annotations: dict, pending: list[int], batch_size: int) -> list[list[int]]:
    captions_by_id = {caption["id"]: caption for caption in annotations["captions"]}
    by_image: dict[int, list[int]] = {}
    for caption_id in pending:
        image_id = captions_by_id[caption_id]["image_id"]
        by_image.setdefault(image_id, []).append(caption_id)

    batches: list[list[int]] = []
    batch: list[int] = []
    for image_caption_ids in by_image.values():
        remaining = image_caption_ids
        while remaining:
            available = batch_size - len(batch)
            if len(remaining) > available and batch:
                batches.append(batch)
                batch = []
                continue
            batch.extend(remaining[:available])
            remaining = remaining[available:]
            if len(batch) == batch_size:
                batches.append(batch)
                batch = []
    if batch:
        batches.append(batch)
    return batches


def _annotations_for_pending(annotations: dict, pending_ids: set[int]) -> dict:
    captions = [caption for caption in annotations["captions"] if caption["id"] in pending_ids]
    image_ids = {caption["image_id"] for caption in captions}
    images = [image for image in annotations["images"] if image["id"] in image_ids]
    if len(captions) != len(pending_ids) or len({image["id"] for image in images}) != len(image_ids):
        raise ValueError("Selected COCO captions or their images are missing from the annotation file.")
    return {"images": images, "annotations": captions}


def _build_coco_job_archive(
    annotations: dict, image_directory: Path, image_names: list[str], archive_path: Path, model_id: str
) -> None:
    manifest = {"model_id": model_id, "images_directory": "images", "annotations": "captions.json"}
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("captions.json", json.dumps(annotations))
        archive.writestr("manifest.json", json.dumps(manifest))
        for backend_file in sorted((PROJECT_ROOT / "backend").rglob("*.py")):
            archive.write(backend_file, backend_file.relative_to(PROJECT_ROOT).as_posix())
        for image_name in image_names:
            archive.write(image_directory / image_name, f"images/{image_name}")


def _merge_coco_gpu_results(
    database: Path, returned_database: Path, annotations: dict, model_id: str
) -> None:
    if not returned_database.is_file():
        raise RuntimeError("Colab did not return the COCO database.")
    expected_captions = {caption["id"]: caption for caption in annotations["annotations"]}
    expected_ids = sorted(expected_captions)
    placeholders = ",".join("?" for _ in expected_ids)
    with closing(sqlite3.connect(f"{returned_database.resolve().as_uri()}?mode=ro", uri=True)) as remote:
        rows = remote.execute(
            "SELECT caption_id, image_id, caption, text_context, text_context_shape, embedding_model "
            f"FROM captions WHERE caption_id IN ({placeholders}) ORDER BY caption_id", expected_ids
        ).fetchall()
    if [row[0] for row in rows] != expected_ids or any(
        row[1:3] != (expected_captions[row[0]]["image_id"], expected_captions[row[0]]["caption"])
        or not row[3] or len(row[3]) != 77 * 768 * 4
        or row[4] != "[77, 768]" or row[5] != model_id
        for row in rows
    ):
        raise RuntimeError("Colab returned incomplete or invalid COCO text-context embeddings.")
    with closing(sqlite3.connect(database)) as local, local:
        local.execute("BEGIN IMMEDIATE")
        current = local.execute(
            f"SELECT caption_id, image_id, caption FROM captions WHERE caption_id IN ({placeholders}) "
            "ORDER BY caption_id", expected_ids
        ).fetchall()
        expected_inputs = [
            (caption_id, expected_captions[caption_id]["image_id"], expected_captions[caption_id]["caption"])
            for caption_id in expected_ids
        ]
        if current != expected_inputs:
            raise RuntimeError("COCO captions changed during GPU inference; no embeddings were applied.")
        local.executemany(
            "UPDATE captions SET text_context = ?, text_context_shape = ?, embedding_model = ? "
            "WHERE caption_id = ? AND text_context IS NULL",
            [(row[3], row[4], row[5], row[0]) for row in rows],
        )


def _extract_text_context(
    caption: str, image_path: Path, output_directory: Path, model_id: str, device: str | None
) -> np.ndarray:
    paths = extract_inner_embeddings(caption, image_path, output_directory, model_id=model_id, device=device)
    embedding = np.load(paths["text_context"], allow_pickle=False)
    return np.ascontiguousarray(embedding)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill COCO caption text-context embeddings.")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIRECTORY)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--limit", type=int, help="Process the first N missing captions on the GPU.")
    parser.add_argument("--batch-size", type=int, default=250,
                        help="Maximum captions per GPU upload (default: 250).")
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--device", help="Torch device, e.g. cuda:0 or cpu.")
    device_group.add_argument("--gpu", "-gpu", action="store_true", help="Run on a temporary Google Colab T4 GPU.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit is not None and not args.gpu:
        parser.error("--limit requires --gpu")
    if args.batch_size != 250 and not args.gpu:
        parser.error("--batch-size requires --gpu")
    runner = fill_coco_database_on_gpu if args.gpu else fill_coco_database
    options = {"limit": args.limit, "batch_size": args.batch_size} if args.gpu else {"device": args.device}
    count = runner(
        annotations_path=args.annotations, images_directory=args.images_dir,
        database_path=args.database, model_id=args.model_id, **options,
    )
    print(f"Stored text-context embeddings for {count} COCO captions in {args.database}.")


if __name__ == "__main__":
    main()
