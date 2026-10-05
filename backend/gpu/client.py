"""Run database embedding extraction or image toxicity scoring on a Colab GPU."""

import argparse
import json
import math
import shutil
import sqlite3
import subprocess
import tempfile
import uuid
import zipfile
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path, PureWindowsPath

from backend.paths import FAIRNESS_ROOT, PROJECT_ROOT

DEFAULT_DATABASE_PATH = FAIRNESS_ROOT / "fairness.sqlite3"
DEFAULT_GPU = "T4"
DEFAULT_MODEL_ID = "openai/clip-vit-large-patch14"
REMOTE_ARCHIVE_PATH = "/content/conf-gen-embedding-job.zip"
REMOTE_DATABASE_PATH = "/content/conf-gen-embedding-output.sqlite3"
REMOTE_DEPENDENCIES = (
    "huggingface-hub>=0.24.0",
    "numpy>=2.0.0",
    "pillow>=10.0.0",
    "sqlalchemy>=2.0",
    "transformers>=4.51.0",
    "tqdm>=4.67.0",
)
TASK_FIELDS = {
    "embeddings": ("inner_embedding", "inner_embedding_shape"),
    "image-toxicity": ("image_harmfulness",),
    "prompt-toxicity": ("prompt_harmfulness",),
}


def run_colab_embeddings(
    database_path: Path,
    output_path: Path,
    *,
    gpu: str = DEFAULT_GPU,
    model_id: str = DEFAULT_MODEL_ID,
    session_name: str | None = None,
    auth: str = "oauth2",
    high_memory: bool = False,
    overwrite: bool = False,
    task: str = "embeddings",
    limit: int | None = None,
    forward_hf_token: bool = False,
    tasks: Sequence[str] | None = None,
    hf_token: str | None = None,
    record_ids: Sequence[int] | None = None,
) -> Path:
    """Populate the selected output remotely and download the resulting database."""
    source_database = database_path.resolve()
    selected_tasks = tuple(tasks) if tasks is not None else (task,)
    destination_database = output_path.resolve()
    _validate_local_inputs(source_database, destination_database, overwrite)
    colab_command = _find_colab_command()
    active_session = session_name or f"conf-gen-{uuid.uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory(prefix="conf_gen_colab_") as temporary_directory:
        temporary_path = Path(temporary_directory)
        archive_path = temporary_path / "embedding-job.zip"
        downloaded_database = temporary_path / "embedded.sqlite3"
        expected_count = _build_job_archive(
            source_database, archive_path, model_id,
            task=task, limit=limit, forward_hf_token=forward_hf_token,
            tasks=selected_tasks, hf_token=hf_token,
            record_ids=record_ids,
        )

        session_started = False
        try:
            _start_session(colab_command, auth, active_session, gpu, high_memory)
            session_started = True
            _install_remote_dependencies(colab_command, auth, active_session)
            _run_command(
                _colab_prefix(colab_command, auth)
                + ["upload", "-s", active_session, str(archive_path), REMOTE_ARCHIVE_PATH]
            )
            _run_command(
                _colab_prefix(colab_command, auth)
                + [
                    "exec",
                    "-s",
                    active_session,
                    "--timeout",
                    "86400",
                    "-f",
                    str(PROJECT_ROOT / "backend" / "gpu" / "fairness_worker.py"),
                ]
            )
            _run_command(
                _colab_prefix(colab_command, auth)
                + ["download", "-s", active_session, REMOTE_DATABASE_PATH, str(downloaded_database)]
            )
            for selected_task in selected_tasks:
                _validate_completed_database(downloaded_database, task=selected_task, expected_count=expected_count)
            destination_database.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded_database, destination_database)
        finally:
            if session_started:
                _stop_session(colab_command, auth, active_session)

    return destination_database


def _validate_local_inputs(database_path: Path, output_path: Path, overwrite: bool) -> None:
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")
    if output_path == database_path and not overwrite:
        raise FileExistsError("Refusing to replace the source database without --overwrite.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output database already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Fail before allocating a GPU if Windows ACLs prevent saving the result.
    with tempfile.TemporaryFile(dir=output_path.parent):
        pass


def _find_colab_command() -> str:
    colab_command = shutil.which("colab")
    if colab_command is None:
        raise RuntimeError(
            "The Colab CLI is not installed. Install it on Linux, macOS, or WSL with "
            "`uv tool install google-colab-cli`."
        )
    return colab_command


def _build_job_archive(
    database_path: Path, archive_path: Path, model_id: str,
    *, task: str = "embeddings", limit: int | None = None, forward_hf_token: bool = False,
    tasks: Sequence[str] | None = None, hf_token: str | None = None,
    record_ids: Sequence[int] | None = None,
) -> int:
    selected_tasks = tuple(tasks) if tasks is not None else (task,)
    if not selected_tasks or any(item not in TASK_FIELDS for item in selected_tasks):
        raise ValueError(f"Unknown or empty task selection: {selected_tasks}")
    token = None
    if forward_hf_token:
        from huggingface_hub import get_token

        token = hf_token or get_token()
        if not token:
            raise RuntimeError("--forward-hf-token requires HF_TOKEN or a cached Hugging Face login.")
    with tempfile.TemporaryDirectory(prefix="conf_gen_snapshot_") as snapshot_directory:
        snapshot_path = Path(snapshot_directory) / "database.sqlite3"
        records = _snapshot_database(
            database_path, snapshot_path, limit=limit, require_images=False, record_ids=record_ids,
        )
        with closing(sqlite3.connect(snapshot_path)) as connection:
            if "embeddings" in selected_tasks:
                connection.execute("UPDATE entries SET inner_embedding = NULL")
                image_ids = {row[0] for row in records}
            elif "image-toxicity" in selected_tasks:
                image_ids = {row[0] for row in connection.execute("SELECT id FROM entries WHERE image_harmfulness IS NULL")}
            else:
                image_ids = set()
            connection.commit()
        image_records = [(record_id, image) for record_id, image in records if record_id in image_ids]
        for _, image in image_records:
            if not image.is_file():
                raise FileNotFoundError(f"Image not found: {image}")
        manifest = {
            "task": task,
            "tasks": selected_tasks,
            "model_id": model_id,
            "images": {
                str(record_id): f"images/{record_id}{image_path.suffix.lower()}"
                for record_id, image_path in image_records
            },
        }

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(snapshot_path, "database.sqlite3")
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
            if token:
                archive.writestr(".hf-token", token)
            for backend_file in sorted((PROJECT_ROOT / "backend").rglob("*.py")):
                archive.write(backend_file, backend_file.relative_to(PROJECT_ROOT).as_posix())
            for record_id, image_path in image_records:
                archive.write(image_path, manifest["images"][str(record_id)])
        return len(records)


def _snapshot_database(
    database_path: Path, snapshot_path: Path, *, limit: int | None = None, require_images: bool = True,
    record_ids: Sequence[int] | None = None,
) -> list[tuple[int, Path]]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive.")
    with closing(sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True)) as source_connection, closing(
        sqlite3.connect(snapshot_path)
    ) as snapshot_connection:
        source_connection.backup(snapshot_connection)
        if record_ids is not None:
            snapshot_connection.execute("CREATE TEMP TABLE selected_ids (id INTEGER PRIMARY KEY)")
            snapshot_connection.executemany("INSERT INTO selected_ids VALUES (?)", [(item,) for item in record_ids])
            snapshot_connection.execute("DELETE FROM entries WHERE id NOT IN (SELECT id FROM selected_ids)")
        if limit is not None:
            snapshot_connection.execute(
                "DELETE FROM entries WHERE id NOT IN (SELECT id FROM entries ORDER BY id LIMIT ?)",
                (limit,),
            )
        snapshot_connection.commit()
        rows = snapshot_connection.execute("SELECT id, image_path FROM entries ORDER BY id").fetchall()

    if not rows:
        raise ValueError("The database contains no records to process.")

    records = []
    for record_id, image_path in rows:
        image = Path(image_path)
        if not image.is_file():
            image = database_path.parent / "images" / PureWindowsPath(image_path).name
            if not image.is_file():
                image = PROJECT_ROOT / "data" / "validation" / "toxic_source" / PureWindowsPath(image_path).name
        records.append((int(record_id), image))
    missing_images = [image_path for _, image_path in records if not image_path.is_file()]
    if require_images and missing_images:
        raise FileNotFoundError(f"Image not found: {missing_images[0]}")
    return records


def _start_session(colab_command: str, auth: str, session_name: str, gpu: str, high_memory: bool) -> None:
    command = _colab_prefix(colab_command, auth) + ["new", "-s", session_name, "--gpu", gpu]
    if high_memory:
        command.append("--high-mem")
    _run_command(command)


def _install_remote_dependencies(colab_command: str, auth: str, session_name: str) -> None:
    _run_command(
        _colab_prefix(colab_command, auth) + ["install", "-s", session_name, *REMOTE_DEPENDENCIES]
    )


def _stop_session(colab_command: str, auth: str, session_name: str) -> None:
    try:
        _run_command(_colab_prefix(colab_command, auth) + ["stop", "-s", session_name])
    except subprocess.CalledProcessError as error:
        print(f"Warning: Colab session cleanup failed: {error}")


def _colab_prefix(colab_command: str, auth: str) -> list[str]:
    return [colab_command, "--auth", auth]


def _run_command(command: Sequence[str], *, timeout: int | None = None) -> None:
    subprocess.run(command, check=True, timeout=timeout)


def _validate_completed_database(
    database_path: Path, *, task: str = "embeddings", expected_count: int | None = None,
) -> None:
    if not database_path.is_file():
        raise RuntimeError("Colab did not return an output database.")
    with closing(sqlite3.connect(database_path)) as connection:
        actual_count = connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        if expected_count is not None and actual_count != expected_count:
            raise RuntimeError(f"Expected {expected_count} records, got {actual_count}.")
        if task in ("image-toxicity", "prompt-toxicity"):
            field = TASK_FIELDS[task][0]
            scores = connection.execute(f"SELECT {field} FROM entries").fetchall()
            if not scores or any(
                not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1
                for (score,) in scores
            ):
                raise RuntimeError(f"Colab returned missing or invalid {task} scores.")
            return
        total_count, embedded_count = connection.execute(
            "SELECT COUNT(*), COUNT(inner_embedding) FROM entries"
        ).fetchone()
    if total_count == 0 or embedded_count != total_count:
        raise RuntimeError(f"Expected {total_count} embeddings, but the output contains {embedded_count}.")


def _default_output_path(database_path: Path, task: str = "embeddings") -> Path:
    suffix = "toxicity" if task == "image-toxicity" else "embedded"
    return database_path.with_name(f"{database_path.stem}.{suffix}{database_path.suffix}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill embeddings or image toxicity scores on a Google Colab GPU.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH, help="SQLite database to upload.")
    parser.add_argument("--task", choices=("embeddings", "image-toxicity"), default="embeddings")
    parser.add_argument("--limit", type=int, help="Process only the first N records, ordered by ID.")
    parser.add_argument(
        "--forward-hf-token", action="store_true",
        help="Explicitly authorize uploading your HF_TOKEN or cached Hugging Face token to the temporary Colab runtime.",
    )
    parser.add_argument("--output", type=Path, help="Downloaded SQLite database path.")
    parser.add_argument("--gpu", choices=("T4", "L4", "G4", "H100", "A100"), default=DEFAULT_GPU)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face CLIP model ID.")
    parser.add_argument("--session", help="Optional Colab session name.")
    parser.add_argument("--auth", choices=("oauth2", "adc"), default="oauth2")
    parser.add_argument("--high-mem", action="store_true", help="Request a high-memory runtime where supported.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of an existing output database.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = args.output or _default_output_path(args.database, args.task)
    completed_database = run_colab_embeddings(
        args.database,
        output_path,
        gpu=args.gpu,
        model_id=args.model_id,
        session_name=args.session,
        auth=args.auth,
        high_memory=args.high_mem,
        overwrite=args.overwrite,
        task=args.task,
        limit=args.limit,
        forward_hf_token=args.forward_hf_token,
    )
    print(f"Downloaded GPU {args.task} database: {completed_database}")


if __name__ == "__main__":
    main()
