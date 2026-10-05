"""Run database embedding extraction on a Google Colab GPU runtime."""

import argparse
import json
import shutil
import sqlite3
import subprocess
import tempfile
import uuid
import zipfile
from contextlib import closing
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "fairness_data.sqlite3"
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
BACKEND_FILES = (
    "__init__.py",
    "create_dataset_database.py",
    "extract_inner_embedding.py",
    "filling_database.py",
    "model_storage.py",
    "rate_harmfulness.py",
)


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
) -> Path:
    """Populate inner embeddings remotely and download the resulting database."""
    source_database = database_path.resolve()
    destination_database = output_path.resolve()
    _validate_local_inputs(source_database, destination_database, overwrite)
    colab_command = _find_colab_command()
    active_session = session_name or f"conf-gen-{uuid.uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory(prefix="conf_gen_colab_") as temporary_directory:
        temporary_path = Path(temporary_directory)
        archive_path = temporary_path / "embedding-job.zip"
        downloaded_database = temporary_path / "embedded.sqlite3"
        _build_job_archive(source_database, archive_path, model_id)

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
                    str(PROJECT_ROOT / "scripts" / "colab_embedding_worker.py"),
                ]
            )
            _run_command(
                _colab_prefix(colab_command, auth)
                + ["download", "-s", active_session, REMOTE_DATABASE_PATH, str(downloaded_database)]
            )
            _validate_completed_database(downloaded_database)
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


def _find_colab_command() -> str:
    colab_command = shutil.which("colab")
    if colab_command is None:
        raise RuntimeError(
            "The Colab CLI is not installed. Install it on Linux, macOS, or WSL with "
            "`uv tool install google-colab-cli`."
        )
    return colab_command


def _build_job_archive(database_path: Path, archive_path: Path, model_id: str) -> None:
    with tempfile.TemporaryDirectory(prefix="conf_gen_snapshot_") as snapshot_directory:
        snapshot_path = Path(snapshot_directory) / "database.sqlite3"
        records = _snapshot_database(database_path, snapshot_path)
        manifest = {
            "model_id": model_id,
            "images": {
                str(record_id): f"images/{record_id}{image_path.suffix.lower()}"
                for record_id, image_path in records
            },
        }

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(snapshot_path, "database.sqlite3")
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
            for backend_file in BACKEND_FILES:
                archive.write(PROJECT_ROOT / "backend" / backend_file, f"backend/{backend_file}")
            for record_id, image_path in records:
                archive.write(image_path, manifest["images"][str(record_id)])


def _snapshot_database(database_path: Path, snapshot_path: Path) -> list[tuple[int, Path]]:
    with closing(sqlite3.connect(database_path)) as source_connection:
        rows = source_connection.execute("SELECT id, image_path FROM entries ORDER BY id").fetchall()
        with closing(sqlite3.connect(snapshot_path)) as snapshot_connection:
            source_connection.backup(snapshot_connection)

    if not rows:
        raise ValueError("The database contains no records to embed.")

    records = [(int(record_id), Path(image_path)) for record_id, image_path in rows]
    missing_images = [image_path for _, image_path in records if not image_path.is_file()]
    if missing_images:
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


def _run_command(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def _validate_completed_database(database_path: Path) -> None:
    if not database_path.is_file():
        raise RuntimeError("Colab did not return an output database.")
    with closing(sqlite3.connect(database_path)) as connection:
        total_count, embedded_count = connection.execute(
            "SELECT COUNT(*), COUNT(inner_embedding) FROM entries"
        ).fetchone()
    if total_count == 0 or embedded_count != total_count:
        raise RuntimeError(f"Expected {total_count} embeddings, but the output contains {embedded_count}.")


def _default_output_path(database_path: Path) -> Path:
    return database_path.with_name(f"{database_path.stem}.embedded{database_path.suffix}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill database embeddings on a Google Colab GPU.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH, help="SQLite database to upload.")
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
    output_path = args.output or _default_output_path(args.database)
    completed_database = run_colab_embeddings(
        args.database,
        output_path,
        gpu=args.gpu,
        model_id=args.model_id,
        session_name=args.session,
        auth=args.auth,
        high_memory=args.high_mem,
        overwrite=args.overwrite,
    )
    print(f"Downloaded GPU-embedded database: {completed_database}")


if __name__ == "__main__":
    main()
