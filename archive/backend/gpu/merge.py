"""Colab job selection and guarded merge for the prompt-image database."""

import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory


def run_selected_gpu_tasks(
    database_path: str | Path,
    *, embedding_model_id: str, token: str | None = None,
    tasks: list[str],
) -> int:
    """Run selected embedding or validation tasks and merge verified results."""
    from backend.gpu.client import TASK_FIELDS, run_colab_embeddings

    tasks = list(tasks)
    if not tasks or any(task not in TASK_FIELDS for task in tasks):
        raise ValueError("Select at least one valid output to fill.")
    source = Path(database_path).resolve()
    condition = " OR ".join(
        "1" if task == "embeddings" else f"{TASK_FIELDS[task][0]} IS NULL" for task in tasks
    )
    with closing(sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)) as connection:
        expected_inputs = connection.execute(
            f"SELECT id, caption, image_path FROM entries WHERE {condition} ORDER BY id"
        ).fetchall()
    if not expected_inputs:
        return 0
    with TemporaryDirectory(prefix="conf_gen_gpu_results_") as directory:
        result = Path(directory) / "result.sqlite3"
        run_colab_embeddings(
            source, result, gpu="T4", model_id=embedding_model_id, tasks=tasks,
            forward_hf_token=bool(token or any(task != "embeddings" for task in tasks)), hf_token=token,
            record_ids=[row[0] for row in expected_inputs],
        )
        fields = [field for task in tasks for field in TASK_FIELDS[task]]
        _merge_gpu_results(source, result, fields, expected_inputs)
    return len(expected_inputs)


def _merge_gpu_results(source: Path, result: Path, fields: list[str], expected_inputs: list[tuple]) -> None:
    """Commit selected outputs together, rejecting stale or mismatched inputs."""
    with closing(sqlite3.connect(f"{result.resolve().as_uri()}?mode=ro", uri=True)) as remote:
        inputs = remote.execute("SELECT id, caption, image_path FROM entries ORDER BY id").fetchall()
        if inputs != expected_inputs:
            raise RuntimeError("Colab returned records that do not match the submitted database.")
        updates = remote.execute(f"SELECT {', '.join(fields)}, id FROM entries ORDER BY id").fetchall()
    with closing(sqlite3.connect(source)) as local, local:
        local.execute("BEGIN IMMEDIATE")
        current = {row[0]: row[1:] for row in local.execute("SELECT id, caption, image_path FROM entries")}
        if any(current.get(row[0]) != row[1:] for row in expected_inputs):
            raise RuntimeError("Database inputs changed during GPU inference; no scores were applied.")
        assignments = ", ".join(
            f"{field} = COALESCE({field}, ?)" if field in ("prompt_harmfulness", "image_harmfulness")
            else f"{field} = ?" for field in fields
        )
        local.executemany(f"UPDATE entries SET {assignments} WHERE id = ?", updates)


