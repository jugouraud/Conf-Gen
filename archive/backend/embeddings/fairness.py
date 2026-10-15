"""Extract inner embeddings for database records; retain the older combined CLI."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from backend.embeddings.clip import DEFAULT_MODEL_ID, extract_inner_embeddings
from backend.gpu.merge import run_selected_gpu_tasks
from backend.storage.fairness import DEFAULT_DATABASE_PATH, SafetyRecord, _create_engine
from backend.validation.fill import fill_validation_scores


def fill_database(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    embedding_model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
    token: str | None = None,
    fill_inner_embeddings: bool = False,
    fill_prompt_harmfulness: bool = False,
    fill_image_harmfulness: bool = False,
) -> int:
    """Fill embeddings, with score flags retained for existing callers.

    New validation work should use ``backend.validation.fill`` directly.
    """
    _require_selected_output(
        fill_inner_embeddings=fill_inner_embeddings,
        fill_prompt_harmfulness=fill_prompt_harmfulness,
        fill_image_harmfulness=fill_image_harmfulness,
    )
    if fill_inner_embeddings:
        engine = _create_engine(database_path)
        try:
            with Session(engine) as session:
                records = session.scalars(select(SafetyRecord).order_by(SafetyRecord.id)).all()
                _fill_inner_embeddings(session, records, embedding_model_id, device)
                count = len(records)
        finally:
            engine.dispose()
    else:
        count = 0
    if fill_prompt_harmfulness or fill_image_harmfulness:
        scored = fill_validation_scores(
            database_path, device=device, token=token,
            fill_prompt_harmfulness=fill_prompt_harmfulness,
            fill_image_harmfulness=fill_image_harmfulness,
        )
        if not fill_inner_embeddings:
            count = scored
    return count


def _require_selected_output(
    *, fill_inner_embeddings: bool, fill_prompt_harmfulness: bool, fill_image_harmfulness: bool
) -> None:
    """Ensure that the caller selected at least one database field to fill."""
    if not any((fill_inner_embeddings, fill_prompt_harmfulness, fill_image_harmfulness)):
        raise ValueError("Select at least one output to fill.")


def fill_database_on_gpu(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *, embedding_model_id: str = DEFAULT_MODEL_ID, token: str | None = None,
    fill_inner_embeddings: bool = False, fill_prompt_harmfulness: bool = False,
    fill_image_harmfulness: bool = False,
) -> int:
    """Compatibility entry point for combined Colab output jobs."""
    tasks = [name for name, selected in (
        ("embeddings", fill_inner_embeddings),
        ("prompt-toxicity", fill_prompt_harmfulness),
        ("image-toxicity", fill_image_harmfulness),
    ) if selected]
    if not tasks:
        raise ValueError("Select at least one output to fill.")
    return run_selected_gpu_tasks(database_path, embedding_model_id=embedding_model_id, token=token, tasks=tasks)


def _fill_inner_embeddings(
    session: Session, records: list[SafetyRecord], embedding_model_id: str, device: str | None
) -> None:
    """Extract and save an inner embedding for every record."""
    with TemporaryDirectory(prefix="conf_gen_embeddings_") as temporary_directory:
        for record in tqdm(records, desc="Filling inner embeddings", unit="record"):
            embedding = _extract_embedding(record, Path(temporary_directory), embedding_model_id, device)
            record.inner_embedding = embedding.tobytes()
            record.inner_embedding_shape = json.dumps(list(embedding.shape))
            session.commit()


def _extract_embedding(
    record: SafetyRecord, temporary_directory: Path, embedding_model_id: str, device: str | None
) -> np.ndarray:
    """Extract the text-context embedding that is stored in the database."""
    output_directory = temporary_directory / str(record.id)
    paths = extract_inner_embeddings(
        record.caption,
        record.image_path,
        output_directory,
        model_id=embedding_model_id,
        device=device,
    )
    embedding = np.load(paths["text_context"], allow_pickle=False)
    return np.ascontiguousarray(embedding)


def main() -> None:
    """Run database population from the command line."""
    parser = argparse.ArgumentParser(description="Extract embeddings or fill missing safety scores.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH, help="SQLite database to populate.")
    parser.add_argument("--embedding-model-id", default=DEFAULT_MODEL_ID, help="CLIP model used for text contexts.")
    parser.add_argument("--device", help="Torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--gpu", action="store_true",
        help="Run on a Colab T4 and save selected outputs in this database; forwards Hugging Face auth for scoring.",
    )
    parser.add_argument("--hf-token", help="Hugging Face access token. Defaults to HF_TOKEN.")
    parser.add_argument("--inner-embeddings", action="store_true", help="Fill inner-embedding fields.")
    parser.add_argument("--prompt-harmfulness", action="store_true", help="Fill missing prompt harmfulness scores.")
    parser.add_argument("--image-harmfulness", action="store_true", help="Fill missing image harmfulness scores.")
    args = parser.parse_args()

    if not any((args.inner_embeddings, args.prompt_harmfulness, args.image_harmfulness)):
        parser.error("select at least one fill flag")

    if args.gpu and args.device:
        parser.error("--device is for local inference; --gpu uses a Colab T4")
    runner = fill_database_on_gpu if args.gpu else fill_database
    options = {} if args.gpu else {"device": args.device}
    record_count = runner(
        args.database,
        embedding_model_id=args.embedding_model_id,
        token=args.hf_token,
        fill_inner_embeddings=args.inner_embeddings,
        fill_prompt_harmfulness=args.prompt_harmfulness,
        fill_image_harmfulness=args.image_harmfulness,
        **options,
    )
    print(f"Filled selected model outputs for {record_count} database records.")


if __name__ == "__main__":
    main()
