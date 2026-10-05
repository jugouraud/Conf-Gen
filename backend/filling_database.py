"""Populate safety scores and inner embeddings for every database record."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from create_dataset_database import DEFAULT_DATABASE_PATH, SafetyRecord, _create_engine
from extract_inner_embedding import DEFAULT_MODEL_ID, extract_inner_embeddings
from rate_harmfulness import score_annotation, score_image


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
    """Populate the selected model outputs for all records.

    ``inner_embedding_harmfulness`` is deliberately not changed because no
    scorer for that representation is currently available.
    """
    _require_selected_output(
        fill_inner_embeddings=fill_inner_embeddings,
        fill_prompt_harmfulness=fill_prompt_harmfulness,
        fill_image_harmfulness=fill_image_harmfulness,
    )
    engine = _create_engine(database_path)
    try:
        with Session(engine) as session:
            records = session.scalars(select(SafetyRecord).order_by(SafetyRecord.id)).all()
            if fill_inner_embeddings:
                _fill_inner_embeddings(session, records, embedding_model_id, device)
            if fill_prompt_harmfulness:
                _fill_prompt_harmfulness(session, records, device, token)
            if fill_image_harmfulness:
                _fill_image_harmfulness(session, records, device, token)
        return len(records)
    finally:
        engine.dispose()


def _require_selected_output(
    *, fill_inner_embeddings: bool, fill_prompt_harmfulness: bool, fill_image_harmfulness: bool
) -> None:
    """Ensure that the caller selected at least one database field to fill."""
    if not any((fill_inner_embeddings, fill_prompt_harmfulness, fill_image_harmfulness)):
        raise ValueError("Select at least one output to fill.")


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


def _fill_prompt_harmfulness(
    session: Session, records: list[SafetyRecord], device: str | None, token: str | None
) -> None:
    """Score and save prompt harmfulness for every record."""
    for record in tqdm(records, desc="Filling prompt harmfulness", unit="record"):
        record.prompt_harmfulness = _maximum_score(score_annotation(record.caption, device=device, token=token))
        session.commit()


def _fill_image_harmfulness(
    session: Session, records: list[SafetyRecord], device: str | None, token: str | None
) -> None:
    """Score and save image harmfulness for every record."""
    for record in tqdm(records, desc="Filling image harmfulness", unit="record"):
        record.image_harmfulness = _maximum_score(score_image(record.image_path, device=device, token=token))
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


def _maximum_score(scores: Mapping[str, Any]) -> float:
    """Return the highest validated policy-violation probability."""
    if not scores:
        raise ValueError("Harmfulness scorer did not return any policy scores.")
    return max(_validate_score(score) for score in scores.values())


def _validate_score(score: Any) -> float:
    """Validate and normalize one policy-violation probability."""
    if isinstance(score, bool) or not isinstance(score, (float, int)):
        raise ValueError("Harmfulness scorer did not return a numeric policy score.")
    score = float(score)
    if not 0.0 <= score <= 1.0:
        raise ValueError("Harmfulness scorer returned a policy score outside the [0, 1] range.")
    return score


def main() -> None:
    """Run database population from the command line."""
    parser = argparse.ArgumentParser(description="Fill selected safety-database fields for every record.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH, help="SQLite database to populate.")
    parser.add_argument("--embedding-model-id", default=DEFAULT_MODEL_ID, help="CLIP model used for text contexts.")
    parser.add_argument("--device", help="Torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--hf-token", help="Hugging Face access token. Defaults to HF_TOKEN.")
    parser.add_argument("--inner-embeddings", action="store_true", help="Fill inner-embedding fields.")
    parser.add_argument("--prompt-harmfulness", action="store_true", help="Fill prompt harmfulness.")
    parser.add_argument("--image-harmfulness", action="store_true", help="Fill image harmfulness.")
    args = parser.parse_args()

    if not any((args.inner_embeddings, args.prompt_harmfulness, args.image_harmfulness)):
        parser.error("select at least one fill flag")

    record_count = fill_database(
        args.database,
        embedding_model_id=args.embedding_model_id,
        device=args.device,
        token=args.hf_token,
        fill_inner_embeddings=args.inner_embeddings,
        fill_prompt_harmfulness=args.prompt_harmfulness,
        fill_image_harmfulness=args.image_harmfulness,
    )
    print(f"Filled selected model outputs for {record_count} database records.")


if __name__ == "__main__":
    main()
