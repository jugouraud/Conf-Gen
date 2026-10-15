"""Safety validation scores for existing prompt-image records.

Scores are evaluation labels and must not be supplied to candidate predictors.
"""

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from backend.embeddings.clip import DEFAULT_MODEL_ID
from backend.gpu.merge import run_selected_gpu_tasks
from backend.storage.fairness import DEFAULT_DATABASE_PATH, SafetyRecord, _create_engine
from backend.validation.shieldgemma import score_annotation, score_image


def fill_validation_scores(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *, device: str | None = None, token: str | None = None,
    fill_prompt_harmfulness: bool = False, fill_image_harmfulness: bool = False,
) -> int:
    """Fill missing validation scores, preserving scores already stored."""
    if not (fill_prompt_harmfulness or fill_image_harmfulness):
        raise ValueError("Select at least one validation score to fill.")
    missing = []
    if fill_prompt_harmfulness:
        missing.append(SafetyRecord.prompt_harmfulness.is_(None))
    if fill_image_harmfulness:
        missing.append(SafetyRecord.image_harmfulness.is_(None))
    engine = _create_engine(database_path)
    try:
        with Session(engine) as session:
            records = session.scalars(
                select(SafetyRecord).where(or_(*missing)).order_by(SafetyRecord.id)
            ).all()
            if fill_prompt_harmfulness:
                _fill_prompt_harmfulness(session, records, device, token)
            if fill_image_harmfulness:
                _fill_image_harmfulness(session, records, device, token)
            return len(records)
    finally:
        engine.dispose()


def fill_validation_scores_on_gpu(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *, token: str | None = None,
    fill_prompt_harmfulness: bool = False, fill_image_harmfulness: bool = False,
) -> int:
    """Score missing validation labels on Colab and merge them locally."""
    tasks = [name for name, selected in (
        ("prompt-toxicity", fill_prompt_harmfulness),
        ("image-toxicity", fill_image_harmfulness),
    ) if selected]
    if not tasks:
        raise ValueError("Select at least one validation score to fill.")
    return run_selected_gpu_tasks(database_path, embedding_model_id=DEFAULT_MODEL_ID, token=token, tasks=tasks)


def _fill_prompt_harmfulness(
    session: Session, records: list[SafetyRecord], device: str | None, token: str | None
) -> None:
    """Score only records whose prompt harmfulness is missing."""
    records = [record for record in records if record.prompt_harmfulness is None]
    for record in tqdm(records, desc="Filling prompt harmfulness", unit="record"):
        record.prompt_harmfulness = _maximum_score(score_annotation(record.caption, device=device, token=token))
        session.commit()


def _fill_image_harmfulness(
    session: Session, records: list[SafetyRecord], device: str | None, token: str | None
) -> None:
    """Score only records whose image harmfulness is missing."""
    records = [record for record in records if record.image_harmfulness is None]
    for record in tqdm(records, desc="Filling image harmfulness", unit="record"):
        record.image_harmfulness = _maximum_score(score_image(record.image_path, device=device, token=token))
        session.commit()


def _maximum_score(scores: Mapping[str, Any]) -> float:
    """Return the highest validated policy-violation probability."""
    if not scores:
        raise ValueError("Harmfulness scorer did not return any policy scores.")
    return max(_validate_score(score) for score in scores.values())


def _validate_score(score: Any) -> float:
    """Validate and normalize one policy-violation probability."""
    if isinstance(score, bool) or not isinstance(score, (float, int)):
        raise ValueError("Harmfulness scorer did not return a numeric policy score.")  # noqa: TRY004
    score = float(score)
    if not 0.0 <= score <= 1.0:
        raise ValueError("Harmfulness scorer returned a policy score outside the [0, 1] range.")
    return score


def main() -> None:
    """Score validation labels independently of embedding extraction."""
    parser = argparse.ArgumentParser(description="Fill missing safety validation scores.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--device", help="Local Torch device, e.g. cuda or cpu.")
    parser.add_argument("--gpu", action="store_true", help="Score on a Colab T4 and merge into the local database.")
    parser.add_argument("--hf-token", help="Hugging Face token. Defaults to HF_TOKEN or cached login.")
    parser.add_argument("--prompt-harmfulness", action="store_true")
    parser.add_argument("--image-harmfulness", action="store_true")
    args = parser.parse_args()
    if not (args.prompt_harmfulness or args.image_harmfulness):
        parser.error("select at least one validation score")
    if args.gpu and args.device:
        parser.error("--device is for local inference; --gpu uses a Colab T4")
    runner = fill_validation_scores_on_gpu if args.gpu else fill_validation_scores
    options = {} if args.gpu else {"device": args.device}
    count = runner(
        args.database, token=args.hf_token,
        fill_prompt_harmfulness=args.prompt_harmfulness,
        fill_image_harmfulness=args.image_harmfulness, **options,
    )
    print(f"Filled validation scores for {count} database records.")


if __name__ == "__main__":
    main()
