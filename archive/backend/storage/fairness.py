"""Create and seed the SQLAlchemy store for prompt-image safety data."""

import argparse
import json
from pathlib import Path

from sqlalchemy import (
    CheckConstraint,
    Float,
    Integer,
    LargeBinary,
    String,
    create_engine,
    inspect,
)
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from backend.paths import FAIRNESS_ROOT

DEFAULT_DATA_DIRECTORY = FAIRNESS_ROOT
DEFAULT_RECORDS_PATH = FAIRNESS_ROOT / "records.json"
DEFAULT_DATABASE_PATH = FAIRNESS_ROOT / "fairness.sqlite3"


class Base(DeclarativeBase):
    """Base class for the safety-data database models."""


class SafetyRecord(Base):
    """One prompt-image safety record, identified by its source JSON ID."""

    __tablename__ = "entries"
    __table_args__ = (
        CheckConstraint("prompt_harmfulness IS NULL OR prompt_harmfulness BETWEEN 0.0 AND 1.0"),
        CheckConstraint("image_harmfulness IS NULL OR image_harmfulness BETWEEN 0.0 AND 1.0"),
        CheckConstraint("inner_embedding_harmfulness IS NULL OR inner_embedding_harmfulness BETWEEN 0.0 AND 1.0"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[str | None] = mapped_column(String, unique=True, index=True, nullable=True)
    caption: Mapped[str] = mapped_column(String, nullable=False)
    image_path: Mapped[str] = mapped_column(String, nullable=False)
    inner_embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    inner_embedding_shape: Mapped[str | None] = mapped_column(String, nullable=True)
    prompt_harmfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_harmfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    inner_embedding_harmfulness: Mapped[float | None] = mapped_column(Float, nullable=True)


def create_database(
    *, records_path: str | Path = DEFAULT_RECORDS_PATH, database_path: str | Path = DEFAULT_DATABASE_PATH
) -> Path:
    """Create the database and insert prompt/image data from the JSON records.

    Existing inner embeddings and harmfulness scores are preserved when the
    database is reseeded.
    """
    source = Path(records_path)
    destination = Path(database_path)
    rows = [_record_to_row(record, source.parent) for record in _read_records(source)]

    destination.parent.mkdir(parents=True, exist_ok=True)
    engine = _create_engine(destination)
    Base.metadata.create_all(engine)
    ensure_source_id_column(engine)
    with Session(engine) as session:
        for record_id, caption, image_path in rows:
            record = session.get(SafetyRecord, record_id)
            if record is None:
                session.add(SafetyRecord(id=record_id, caption=caption, image_path=image_path))
            else:
                record.caption = caption
                record.image_path = image_path
        session.commit()
    engine.dispose()
    return destination


def _create_engine(database_path: str | Path):
    """Return a SQLAlchemy engine for a local SQLite database."""
    database_url = URL.create("sqlite", database=str(Path(database_path).resolve()))
    return create_engine(database_url)


def ensure_source_id_column(engine) -> None:
    """Add optional dataset identifiers without changing existing integer IDs."""
    with engine.begin() as connection:
        columns = {column["name"] for column in inspect(connection).get_columns("entries")}
        if "source_id" not in columns:
            connection.exec_driver_sql("ALTER TABLE entries ADD COLUMN source_id VARCHAR")
        connection.exec_driver_sql("CREATE UNIQUE INDEX IF NOT EXISTS ix_entries_source_id ON entries (source_id)")


def _read_records(records_path: Path) -> list[dict]:
    """Read the JSON record list from disk."""
    try:
        records = json.loads(records_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Data records file not found: {records_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Data records file is not valid JSON: {records_path}") from error
    if not isinstance(records, list):
        raise ValueError("Data records must be a JSON list.")  # noqa: TRY004
    return records


def _record_to_row(record: dict, data_directory: Path) -> tuple[int, str, str]:
    """Extract one validated database row from a dataset record."""
    record_id = record.get("id")
    caption = record.get("caption")
    reference = record.get("image")
    if isinstance(reference, list) and len(reference) == 1:
        reference = reference[0]

    if not isinstance(record_id, int) or isinstance(record_id, bool):
        raise ValueError("Each data record must have an integer id.")  # noqa: TRY004
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(f"Record {record_id} has no usable caption.")
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"Record {record_id} must contain one image path.")

    image_path = data_directory / "images" / Path(reference).name
    if not image_path.is_file():
        image_path = data_directory / Path(reference).name
    if not image_path.is_file():
        raise FileNotFoundError(f"Image for record {record_id} not found: {image_path}")
    return record_id, caption, str(image_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the prompt-image safety SQLite database.")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS_PATH, help="Source JSON records file.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH, help="SQLite database to create.")
    args = parser.parse_args()
    database = create_database(records_path=args.records, database_path=args.database)
    print(f"Created and seeded database: {database}")


if __name__ == "__main__":
    main()
