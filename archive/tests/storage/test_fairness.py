import json
import tempfile
import unittest
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.storage.fairness import (
    SafetyRecord,
    _create_engine,
    create_database,
    ensure_source_id_column,
)


class CreateDatasetDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_directory = Path(self.temp_dir.name)
        (self.data_directory / "example.jpg").touch()
        self.records_path = self.data_directory / "records.json"
        self.records_path.write_text(
            json.dumps(
                [{"id": 4, "caption": "Example caption", "image": ["source/example.jpg"]}]
            ),
            encoding="utf-8",
        )
        self.database_path = self.data_directory / "data.sqlite3"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_creates_rows_with_nullable_model_outputs(self) -> None:
        create_database(records_path=self.records_path, database_path=self.database_path)

        engine = _create_engine(self.database_path)
        with Session(engine) as session:
            record = session.get(SafetyRecord, 4)
        engine.dispose()

        self.assertIsNotNone(record)
        self.assertEqual(record.caption, "Example caption")
        self.assertEqual(Path(record.image_path), self.data_directory / "example.jpg")
        self.assertIsNone(record.inner_embedding)
        self.assertIsNone(record.prompt_harmfulness)
        self.assertIsNone(record.image_harmfulness)
        self.assertIsNone(record.inner_embedding_harmfulness)

    def test_source_id_migration_preserves_integer_ids_and_scores_and_enforces_uniqueness(self):
        engine = _create_engine(self.database_path)
        try:
            with engine.begin() as connection:
                connection.exec_driver_sql("CREATE TABLE entries (id INTEGER PRIMARY KEY, image_harmfulness REAL)")
                connection.exec_driver_sql("INSERT INTO entries VALUES (4, 0.7), (5, 0.2)")
            ensure_source_id_column(engine)
            ensure_source_id_column(engine)
            with engine.begin() as connection:
                self.assertEqual(connection.exec_driver_sql("SELECT id, image_harmfulness, source_id FROM entries ORDER BY id").fetchall(),
                                 [(4, 0.7, None), (5, 0.2, None)])
                connection.exec_driver_sql("UPDATE entries SET source_id = 'toxic_1' WHERE id = 4")
            with self.assertRaises(IntegrityError), engine.begin() as connection:
                connection.exec_driver_sql("UPDATE entries SET source_id = 'toxic_1' WHERE id = 5")
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
