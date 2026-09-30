import json
import tempfile
import unittest
from pathlib import Path

from sqlalchemy.orm import Session

from backend.create_dataset_database import SafetyRecord, _create_engine, create_database


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


if __name__ == "__main__":
    unittest.main()
