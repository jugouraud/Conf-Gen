import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sqlalchemy.orm import Session

from backend.create_dataset_database import SafetyRecord, _create_engine, create_database
from backend.filling_database import fill_database


class FillingDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_directory = Path(self.temporary_directory.name)
        (self.data_directory / "example.jpg").touch()
        records_path = self.data_directory / "records.json"
        records_path.write_text(
            json.dumps([{"id": 9, "caption": "Example caption", "image": "example.jpg"}]),
            encoding="utf-8",
        )
        self.database_path = self.data_directory / "data.sqlite3"
        create_database(records_path=records_path, database_path=self.database_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_fills_selected_outputs_without_changing_inner_embedding_harmfulness(self) -> None:
        context = np.arange(6, dtype=np.float32).reshape(2, 3)
        self._set_inner_embedding_harmfulness(0.5)

        with patch("backend.filling_database.extract_inner_embeddings", side_effect=self._write_context(context)), patch(
            "backend.filling_database.score_annotation", return_value={"sexual": 0.25}
        ), patch(
            "backend.filling_database.score_image", return_value={"violence": 0.75}
        ):
            record_count = fill_database(
                self.database_path,
                device="cpu",
                fill_inner_embeddings=True,
                fill_prompt_harmfulness=True,
                fill_image_harmfulness=True,
            )

        engine = _create_engine(self.database_path)
        with Session(engine) as session:
            record = session.get(SafetyRecord, 9)
        engine.dispose()

        self.assertEqual(record_count, 1)
        self.assertEqual(record.inner_embedding, context.tobytes())
        self.assertEqual(record.inner_embedding_shape, "[2, 3]")
        self.assertEqual(record.prompt_harmfulness, 0.25)
        self.assertEqual(record.image_harmfulness, 0.75)
        self.assertEqual(record.inner_embedding_harmfulness, 0.5)

    def test_fills_only_the_requested_field(self) -> None:
        with patch("backend.filling_database.score_annotation", return_value={"hate": 0.4}) as score_annotation, patch(
            "backend.filling_database.score_image"
        ) as score_image, patch("backend.filling_database.extract_inner_embeddings") as extract_inner_embeddings:
            fill_database(self.database_path, fill_prompt_harmfulness=True)

        engine = _create_engine(self.database_path)
        with Session(engine) as session:
            record = session.get(SafetyRecord, 9)
        engine.dispose()

        self.assertEqual(record.prompt_harmfulness, 0.4)
        self.assertIsNone(record.image_harmfulness)
        self.assertIsNone(record.inner_embedding)
        score_annotation.assert_called_once()
        score_image.assert_not_called()
        extract_inner_embeddings.assert_not_called()

    def _set_inner_embedding_harmfulness(self, score: float) -> None:
        engine = _create_engine(self.database_path)
        with Session(engine) as session:
            record = session.get(SafetyRecord, 9)
            record.inner_embedding_harmfulness = score
            session.commit()
        engine.dispose()

    @staticmethod
    def _write_context(context: np.ndarray):
        def write_context(_caption, _image_path, output_directory, **_kwargs):
            output_directory = Path(output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            context_path = output_directory / "text_context.npy"
            np.save(context_path, context)
            return {"text_context": context_path}

        return write_context


if __name__ == "__main__":
    unittest.main()
