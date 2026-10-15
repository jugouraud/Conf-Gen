"""The validation command fills labels without making them embedding inputs."""

import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from backend.storage.fairness import create_database
from backend.validation.fill import (
    fill_validation_scores,
    fill_validation_scores_on_gpu,
    main,
)


class ValidationDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        directory = Path(self.temporary_directory.name)
        (directory / "image.jpg").touch()
        records = directory / "records.json"
        records.write_text(json.dumps([{"id": 1, "caption": "A caption", "image": "image.jpg"}]), encoding="utf-8")
        self.database = directory / "records.sqlite3"
        create_database(records_path=records, database_path=self.database)

    def test_scoring_preserves_existing_zero_and_never_changes_embedding(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("UPDATE entries SET prompt_harmfulness = 0.0, inner_embedding = X'0102'")
        with patch("backend.validation.fill.score_annotation") as text, patch(
            "backend.validation.fill.score_image", return_value={"violence": 0.6}
        ) as image:
            self.assertEqual(fill_validation_scores(
                self.database, fill_prompt_harmfulness=True, fill_image_harmfulness=True
            ), 1)
            self.assertEqual(fill_validation_scores(
                self.database, fill_prompt_harmfulness=True, fill_image_harmfulness=True
            ), 0)
        text.assert_not_called()
        image.assert_called_once()
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute(
                "SELECT prompt_harmfulness, image_harmfulness, inner_embedding FROM entries"
            ).fetchone(), (0.0, 0.6, b"\x01\x02"))

    def test_gpu_scoring_selects_only_validation_tasks(self) -> None:
        with patch("backend.validation.fill.run_selected_gpu_tasks", return_value=1) as runner:
            count = fill_validation_scores_on_gpu(
                self.database, fill_prompt_harmfulness=True, fill_image_harmfulness=True
            )
        self.assertEqual(count, 1)
        self.assertEqual(runner.call_args.kwargs["tasks"], ["prompt-toxicity", "image-toxicity"])

    def test_cli_routes_gpu_scoring_to_validation_runner(self) -> None:
        with patch.object(sys, "argv", ["validation_database", "--database", str(self.database), "--gpu", "--image-harmfulness"]), patch(
            "backend.validation.fill.fill_validation_scores_on_gpu", return_value=1
        ) as runner:
            main()
        self.assertEqual(runner.call_args.args, (self.database,))
        self.assertTrue(runner.call_args.kwargs["fill_image_harmfulness"])
        self.assertFalse(runner.call_args.kwargs["fill_prompt_harmfulness"])


if __name__ == "__main__":
    unittest.main()
