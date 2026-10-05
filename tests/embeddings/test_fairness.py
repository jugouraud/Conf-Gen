import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sqlalchemy.orm import Session

from backend.embeddings.fairness import fill_database, fill_database_on_gpu, main
from backend.storage.fairness import SafetyRecord, _create_engine, create_database


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

        with patch("backend.embeddings.fairness.extract_inner_embeddings", side_effect=self._write_context(context)), patch(
            "backend.validation.fill.score_annotation", return_value={"sexual": 0.25}
        ), patch(
            "backend.validation.fill.score_image", return_value={"violence": 0.75}
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
        with patch("backend.validation.fill.score_annotation", return_value={"hate": 0.4}) as score_annotation, patch(
            "backend.validation.fill.score_image"
        ) as score_image, patch("backend.embeddings.fairness.extract_inner_embeddings") as extract_inner_embeddings:
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

    def test_gpu_merges_scores_into_original_and_preserves_other_local_changes(self):
        def remote_job(source, output, **options):
            self.assertEqual(options["gpu"], "T4")
            self.assertEqual(options["tasks"], ["prompt-toxicity", "image-toxicity"])
            self.assertTrue(options["forward_hf_token"])
            self.assertEqual(options["hf_token"], "test-token")
            with closing(sqlite3.connect(source)) as local, closing(sqlite3.connect(output)) as remote:
                local.backup(remote)
                remote.execute("UPDATE entries SET prompt_harmfulness = 0.2, image_harmfulness = 0.8")
                remote.commit()
                local.execute("UPDATE entries SET inner_embedding_harmfulness = 0.6")
                local.execute("INSERT INTO entries (id, caption, image_path) VALUES (10, 'new', 'new.jpg')")
                local.commit()
            return output

        with patch("backend.gpu.client.run_colab_embeddings", side_effect=remote_job) as run:
            count = fill_database_on_gpu(
                self.database_path, token="test-token", fill_prompt_harmfulness=True, fill_image_harmfulness=True,
            )
        self.assertEqual(count, 1)
        run.assert_called_once()
        with closing(sqlite3.connect(self.database_path)) as connection:
            self.assertEqual(connection.execute(
                "SELECT prompt_harmfulness, image_harmfulness, inner_embedding_harmfulness FROM entries WHERE id = 9"
            ).fetchone(), (0.2, 0.8, 0.6))
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0], 2)

    def test_gpu_refuses_stale_results_without_applying_scores(self):
        def remote_job(source, output, **options):
            with closing(sqlite3.connect(source)) as local, closing(sqlite3.connect(output)) as remote:
                local.backup(remote)
                remote.execute("UPDATE entries SET image_harmfulness = 0.8")
                remote.commit()
                local.execute("UPDATE entries SET caption = 'edited during inference'")
                local.commit()
            return output

        with patch("backend.gpu.client.run_colab_embeddings", side_effect=remote_job), self.assertRaisesRegex(RuntimeError, "inputs changed"):
            fill_database_on_gpu(self.database_path, fill_image_harmfulness=True)
        with closing(sqlite3.connect(self.database_path)) as connection:
            self.assertIsNone(connection.execute("SELECT image_harmfulness FROM entries").fetchone()[0])

    def test_gpu_failure_leaves_original_database_unchanged(self):
        original = self.database_path.read_bytes()
        with patch("backend.gpu.client.run_colab_embeddings", side_effect=RuntimeError("remote failed")), self.assertRaisesRegex(RuntimeError, "remote failed"):
            fill_database_on_gpu(self.database_path, fill_image_harmfulness=True)
        self.assertEqual(self.database_path.read_bytes(), original)

    def test_gpu_parser_routes_selected_outputs_to_default_database(self):
        from backend.storage.fairness import DEFAULT_DATABASE_PATH
        with patch.object(sys, "argv", ["filling_database", "--gpu", "--image-harmfulness"]), patch(
            "backend.embeddings.fairness.fill_database_on_gpu", return_value=10
        ) as gpu, patch("backend.embeddings.fairness.fill_database") as local:
            main()
        local.assert_not_called()
        self.assertEqual(gpu.call_args.args, (DEFAULT_DATABASE_PATH,))
        self.assertTrue(gpu.call_args.kwargs["fill_image_harmfulness"])
        self.assertFalse(gpu.call_args.kwargs["fill_prompt_harmfulness"])

    def test_local_parser_keeps_existing_behavior(self):
        with patch.object(sys, "argv", ["filling_database", "--prompt-harmfulness", "--device", "cpu"]), patch(
            "backend.embeddings.fairness.fill_database", return_value=1
        ) as local, patch("backend.embeddings.fairness.fill_database_on_gpu") as gpu:
            main()
        gpu.assert_not_called()
        self.assertEqual(local.call_args.kwargs["device"], "cpu")
        self.assertTrue(local.call_args.kwargs["fill_prompt_harmfulness"])

    def _seed_partial_scores(self):
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute("UPDATE entries SET prompt_harmfulness = 0.0 WHERE id = 9")
            connection.executemany(
                "INSERT INTO entries (id, caption, image_path, prompt_harmfulness, image_harmfulness) VALUES (?, ?, ?, ?, ?)",
                [(10, "text only", "missing.jpg", None, 0.0),
                 (11, "complete", "missing.jpg", 0.3, 0.4),
                 (12, "both", str(self.data_directory / "example.jpg"), None, None)],
            )

    def test_local_scoring_skips_filled_fields_independently_including_zero(self):
        self._seed_partial_scores()
        with patch("backend.validation.fill.score_annotation", return_value={"sexual": 0.2}) as text, patch(
            "backend.validation.fill.score_image", return_value={"violence": 0.8}
        ) as image:
            self.assertEqual(fill_database(self.database_path, fill_prompt_harmfulness=True, fill_image_harmfulness=True), 3)
            self.assertEqual(text.call_count, 2)
            self.assertEqual(image.call_count, 2)
            self.assertEqual(fill_database(self.database_path, fill_prompt_harmfulness=True, fill_image_harmfulness=True), 0)
            self.assertEqual(text.call_count, 2)
            self.assertEqual(image.call_count, 2)
        with closing(sqlite3.connect(self.database_path)) as connection:
            self.assertEqual(connection.execute("SELECT id, prompt_harmfulness, image_harmfulness FROM entries ORDER BY id").fetchall(),
                             [(9, 0.0, 0.8), (10, 0.2, 0.0), (11, 0.3, 0.4), (12, 0.2, 0.8)])

    def test_gpu_uploads_only_pending_rows_and_needed_images_and_preserves_scores(self):
        import zipfile

        from backend.gpu.client import _build_job_archive
        self._seed_partial_scores()

        def remote_job(source, output, **options):
            self.assertEqual(options["record_ids"], [9, 10, 12])
            archive_path = output.parent / "job.zip"
            _build_job_archive(source, archive_path, "unused", tasks=options["tasks"], record_ids=options["record_ids"])
            with zipfile.ZipFile(archive_path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(set(manifest["images"]), {"9", "12"})
                output.write_bytes(archive.read("database.sqlite3"))
            with closing(sqlite3.connect(output)) as remote, remote:
                self.assertEqual(remote.execute("SELECT id FROM entries ORDER BY id").fetchall(), [(9,), (10,), (12,)])
                remote.execute("UPDATE entries SET prompt_harmfulness = COALESCE(prompt_harmfulness, 0.2), image_harmfulness = COALESCE(image_harmfulness, 0.8)")
            with closing(sqlite3.connect(source)) as local, local:
                local.execute("UPDATE entries SET image_harmfulness = 0.6 WHERE id = 12")
            return output

        with patch("backend.gpu.client.run_colab_embeddings", side_effect=remote_job) as run:
            self.assertEqual(fill_database_on_gpu(self.database_path, fill_prompt_harmfulness=True, fill_image_harmfulness=True), 3)
            self.assertEqual(fill_database_on_gpu(self.database_path, fill_prompt_harmfulness=True, fill_image_harmfulness=True), 0)
            run.assert_called_once()
        with closing(sqlite3.connect(self.database_path)) as connection:
            self.assertEqual(connection.execute("SELECT id, prompt_harmfulness, image_harmfulness FROM entries ORDER BY id").fetchall(),
                             [(9, 0.0, 0.8), (10, 0.2, 0.0), (11, 0.3, 0.4), (12, 0.2, 0.6)])

    def test_gpu_checks_only_the_requested_score_for_pending_work(self):
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute("UPDATE entries SET image_harmfulness = 0.0")
        with patch("backend.gpu.client.run_colab_embeddings") as run:
            self.assertEqual(fill_database_on_gpu(self.database_path, fill_image_harmfulness=True), 0)
            run.assert_not_called()

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
