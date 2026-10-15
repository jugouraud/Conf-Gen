import json
import sqlite3
import tempfile
import unittest
import zipfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from backend.gpu.client import (
    _build_job_archive,
    _default_output_path,
    _validate_completed_database,
    run_colab_embeddings,
)
from backend.gpu.fairness_worker import (
    _read_image_paths,
    _set_remote_image_paths,
    _write_image_paths,
)


class ColabEmbeddingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.image_path = self.directory / "sample.jpg"
        self.image_path.write_bytes(b"image")
        self.database_path = self.directory / "safety.sqlite3"
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                "CREATE TABLE entries (id INTEGER PRIMARY KEY, image_path TEXT, inner_embedding BLOB)"
            )
            connection.execute(
                "INSERT INTO entries (id, image_path) VALUES (?, ?)",
                (7, str(self.image_path)),
            )
            connection.commit()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_builds_archive_with_database_code_images_and_manifest(self) -> None:
        archive_path = self.directory / "job.zip"

        _build_job_archive(self.database_path, archive_path, "example/model")

        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
        self.assertIn("database.sqlite3", names)
        self.assertIn("backend/embeddings/fairness.py", names)
        self.assertIn("backend/validation/fill.py", names)
        self.assertIn("backend/gpu/merge.py", names)
        self.assertIn("images/7.jpg", names)
        self.assertEqual(manifest["model_id"], "example/model")
        self.assertEqual(manifest["images"], {"7": "images/7.jpg"})

    def test_stops_started_session_when_remote_command_fails(self) -> None:
        output_path = self.directory / "output.sqlite3"
        commands: list[list[str]] = []

        def run_command(command):
            commands.append(list(command))
            if "install" in command:
                raise RuntimeError("remote failure")

        with patch("backend.gpu.client._find_colab_command", return_value="colab"), patch(
            "backend.gpu.client._run_command", side_effect=run_command
        ), self.assertRaisesRegex(RuntimeError, "remote failure"):
            run_colab_embeddings(self.database_path, output_path, session_name="test-session")

        self.assertIn(["colab", "--auth", "oauth2", "stop", "-s", "test-session"], commands)

    def test_downloads_valid_database_and_stops_session(self) -> None:
        output_path = self.directory / "output.sqlite3"
        commands: list[list[str]] = []

        def run_command(command):
            commands.append(list(command))
            if "download" not in command:
                return
            downloaded_database = Path(command[-1])
            with closing(sqlite3.connect(downloaded_database)) as connection:
                connection.execute(
                    "CREATE TABLE entries (id INTEGER PRIMARY KEY, inner_embedding BLOB)"
                )
                connection.execute(
                    "INSERT INTO entries (id, inner_embedding) VALUES (?, ?)",
                    (7, b"embedding"),
                )
                connection.commit()

        with patch("backend.gpu.client._find_colab_command", return_value="colab"), patch(
            "backend.gpu.client._run_command", side_effect=run_command
        ):
            result = run_colab_embeddings(self.database_path, output_path, session_name="test-session")

        self.assertEqual(result, output_path.resolve())
        self.assertTrue(output_path.is_file())
        self.assertIn(["colab", "--auth", "oauth2", "stop", "-s", "test-session"], commands)

    def test_refuses_to_overwrite_existing_output_by_default(self) -> None:
        with self.assertRaisesRegex(FileExistsError, "source database"):
            run_colab_embeddings(self.database_path, self.database_path)

    def test_unwritable_output_fails_before_gpu_allocation(self) -> None:
        with patch("backend.gpu.client.tempfile.TemporaryFile", side_effect=PermissionError("unwritable")), patch(
            "backend.gpu.client._start_session"
        ) as start_session, self.assertRaises(PermissionError):
            run_colab_embeddings(self.database_path, self.directory / "output.sqlite3")
        start_session.assert_not_called()

    def test_default_output_keeps_database_extension(self) -> None:
        self.assertEqual(
            _default_output_path(Path("data/safety.sqlite3")),
            Path("data/safety.embedded.sqlite3"),
        )

    def test_toxicity_archive_limits_records_and_resolves_old_windows_paths(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("ALTER TABLE entries ADD COLUMN image_harmfulness REAL")
            connection.execute("UPDATE entries SET image_path = ?, image_harmfulness = 0.7", (r"C:\old\data\sample.jpg",))
            connection.execute("INSERT INTO entries (id, image_path) VALUES (8, 'missing.jpg')")
            connection.commit()
        archive_path = self.directory / "toxicity.zip"
        count = _build_job_archive(
            self.database_path, archive_path, "unused", task="image-toxicity", limit=1,
        )
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertFalse(any("token" in name.lower() for name in archive.namelist()))
            archive.extract("database.sqlite3", self.directory / "extracted")
        self.assertEqual(count, 1)
        self.assertEqual(manifest["task"], "image-toxicity")
        self.assertEqual(manifest["images"], {})
        with closing(sqlite3.connect(self.directory / "extracted/database.sqlite3")) as connection:
            self.assertEqual(connection.execute("SELECT id, image_harmfulness FROM entries").fetchall(), [(7, 0.7)])
        with closing(sqlite3.connect(self.database_path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT image_harmfulness FROM entries WHERE id = 7").fetchone()[0], 0.7)

    def test_toxicity_validation_rejects_missing_invalid_or_incomplete_scores(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("ALTER TABLE entries ADD COLUMN image_harmfulness REAL")
        for score in (None, -0.1, 1.1, float("inf")):
            with self.subTest(score=score):
                with closing(sqlite3.connect(self.database_path)) as connection:
                    connection.execute("UPDATE entries SET image_harmfulness = ?", (score,))
                    connection.commit()
                with self.assertRaisesRegex(RuntimeError, "invalid"):
                    _validate_completed_database(self.database_path, task="image-toxicity")
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("UPDATE entries SET image_harmfulness = 0.2")
            connection.commit()
        _validate_completed_database(self.database_path, task="image-toxicity", expected_count=1)
        with self.assertRaisesRegex(RuntimeError, "Expected 10 records"):
            _validate_completed_database(self.database_path, task="image-toxicity", expected_count=10)

    def test_prompt_only_job_does_not_require_or_upload_images(self):
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("ALTER TABLE entries ADD COLUMN prompt_harmfulness REAL")
            connection.execute("UPDATE entries SET image_path = 'missing.jpg', prompt_harmfulness = 0.5")
            connection.commit()
        archive_path = self.directory / "prompt.zip"
        _build_job_archive(self.database_path, archive_path, "unused", tasks=["prompt-toxicity"])
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["tasks"], ["prompt-toxicity"])
            self.assertEqual(manifest["images"], {})
            self.assertFalse(any(name.startswith("images/") for name in archive.namelist()))

    def test_combined_score_job_preserves_existing_outputs_and_embeddings(self):
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("ALTER TABLE entries ADD COLUMN prompt_harmfulness REAL")
            connection.execute("ALTER TABLE entries ADD COLUMN image_harmfulness REAL")
            connection.execute("UPDATE entries SET prompt_harmfulness = 0.1, image_harmfulness = 0.2, inner_embedding = ?", (b"keep",))
            connection.commit()
        archive_path = self.directory / "scores.zip"
        _build_job_archive(self.database_path, archive_path, "unused", tasks=["prompt-toxicity", "image-toxicity"])
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(json.loads(archive.read("manifest.json"))["tasks"], ["prompt-toxicity", "image-toxicity"])
            archive.extract("database.sqlite3", self.directory / "combined")
        with closing(sqlite3.connect(self.directory / "combined/database.sqlite3")) as connection:
            self.assertEqual(connection.execute("SELECT prompt_harmfulness, image_harmfulness, inner_embedding FROM entries").fetchone(),
                             (0.1, 0.2, b"keep"))

    def test_rejects_nonpositive_limit_before_starting_remote_session(self) -> None:
        with patch("backend.gpu.client._find_colab_command", return_value="colab"), patch(
            "backend.gpu.client._start_session"
        ) as start_session, self.assertRaisesRegex(ValueError, "positive"):
            run_colab_embeddings(self.database_path, self.directory / "out.sqlite3", limit=0)
        start_session.assert_not_called()

    def test_worker_selects_only_image_toxicity_and_requires_cuda(self) -> None:
        from backend.gpu.fairness_worker import _fill_embeddings

        with patch("backend.embeddings.fairness.fill_database", return_value=10) as fill_database:
            _fill_embeddings(self.database_path, "unused", task="image-toxicity")
        fill_database.assert_called_once_with(
            self.database_path, embedding_model_id="unused", device="cuda",
            fill_inner_embeddings=False, fill_image_harmfulness=True,
            fill_prompt_harmfulness=False,
        )

    def test_token_forwarding_requires_explicit_opt_in(self) -> None:
        archive_path = self.directory / "job.zip"
        with patch("huggingface_hub.get_token", return_value="test-token") as get_token:
            _build_job_archive(self.database_path, archive_path, "example/model")
            get_token.assert_not_called()
            with zipfile.ZipFile(archive_path) as archive:
                self.assertNotIn(".hf-token", archive.namelist())
            _build_job_archive(self.database_path, archive_path, "example/model", forward_hf_token=True)
            get_token.assert_called_once_with()
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(archive.read(".hf-token"), b"test-token")

    def test_remote_image_paths_can_be_restored(self) -> None:
        original_paths = _read_image_paths(self.database_path)

        with patch("backend.gpu.fairness_worker.WORK_DIRECTORY", self.directory):
            _set_remote_image_paths(self.database_path, {"7": "images/7.jpg"})
            self.assertEqual(
                _read_image_paths(self.database_path),
                [(str((self.directory / "images" / "7.jpg").resolve()), 7)],
            )
        _write_image_paths(self.database_path, original_paths)

        self.assertEqual(_read_image_paths(self.database_path), original_paths)


if __name__ == "__main__":
    unittest.main()
