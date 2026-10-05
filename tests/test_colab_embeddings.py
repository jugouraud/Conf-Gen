import json
import sqlite3
import tempfile
import unittest
import zipfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from scripts.colab_embeddings import _build_job_archive, _default_output_path, run_colab_embeddings
from scripts.colab_embedding_worker import (
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
        self.assertIn("backend/filling_database.py", names)
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

        with patch("scripts.colab_embeddings._find_colab_command", return_value="colab"), patch(
            "scripts.colab_embeddings._run_command", side_effect=run_command
        ):
            with self.assertRaisesRegex(RuntimeError, "remote failure"):
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

        with patch("scripts.colab_embeddings._find_colab_command", return_value="colab"), patch(
            "scripts.colab_embeddings._run_command", side_effect=run_command
        ):
            result = run_colab_embeddings(self.database_path, output_path, session_name="test-session")

        self.assertEqual(result, output_path.resolve())
        self.assertTrue(output_path.is_file())
        self.assertIn(["colab", "--auth", "oauth2", "stop", "-s", "test-session"], commands)

    def test_refuses_to_overwrite_existing_output_by_default(self) -> None:
        with self.assertRaisesRegex(FileExistsError, "source database"):
            run_colab_embeddings(self.database_path, self.database_path)

    def test_default_output_keeps_database_extension(self) -> None:
        self.assertEqual(
            _default_output_path(Path("data/safety.sqlite3")),
            Path("data/safety.embedded.sqlite3"),
        )

    def test_remote_image_paths_can_be_restored(self) -> None:
        original_paths = _read_image_paths(self.database_path)

        with patch("scripts.colab_embedding_worker.WORK_DIRECTORY", self.directory):
            _set_remote_image_paths(self.database_path, {"7": "images/7.jpg"})
            self.assertEqual(
                _read_image_paths(self.database_path),
                [(str((self.directory / "images" / "7.jpg").resolve()), 7)],
            )
        _write_image_paths(self.database_path, original_paths)

        self.assertEqual(_read_image_paths(self.database_path), original_paths)


if __name__ == "__main__":
    unittest.main()
