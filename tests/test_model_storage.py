import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.model_storage import get_local_model_directory


class ModelStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.models_directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_uses_existing_local_model_without_downloading(self) -> None:
        model_directory = self.models_directory / "openai--clip-vit-large-patch14"
        model_directory.mkdir()
        (model_directory / "config.json").touch()

        with patch("backend.model_storage.snapshot_download") as download:
            result = get_local_model_directory("openai/clip-vit-large-patch14", models_directory=self.models_directory)

        self.assertEqual(result, model_directory)
        download.assert_not_called()

    def test_downloads_missing_model_into_models_directory(self) -> None:
        with patch("backend.model_storage.snapshot_download") as download:
            result = get_local_model_directory(
                "google/shieldgemma-2b", token="test-token", models_directory=self.models_directory
            )

        expected_directory = self.models_directory / "google--shieldgemma-2b"
        self.assertEqual(result, expected_directory)
        self.assertTrue(expected_directory.is_dir())
        download.assert_called_once_with(
            repo_id="google/shieldgemma-2b", local_dir=expected_directory, token="test-token"
        )


if __name__ == "__main__":
    unittest.main()
