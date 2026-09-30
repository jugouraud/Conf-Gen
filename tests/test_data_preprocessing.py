import json
import tempfile
import unittest
from pathlib import Path

from utils.data_preprocessing import PromptImagePair, load_prompt_image_pair


class PromptImagePairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_directory = Path(self.temp_dir.name)
        (self.data_directory / "example.jpg").touch()
        (self.data_directory / "hf_test_fairness_real.json").write_text(
            json.dumps(
                [
                    {
                        "id": 7,
                        "caption": "An example prompt.",
                        "image": ["evaluator_test/fairness_data/facet/example.jpg"],
                    }
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_loads_prompt_and_local_image_for_id(self) -> None:
        pair = load_prompt_image_pair(7, data_directory=self.data_directory)

        self.assertIsInstance(pair, PromptImagePair)
        self.assertEqual(pair.prompt, "An example prompt.")
        self.assertEqual(pair.image, self.data_directory / "example.jpg")

    def test_rejects_unknown_id(self) -> None:
        with self.assertRaises(KeyError):
            load_prompt_image_pair(8, data_directory=self.data_directory)


if __name__ == "__main__":
    unittest.main()
