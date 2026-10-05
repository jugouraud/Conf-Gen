"""Checks image-to-caption mapping for live COCO validation."""

import json
import tempfile
import unittest
from pathlib import Path

from frontend.coco_region_validation import load_validation_items


class ValidationItemTests(unittest.TestCase):
    def test_dataset_caption_records_match_image_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            (images / "First_1.jpg").touch()
            (images / "Second_2.png").touch()
            mapping = root / "captions.json"
            mapping.write_text(json.dumps([
                {"image": ["source/subdir/First_1.jpg"], "caption": "The first prompt"},
                {"image": ["source/subdir/Second_2.png"], "caption": "The second prompt"},
            ]), encoding="utf-8")

            items = load_validation_items(images, mapping)

            self.assertEqual([item.prompt for item in items], ["The first prompt", "The second prompt"])
            self.assertTrue(all(item.prompt_source == "Dataset caption" for item in items))


if __name__ == "__main__":
    unittest.main()
