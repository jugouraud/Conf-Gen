import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from utils.data_preprocessing import create_coco_prompt_image_bundle, iter_coco_prompt_image_pairs


class CocoPromptImageBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.images_dir = self.root / "images"
        self.images_dir.mkdir()
        Image.new("RGB", (8, 6), "red").save(self.images_dir / "example.png")
        self.annotations = self.root / "captions.json"
        self.annotations.write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "example.png", "width": 8, "height": 6}],
                    "annotations": [
                        {"id": 10, "image_id": 1, "caption": " first caption "},
                        {"id": 11, "image_id": 1, "caption": "second caption"},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_iter_pairs_normalizes_and_selects_first_caption(self) -> None:
        pairs = list(iter_coco_prompt_image_pairs(self.images_dir, self.annotations, caption_policy="first"))

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["prompt"], "first caption")
        self.assertEqual(pairs[0]["caption_id"], 10)

    def test_copy_mode_preserves_relative_image_path(self) -> None:
        manifest = create_coco_prompt_image_bundle(
            self.images_dir, self.annotations, self.root / "bundle", image_mode="copy", caption_policy="first"
        )

        record = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(record["image"], "images/example.png")
        self.assertTrue((manifest.parent / record["image"]).is_file())
        self.assertNotIn("image_path", record)

    def test_resize_mode_writes_requested_square_size(self) -> None:
        manifest = create_coco_prompt_image_bundle(
            self.images_dir, self.annotations, self.root / "bundle", image_mode="resize", image_size=4, limit=1
        )

        record = json.loads(manifest.read_text(encoding="utf-8"))
        with Image.open(manifest.parent / record["image"]) as image:
            self.assertEqual(image.size, (4, 4))


if __name__ == "__main__":
    unittest.main()
