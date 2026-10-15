"""Check COCO region calibration, scoring, and cache invalidation."""

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.embeddings.define_region import _load_eos_vectors, _nearest_distances, define_coco_region
from backend.storage.coco import create_empty_coco_database


class CocoRegionTests(unittest.TestCase):
    def test_reads_the_eos_row_from_stored_coco_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = create_empty_coco_database(Path(directory) / "coco.sqlite3")
            context = np.broadcast_to(np.arange(77, dtype=np.float32)[:, None], (77, 768)).copy()
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute("INSERT INTO images VALUES (7, '7.jpg')")
                connection.execute(
                    "INSERT INTO captions VALUES (101, 7, 'A test caption.', ?, '[77, 768]', ?)",
                    (context.tobytes(), "openai/clip-vit-large-patch14"),
                )

            class Tokenizer:
                def __call__(self, captions, **_):
                    assert captions == ["A test caption."]
                    token_ids = np.zeros((1, 77), dtype=np.int64)
                    token_ids[0, 3] = 99
                    return {"input_ids": token_ids}

            with (
                patch("backend.embeddings.define_region.get_local_model_directory", return_value=Path(directory)),
                patch("transformers.CLIPTokenizerFast.from_pretrained", return_value=Tokenizer()),
            ):
                caption_ids, image_ids, vectors = _load_eos_vectors(database, "openai/clip-vit-large-patch14")
            np.testing.assert_array_equal(caption_ids, [101])
            np.testing.assert_array_equal(image_ids, [7])
            np.testing.assert_array_equal(vectors, np.full((1, 768), 3, dtype=np.float32))

    def test_nearest_distances_match_direct_euclidean_distances(self) -> None:
        queries = np.array([[1, 0], [0, 3], [2, 2]], dtype=np.float32)
        anchors = np.array([[0, 0], [3, 0]], dtype=np.float32)
        expected = np.linalg.norm(queries[:, None, :] - anchors[None, :, :], axis=2).min(axis=1)
        np.testing.assert_allclose(_nearest_distances(queries, anchors, batch_size=2), expected, atol=1e-6)

    def test_region_is_calibrated_saved_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "coco.sqlite3"
            database.write_bytes(b"first version")
            artifact = Path(directory) / "region.npz"
            rng = np.random.default_rng(7)
            vectors = rng.normal(size=(80, 3)).astype(np.float32)
            caption_ids = np.arange(100, 180, dtype=np.int64)
            image_ids = np.repeat(np.arange(40, dtype=np.int64), 2)

            with patch("backend.embeddings.define_region._load_eos_vectors", return_value=(caption_ids, image_ids, vectors)) as load:
                region = define_coco_region(
                    database_path=database, region_path=artifact,
                    miscoverage=0.1, calibration_fraction=0.25, m_pca=1,
                )
                self.assertEqual(load.call_count, 1)
                self.assertEqual(region.metadata["calibration_rank"], 10)
                self.assertEqual(region.metadata["n_reference_images"], 30)
                self.assertEqual(region.metadata["n_calibration_images"], 10)
                self.assertTrue(artifact.is_file())
                self.assertEqual(region.contains(vectors[:1]).shape, (1,))
                self.assertTrue(np.all(region.score(vectors[:1]) >= 0))
                reference_ids = set(region.metadata["reference_caption_ids"])
                reference_mask = np.array([caption_id in reference_ids for caption_id in caption_ids])
                self.assertTrue(all(reference_mask[2 * i] == reference_mask[2 * i + 1] for i in range(40)))
                calibration = vectors[~reference_mask]
                calibration_scores = region.score(calibration)
                calibration_images = image_ids[~reference_mask]
                group_scores = [calibration_scores[calibration_images == image_id].max()
                                for image_id in np.unique(calibration_images)]
                self.assertAlmostEqual(region.radius, float(np.partition(group_scores, 9)[9]), places=5)

                reused = define_coco_region(
                    database_path=database, region_path=artifact,
                    miscoverage=0.1, calibration_fraction=0.25, m_pca=1,
                )
                self.assertEqual(load.call_count, 1)
                self.assertEqual(reused.radius, region.radius)
                np.testing.assert_array_equal(reused.anchors, region.anchors)

                database.write_bytes(b"changed database")
                os.utime(database, None)
                define_coco_region(
                    database_path=database, region_path=artifact,
                    miscoverage=0.1, calibration_fraction=0.25, m_pca=1,
                )
                self.assertEqual(load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
