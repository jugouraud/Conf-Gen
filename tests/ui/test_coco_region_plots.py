"""Checks for the cached COCO 3D views and full-region display labels."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.embeddings.define_region import CocoRegion, _source_signature
from backend.embeddings.project_region import project_coco_region
from frontend.coco_region_plots import CocoRegionVisualizationController, load_region_visualization_data
from frontend.coco_region_validation import ValidationResult


class CocoRegionVisualizationTests(unittest.TestCase):
    def test_cache_is_reused_and_toggles_precomputed_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, artifact, cache = root / "coco.sqlite3", root / "region.npz", root / "views.sqlite3"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE captions (caption_id INTEGER PRIMARY KEY, caption TEXT NOT NULL)")
                connection.executemany("INSERT INTO captions VALUES (?, ?)",
                                       [(i, f"Caption {i}") for i in range(8)])
                connection.commit()
            artifact.write_bytes(b"region placeholder")
            vectors = np.array([
                [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                [0.1, 0, 0, 0], [2, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 2, 0],
            ], dtype=np.float32)
            region = CocoRegion(
                anchors=vectors[:4], mean=np.zeros(4, dtype=np.float32),
                components=np.array([[1], [0], [0], [0]], dtype=np.float32),
                eigenvalues=np.ones(1, dtype=np.float32), gamma=1.0, radius=0.5,
                metadata={
                    "source": _source_signature(database), "model_id": "test",
                    "reference_caption_ids": [0, 1, 2, 3], "miscoverage": 0.05,
                },
            )
            with patch("backend.embeddings.project_region._load_region", return_value=region), patch(
                "backend.embeddings.project_region._load_eos_vectors",
                return_value=(np.arange(8), np.arange(8), vectors),
            ) as load:
                project_coco_region(database_path=database, region_path=artifact, output_path=cache)
                project_coco_region(database_path=database, region_path=artifact, output_path=cache)
                self.assertEqual(load.call_count, 1)

                with closing(sqlite3.connect(cache)) as connection:
                    summary = json.loads(connection.execute(
                        "SELECT value FROM metadata WHERE key = 'region_summary'"
                    ).fetchone()[0])
                    self.assertEqual(len(summary["pca_explained_variance_ratio"]), 3)
                    summary.pop("pca_explained_variance_ratio")
                    connection.execute("UPDATE metadata SET value = ? WHERE key = 'region_summary'", (json.dumps(summary),))
                    connection.commit()
                with patch("backend.embeddings.project_region.TSNE.fit_transform", side_effect=AssertionError("t-SNE was recomputed")):
                    project_coco_region(database_path=database, region_path=artifact, output_path=cache)
                self.assertEqual(load.call_count, 2)
                with closing(sqlite3.connect(cache)) as connection:
                    connection.execute("DELETE FROM metadata WHERE key = 'pca_model'")
                    connection.commit()
                with patch("backend.embeddings.project_region.TSNE.fit_transform", side_effect=AssertionError("t-SNE was recomputed")):
                    project_coco_region(database_path=database, region_path=artifact, output_path=cache)
                self.assertEqual(load.call_count, 3)

            data = load_region_visualization_data(database, artifact, cache, max_anchors=4, max_calibration=4)
            controller = CocoRegionVisualizationController(data)
            pca = controller.build_figure()
            controller.set_projection_method("t-SNE")
            tsne = controller.build_figure()

            self.assertEqual([len(trace["x"]) for trace in pca["data"]], [4, 2, 2])
            self.assertEqual([trace["customdata"][0][0] for trace in pca["data"]], [0, 4, 5])
            self.assertIn("variance", pca["layout"]["scene"]["zaxis"]["title"])
            self.assertEqual(tsne["layout"]["scene"]["zaxis"]["title"], "t-SNE 3")
            self.assertFalse(np.array_equal(data.pca_coordinates, data.tsne_coordinates))
            self.assertEqual(data.pca_explained_variance_ratio.shape, (3,))

            result = ValidationResult(
                prompt="A test caption", score=1.2, radius=0.5, blocked=True,
                pca_coordinates=np.array([3.0, 4.0, 5.0]),
                tsne_coordinates=np.array([6.0, 7.0, 8.0]),
                nearest_caption_id=0,
                reference_caption_ids=np.arange(4),
                anchor_distances=np.array([1.2, 1.5, 1.8, 2.0]),
            )
            controller.set_validation(result)
            checked_tsne = controller.build_figure()
            self.assertIn("BLOCK", checked_tsne["layout"]["title"]["text"])
            self.assertEqual(checked_tsne["data"][-1]["z"], [8.0])
            np.testing.assert_allclose(checked_tsne["data"][0]["marker"]["color"], [1.2, 1.5, 1.8, 2.0])
            controller.set_projection_method("PCA")
            self.assertEqual(controller.build_figure()["data"][-1]["z"], [5.0])


if __name__ == "__main__":
    unittest.main()
