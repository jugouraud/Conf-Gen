"""Check the COCO transportation scores against the research definitions."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np

from backend.embeddings.define_region import CocoRegion
from backend.embeddings.transport_budget import TransportBudget, _order1_scores, order1_score
from research.wasserstein import score_order1, score_orderinf, task_wasserstein1


class TransportBudgetTests(unittest.TestCase):
    def test_both_scores_use_the_research_definitions(self) -> None:
        anchors = np.asarray([
            [0, 0], [1, 0], [0, 1], [2, 1], [4, 0], [1, 3],
        ], dtype=np.float32)
        clouds = [np.asarray([[float(i), 0], [float(i), 1]], dtype=np.float32)
                  for i in range(len(anchors))]
        query = np.asarray([[1.1, 0.1], [1.2, 0.9]], dtype=np.float32)
        vector = np.asarray([0.8, 0.4], dtype=np.float32)
        region = CocoRegion(
            anchors=anchors, mean=np.zeros(2, dtype=np.float32),
            components=np.asarray([[1], [0]], dtype=np.float32),
            eigenvalues=np.ones(1, dtype=np.float32), gamma=1.0,
            radius=1.0, metadata={"reference_caption_ids": list(range(6))},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.sqlite3"
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.executescript("""
                    CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE anchor_clouds (
                        task_index INTEGER PRIMARY KEY, anchor_index INTEGER UNIQUE NOT NULL,
                        caption_id INTEGER UNIQUE NOT NULL,
                        mean BLOB NOT NULL, cloud BLOB NOT NULL, n_tokens INTEGER NOT NULL
                    );
                """)
                connection.executemany("INSERT INTO anchor_clouds VALUES (?, ?, ?, ?, ?, ?)", (
                    (i, i, i, cloud.mean(axis=0).astype("<f4").tobytes(), cloud.tobytes(), len(cloud))
                    for i, cloud in enumerate(clouds)
                ))
            budget = TransportBudget(path, region)
            try:
                selected = budget.task_nearest(query)
                expected = np.argsort([task_wasserstein1(query, cloud) for cloud in clouds])[:5]
                np.testing.assert_array_equal(selected, expected)
                score1, scoreinf = budget.score(vector, query)
                self.assertAlmostEqual(score1, score_order1(vector, anchors)[0])
                self.assertAlmostEqual(scoreinf, score_orderinf(vector, anchors[expected])[0])
                self.assertAlmostEqual(order1_score(np.linalg.norm(anchors - vector, axis=1)), score1)
            finally:
                budget.connection.close()

    def test_batched_order1_matches_direct_research_score(self) -> None:
        rng = np.random.default_rng(3)
        anchors = rng.normal(size=(17, 8)).astype(np.float32)
        queries = rng.normal(size=(9, 8)).astype(np.float32)
        np.testing.assert_allclose(
            _order1_scores(queries, anchors, batch_size=3),
            score_order1(queries, anchors), rtol=1e-6, atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
