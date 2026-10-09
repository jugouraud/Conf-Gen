"""The T2I adaptive decision must search the whole safe reference split."""

import sqlite3
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backend.embeddings.prompt_transport import AdaptiveOrderInf, _anchor_clouds
from research.wasserstein import score_orderinf


class PromptTransportTests(unittest.TestCase):
    def test_adaptive_score_selects_from_every_safe_reference(self):
        reference = pd.DataFrame({"prompt": [str(i) for i in range(61)]})
        anchors = np.column_stack((np.arange(61, dtype=np.float32),
                                   np.zeros(61, dtype=np.float32)))
        region = SimpleNamespace(anchors=anchors)

        class Encoder:
            def clouds(self, prompts):
                return [np.pad(np.array([[float(prompt)]], dtype=np.float32),
                               ((0, 0), (0, 767))) for prompt in prompts]

        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE anchor_clouds (anchor_index INTEGER PRIMARY KEY, "
                               "cloud BLOB NOT NULL, n_tokens INTEGER NOT NULL)")
            indices, clouds = _anchor_clouds(connection, reference, region, Encoder())

        self.assertEqual(len(indices), len(reference))
        adaptive = AdaptiveOrderInf(region, indices, clouds)
        query_cloud = Encoder().clouds(["60"])[0]
        selected = adaptive.task_nearest(query_cloud)
        self.assertEqual(set(selected), {56, 57, 58, 59, 60})
        query = np.array([60.5, 0.0], dtype=np.float32)
        self.assertAlmostEqual(adaptive.score(query, query_cloud),
                               score_orderinf(query[None, :], anchors[selected], n_cuts=0)[0])


if __name__ == "__main__":
    unittest.main()
