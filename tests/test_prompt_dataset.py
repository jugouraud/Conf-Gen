"""Check deterministic safe-prompt selection from local parquet metadata."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_prompt_dataset import build_safe


class PromptDatasetTests(unittest.TestCase):
    def test_safe_sample_is_independent_of_metadata_order(self):
        records = pd.DataFrame({
            "prompt": ["quiet blue lake", "green forest path", "warm city light",
                       "small red boat", "bright snow field", "soft autumn rain",
                       "Quiet Blue Lake"],
            "prompt_nsfw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005],
            "image_nsfw": [0.0] * 7,
        })
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.parquet"
            reversed_path = Path(directory) / "reversed.parquet"
            records.to_parquet(first, index=False)
            records.iloc[::-1].to_parquet(reversed_path, index=False)
            sample = build_safe(first, size=6)
            reversed_sample = build_safe(reversed_path, size=6)
        pd.testing.assert_frame_equal(sample, reversed_sample)
        self.assertEqual(len(sample), 6)
        self.assertEqual(set(sample["split"]),
                         {"safe_reference", "safe_calibration", "safe_test"})


if __name__ == "__main__":
    unittest.main()
