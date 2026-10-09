"""CPU-only invariants for the safe-flow experiment; no weights required."""
import unittest

import pandas as pd

from backend.flow_experiment import canonical, nearest, split_corpus


class SplitTests(unittest.TestCase):
    def corpus(self):
        rows = []
        for split, count in (("safe_reference", 12), ("safe_calibration", 8),
                             ("safe_test", 5), ("toxic_test_i2p", 5)):
            safe = split.startswith("safe")
            for j in range(count):
                rows.append({
                    "prompt_key": f"{split}-{j}", "prompt": f"An image of {split} {j}",
                    "split": split, "label": "safe" if safe else "risky",
                    "source": "diffusiondb" if safe else "i2p",
                    "eligible_for_region": split in ("safe_reference", "safe_calibration"),
                    "evaluation_only": split not in ("safe_reference", "safe_calibration"),
                })
        return pd.DataFrame(rows)

    def test_safe_only_disjoint_partitions(self):
        result, duplicates = split_corpus(self.corpus(), 4, 42)
        self.assertEqual(duplicates, 0)
        self.assertEqual(len(result["safe_probe_source"]), 4)
        self.assertEqual(len(result["safe_reference"]), 8)
        self.assertEqual(len(result["safe_validation"]), 8)
        self.assertEqual(len(result["safe_test"]), 5)
        self.assertEqual(len(result["unsafe_test"]), 5)
        seen = set()
        for split, rows in result.items():
            for row in rows:
                self.assertNotIn(row["id"], seen)
                seen.add(row["id"])
                self.assertEqual(row["label"] == "risky", split == "unsafe_test")

    def test_safe_splits_deterministic(self):
        a, _ = split_corpus(self.corpus(), 4, 42)
        b, _ = split_corpus(self.corpus().sample(frac=1, random_state=11), 4, 42)
        self.assertEqual(a, b)

    def test_risky_in_safe_split_rejected(self):
        df = self.corpus()
        df.loc[0, "label"] = "risky"
        with self.assertRaises(ValueError):
            split_corpus(df, 4, 42)

    def test_canonical_duplicates_dropped_before_fit(self):
        df = self.corpus()
        df.loc[df["split"] == "toxic_test_i2p", "prompt"] = df.loc[
            df["split"] == "safe_reference", "prompt"
        ].iloc[0]
        result, dropped = split_corpus(df, 4, 42)
        self.assertEqual(dropped, 5)
        self.assertEqual(len(result["unsafe_test"]), 0)

    def test_canonical_whitespace_punctuation(self):
        self.assertEqual(canonical("A-cat!"), canonical("a cat"))

    def test_nearest_weighted_euclidean(self):
        import torch
        q = torch.tensor([1.0, 2.0])
        r = torch.tensor([[0.0, 0.0], [1.0, 5.0]])
        self.assertAlmostEqual(nearest(q, r), (5.0 ** 0.5))


if __name__ == "__main__":
    unittest.main()
