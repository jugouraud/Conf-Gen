"""One-class corpus and region guardrail tests."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.embeddings.prompt_region import load_corpus
from scripts.build_prompt_dataset import build_safe, prompt_key


class PromptCorpusTests(unittest.TestCase):
    def test_safe_filter_and_split_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            pd.DataFrame([
                {"prompt": "a calm blue landscape", "prompt_nsfw": 0.001, "image_nsfw": 0.01},
                {"prompt": "a calm blue landscape", "prompt_nsfw": 0.009, "image_nsfw": 0.04},
                {"prompt": "a red forest at dawn", "prompt_nsfw": 0.001, "image_nsfw": 0.01},
                {"prompt": "a castle in clouds", "prompt_nsfw": 0.001, "image_nsfw": 0.01},
                {"prompt": "an unsafe sample here", "prompt_nsfw": 0.2, "image_nsfw": 0.01},
                {"prompt": "short", "prompt_nsfw": 0.001, "image_nsfw": 0.01},
            ]).to_parquet(source)
            first = build_safe(source, size=3)
            second = build_safe(source, size=3)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(first["prompt_key"].nunique(), 3)
            self.assertEqual(set(first["split"]), {"safe_reference", "safe_calibration"})
            self.assertTrue(first.loc[first["eligible_for_region"], "split"].isin(
                {"safe_reference", "safe_calibration"}
            ).all())
            self.assertEqual(prompt_key("A calm blue landscape"), prompt_key("a calm blue landscape"))

    def test_fitter_rejects_risky_eligibility_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.parquet"
            base = pd.DataFrame([
                {"prompt_key": "safe", "prompt": "safe prompt", "split": "safe_reference",
                 "label": "safe", "eligible_for_region": True, "evaluation_only": False},
                {"prompt_key": "risky", "prompt": "risky prompt", "split": "toxic_test_i2p",
                 "label": "risky", "eligible_for_region": True, "evaluation_only": True},
            ])
            base.to_parquet(path)
            with self.assertRaisesRegex(ValueError, "eligibility"):
                load_corpus(path)
            base.loc[1, "eligible_for_region"] = False
            base.loc[1, "prompt_key"] = "safe"
            base.to_parquet(path)
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_corpus(path)


if __name__ == "__main__":
    unittest.main()
