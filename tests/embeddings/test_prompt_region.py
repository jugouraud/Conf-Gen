"""One-class corpus and region guardrail tests."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from backend.embeddings.compare_representations import benign_variants, select_candidate
from backend.embeddings.prompt_region import EosEncoder, load_corpus
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


class RepresentationTests(unittest.TestCase):
    def test_pooling_uses_attention_mask_and_layer_eos(self):
        class Tokenizer:
            def __call__(self, prompts, **_kwargs):
                return {"input_ids": torch.tensor([[1, 2, 9, 9], [1, 9, 9, 9]]),
                        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])}

        class Model:
            def __call__(self, **_kwargs):
                final = torch.zeros((2, 4, 768))
                final[:, 0] = 2
                final[:, 1] = 4
                final[:, 2] = 6
                final[:, 3] = 1000
                early = torch.full_like(final, 7)
                early[:, 1] = 11
                early[:, 2] = 13
                return SimpleNamespace(last_hidden_state=final,
                                       hidden_states=(None, None, None, None, early))

        with tempfile.TemporaryDirectory() as directory:
            encoder = EosEncoder(Path(directory) / "cache.sqlite3", model_id="test", device="cpu")
            encoder.torch = torch
            encoder.device = torch.device("cpu")
            encoder.tokenizer = Tokenizer()
            encoder.model = Model()
            try:
                encoder.representation = "mean_final"
                mean = encoder._encode(["one", "two"])
                np.testing.assert_allclose(mean[:, 0], [4, 3])
                encoder.representation = "eos_layer_4"
                layer = encoder._encode(["one", "two"])
                np.testing.assert_allclose(layer[:, 0], [13, 11])
            finally:
                encoder.close()

    def test_cache_separates_representations(self):
        frame = pd.DataFrame({"prompt_key": ["key"], "prompt": ["safe text"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            for representation, value in (("eos_final", 1), ("mean_final", 2)):
                encoder = EosEncoder(path, model_id="test", device="cpu",
                                     representation=representation)
                with patch.object(encoder, "_encode", return_value=np.full((1, 768), value, dtype=np.float32)):
                    self.assertEqual(encoder.vectors(frame)[0, 0], value)
                encoder.close()
            encoder = EosEncoder(path, model_id="test", device="cpu", representation="mean_final")
            with patch.object(encoder, "_encode", side_effect=AssertionError("cache miss")):
                self.assertEqual(encoder.vectors(frame)[0, 0], 2)
            encoder.close()

    def test_safe_selection_ignores_toxic_results(self):
        frame = pd.DataFrame({"prompt": ["a calm castle, blue sky"], "prompt_key": ["safe"]})
        self.assertEqual(set(benign_variants(frame)["change"]),
                         {"style", "quality", "clause_order"})
        metrics = {
            "eos_final": {"safe_coverage": 0.95, "median_normalized_score_shift": 0.2,
                          "absolute_length_correlation": 0.2, "radius": 30, "toxic_rejection": 0.99},
            "mean_final": {"safe_coverage": 0.95, "median_normalized_score_shift": 0.1,
                           "absolute_length_correlation": 0.1, "radius": 20, "toxic_rejection": 0.01},
        }
        self.assertEqual(select_candidate(metrics, 0.95), "mean_final")


if __name__ == "__main__":
    unittest.main()
