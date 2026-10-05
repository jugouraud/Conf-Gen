import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from backend.validation import shieldgemma as scorer


class ImageScoringTests(unittest.TestCase):
    def test_single_policy_batches_preserve_yes_probabilities(self):
        transformers = ModuleType("transformers")
        transformers.AutoProcessor = Mock()
        transformers.ShieldGemma2ForImageClassification = Mock()
        processor = Mock()
        processor.return_value.to.return_value = {"pixel_values": torch.zeros(1)}
        model = Mock(dtype=torch.float16)
        model.side_effect = [
            SimpleNamespace(probabilities=torch.tensor([[p, 1 - p]])) for p in (0.1, 0.8, 0.3)
        ]
        opened = MagicMock()
        with patch.dict(sys.modules, {"transformers": transformers}), patch.object(
            scorer, "_validate_image_path", return_value=Path("image.jpg")
        ), patch.object(scorer, "get_local_model_directory"), patch.object(
            scorer, "_load_image_scorer", return_value=(processor, model)
        ), patch("PIL.Image.open", return_value=opened):
            scores = scorer.score_image("image.jpg", device="cpu")
        self.assertEqual(list(scores), ["sexual", "dangerous", "violence"])
        for actual, expected in zip(scores.values(), (0.1, 0.8, 0.3)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual([call.kwargs["policies"] for call in processor.call_args_list],
                         [["sexual"], ["dangerous"], ["violence"]])
        for call in model.call_args_list:
            self.assertEqual(call.kwargs["logits_to_keep"], 1)
            self.assertFalse(call.kwargs["use_cache"])

    def test_gpu_preserves_bfloat16_to_avoid_float16_overflow(self):
        processor_class, model_class = Mock(), Mock()
        scorer._load_image_scorer.cache_clear()
        try:
            scorer._load_image_scorer(processor_class, model_class, Path("model"), "cuda")
            self.assertEqual(model_class.from_pretrained.call_args.kwargs["torch_dtype"], torch.bfloat16)
        finally:
            scorer._load_image_scorer.cache_clear()


if __name__ == "__main__":
    unittest.main()
