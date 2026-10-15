"""Prompt encodings survive repeated checks and process-like validator reloads."""

import tempfile
import unittest
from pathlib import Path
from threading import Lock
from unittest.mock import Mock

import numpy as np

from backend.validation.prompt_cache import PromptEncodingStore
from frontend.coco_region_validation import LiveCocoRegionValidator


class PromptCacheTests(unittest.TestCase):
    def test_cache_survives_new_validator_and_encodes_only_missing_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.sqlite3"

            def validator() -> LiveCocoRegionValidator:
                instance = object.__new__(LiveCocoRegionValidator)
                instance.region = Mock(mean=np.zeros(2, dtype=np.float32))
                instance.encoder_key = "clip-test|eos-cloud-v1"
                instance.encoding_store = PromptEncodingStore(path)
                instance._model_lock = Lock()
                return instance

            first = validator()
            first._encode_uncached = Mock(side_effect=lambda prompts: (
                np.asarray([[float(len(prompt)), 2.0] for prompt in prompts], dtype=np.float32),
                [np.asarray([[float(len(prompt)), 3.0]], dtype=np.float32) for prompt in prompts],
            ))
            vectors, clouds = first.encode_prompts_with_clouds(["one", "two", "one"], batch_size=1)
            self.assertEqual(first._encode_uncached.call_count, 2)
            np.testing.assert_array_equal(vectors[0], vectors[2])
            self.assertEqual(len(clouds), 3)

            second = validator()
            second._encode_uncached = Mock(side_effect=AssertionError("CLIP should not run"))
            restored, restored_clouds = second.encode_prompts_with_clouds(["two", "one", "two"])
            np.testing.assert_array_equal(restored, vectors[[1, 0, 1]])
            np.testing.assert_array_equal(restored_clouds[0], clouds[1])
            second._encode_uncached.assert_not_called()

            padded, _ = second.encode_prompts_with_clouds(["  one  "])
            np.testing.assert_array_equal(padded[0], vectors[0])
            second._encode_uncached.assert_not_called()

    def test_encoder_key_and_dimension_separate_incompatible_encodings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PromptEncodingStore(Path(directory) / "validation.sqlite3")
            store.write_many("model-a", {"prompt": (
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([[3.0, 4.0]], dtype=np.float32),
            )}, 2)
            self.assertEqual(store.read_many("model-a", ["prompt"], 2)["prompt"][0][0], 1.0)
            self.assertEqual(store.read_many("model-b", ["prompt"], 2), {})
            self.assertEqual(store.read_many("model-a", ["prompt"], 3), {})


if __name__ == "__main__":
    unittest.main()
