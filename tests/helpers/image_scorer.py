"""Test doubles for the expensive image inference boundary only."""

from PIL import Image


class PixelPolicyScorer:
    """Recognize fixture colors through real image decoding; expose fixed policy scores."""

    metadata = {
        "model_id": "test/pixel-policy",
        "revision": "fixture-revision-1",
        "policies": ["sexual", "dangerous", "violence"],
    }

    def score_image(self, path):
        with Image.open(path) as image:
            red, _, blue = image.convert("RGB").getpixel((0, 0))
        return {"sexual": 0.0, "dangerous": 0.1, "violence": 0.8 if red > blue else 0.2}


class UnavailablePolicyScorer(PixelPolicyScorer):
    def score_image(self, path):
        raise TimeoutError("inference unavailable")


class CachedOnlyPolicyScorer(PixelPolicyScorer):
    def score_image(self, path):
        raise AssertionError("A verified cached score must not require inference")
