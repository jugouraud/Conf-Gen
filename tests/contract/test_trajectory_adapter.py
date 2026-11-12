"""Actual Torch weight loading, freezing and model boundary behavior."""

import numpy as np
import pytest
import torch

from backend.src.common.storage import calculate_file_sha256
from backend.src.control.contracts import Context
from backend.src.control.model import TorchTrajectoryAdapter

pytestmark = pytest.mark.contract


class MotionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    def encode(self, obs):
        return obs

    def drift(self, x, t, c, a):
        return self.weight * x

    def baseline(self, x, t):
        return x * 0

    def decode(self, x):
        return x

    def noise(self, x, t):
        return torch.zeros_like(x)

    def initialize(self, c, rng, warm):
        return torch.tensor(rng.standard_normal((c.horizon + 1, 2)))


def test_pinned_weights_load_frozen_and_all_model_methods_obey_contract(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save(MotionModule().state_dict(), path)
    digest = calculate_file_sha256(path)
    adapter = TorchTrajectoryAdapter.from_checkpoint(MotionModule(), path, digest)
    c = Context("scene", 0, (0, 0), (1, 1), (), 0.1, 2)
    assert adapter.encode(c) is c
    assert not adapter.module.training and not adapter.module.weight.requires_grad
    x = adapter.initialize(c, np.random.default_rng(2), None)
    np.testing.assert_array_equal(adapter.drift(x, 0.5, c, "nominal"), 2 * x)
    np.testing.assert_array_equal(adapter.baseline(x, 0.5), 0 * x)
    np.testing.assert_array_equal(adapter.noise(x, 0.5), 0 * x)
    np.testing.assert_array_equal(adapter.decode(x), x)
    with pytest.raises(ValueError, match="hash"):
        TorchTrajectoryAdapter.from_checkpoint(MotionModule(), path, "0" * 64)
    with torch.no_grad():
        adapter.module.weight.add_(1)
    with pytest.raises(ValueError, match="changed"):
        adapter.drift(x, 0.5, c, "nominal")


def test_training_mode_after_freeze_is_rejected():
    adapter = TorchTrajectoryAdapter(MotionModule(), "test-revision")
    adapter.module.train()
    with pytest.raises(ValueError, match="changed"):
        adapter.assert_frozen()
