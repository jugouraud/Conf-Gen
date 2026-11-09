import pytest
import torch

from backend.src.repair.steering import (
    RepairedVelocityField,
    build_constant_steering_schedule,
    sample_repaired_trajectory,
)

pytestmark = pytest.mark.unit


class ConstantVelocityAdapter:
    """A flow with a closed-form displacement over [0,1]."""

    device = "cpu"
    shape = (1, 1, 1)

    def evaluate_guided_velocity(self, states, time, embeddings):
        return torch.full_like(states, embeddings)


def test_paired_sampling_uses_identical_seeded_noise_for_original_and_repair():
    adapter = ConstantVelocityAdapter()
    original = RepairedVelocityField(adapter, 4, 0, build_constant_steering_schedule(0))
    repaired = RepairedVelocityField(
        adapter, 4, 0, build_constant_steering_schedule(0.5)
    )

    before, _ = sample_repaired_trajectory(original, seed=17, steps=4)
    after, _ = sample_repaired_trajectory(repaired, seed=17, steps=4)

    # Constant velocity 4 versus 2 for one unit of time: displacement differs by 2.
    torch.testing.assert_close(before - after, torch.full_like(before, 2))


def test_sampling_replays_the_same_latents_for_a_fixed_seed():
    adapter = ConstantVelocityAdapter()
    field = RepairedVelocityField(adapter, 4, 0, build_constant_steering_schedule(0))

    first, _ = sample_repaired_trajectory(field, seed=17, steps=4)
    second, _ = sample_repaired_trajectory(field, seed=17, steps=4)

    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_sampling_rejects_nonfinite_trajectories():
    adapter = ConstantVelocityAdapter()
    field = RepairedVelocityField(
        adapter, float("inf"), 0, build_constant_steering_schedule(0)
    )

    with pytest.raises(ValueError, match="Non-finite repaired trajectory"):
        sample_repaired_trajectory(field, seed=17, steps=4)
