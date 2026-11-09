import pytest
import torch

from backend.src.repair.steering import (
    RepairedVelocityField,
    build_constant_steering_schedule,
)

pytestmark = pytest.mark.unit


class ConstantConditionAdapter:
    """An analytic external model double whose condition is its velocity."""

    device = "cpu"

    def evaluate_guided_velocity(self, states, time, embeddings):
        return torch.full_like(states, embeddings)


@pytest.mark.parametrize("strength,expected", [(0, 8), (0.25, 6), (1, 0)])
def test_repair_blends_original_and_approved_velocity(strength, expected):
    field = RepairedVelocityField(
        ConstantConditionAdapter(), 8, 0, build_constant_steering_schedule(strength)
    )
    actual = field.evaluate_guided_velocity(torch.zeros(1, 1, 1, 1), 0.5)
    torch.testing.assert_close(actual, torch.full((1, 1, 1, 1), float(expected)))


def test_nonzero_repair_requires_an_approved_condition():
    field = RepairedVelocityField(
        ConstantConditionAdapter(), 8, None, build_constant_steering_schedule(0.1)
    )
    with pytest.raises(ValueError, match="approved safe condition"):
        field.evaluate_guided_velocity(torch.zeros(1, 1, 1, 1), 0.5)
