"""Behavioral checks for trainable field, scene geometry and flow coupling."""

import numpy as np
import pytest
import torch

from backend.src.control.contracts import Context
from backend.src.mpc.geometry import Obstacle
from backend.src.trajectory_flow.data import FlowExample, split_training_groups
from backend.src.trajectory_flow.model import FlowArchitecture, TrajectoryFlowModel
from backend.src.trajectory_flow.sample import run_flow_sampling

pytestmark = pytest.mark.unit


@pytest.fixture
def flow_model():
    torch.manual_seed(7)
    return TrajectoryFlowModel(
        FlowArchitecture(horizon=4, dt=0.2, width=32, depth=2, context_width=16),
        {
            "left": {"side": 1, "speed_scale": 1.0, "obstacle_index": 0},
            "right": {"side": -1, "speed_scale": 1.0, "obstacle_index": 0},
            "free": {"side": 0, "speed_scale": 1.0, "obstacle_index": 0},
        },
    )


def context(group="episode-a", obstacles=()):
    return Context(
        group, 0, (0.0, 0.0), (2.0, 0.0), obstacles, 0.2, 4,
        features=(0.0, 0.0, -1.0, 3.0, -1.0, 1.0),
    )


def moving_obstacle(offset: float):
    return Obstacle(tuple((offset, 0.3 + 0.03 * i) for i in range(5)), 0.1)


def test_obstacle_permutation_and_padding_invariance(flow_model):
    a, b = moving_obstacle(0.4), moving_obstacle(1.4)
    one = context(obstacles=(a, b))
    two = context(obstacles=(b, a))
    x = torch.randn(10)
    with torch.no_grad():
        first = flow_model.drift(x, 0.25, one, "free")
        second = flow_model.drift(x, 0.25, two, "free")
        empty = flow_model.drift(x, 0.25, context(), "free")
    assert torch.allclose(first, second, atol=1e-6)
    assert empty.shape == x.shape
    assert torch.isfinite(empty).all()


def test_flow_interpolation_target_and_parameter_update(flow_model):
    scene = context(obstacles=(moving_obstacle(1.0),))
    y = np.stack((np.linspace(0, 2, 5), np.zeros(5)), axis=-1)
    row = FlowExample(scene, "free", y, scene.episode)
    flow_model.set_normalization(y[None], [scene])
    generator = torch.Generator().manual_seed(3)
    x0 = torch.randn((1, 10), generator=generator)
    t = torch.rand((1, 1), generator=generator)
    x1 = flow_model.normalize(torch.as_tensor(y, dtype=torch.float32))[None]
    xt, target_velocity = (1 - t) * x0 + t * x1, x1 - x0
    assert torch.allclose(xt - x0, t * target_velocity)
    assert torch.allclose(flow_model.decode(x1[0]), torch.as_tensor(y, dtype=torch.float32), atol=1e-6)

    optimizer = torch.optim.AdamW(flow_model.parameters(), lr=0.002)
    weights_before = flow_model.flow_net[-1].weight.detach().clone()
    initial = None
    for index in range(70):
        optimizer.zero_grad()
        velocity = flow_model.drift_batch(xt, t, [scene], ["free"])
        loss = torch.mean((velocity - target_velocity).square())
        if index == 0:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < initial * 0.5
    assert not torch.equal(flow_model.flow_net[-1].weight, weights_before)


def test_classifier_free_unconditional_baseline_is_learned(flow_model):
    scene = context(obstacles=(moving_obstacle(0.8),))
    y = np.stack((np.linspace(0, 2, 5), np.zeros(5)), axis=-1)
    flow_model.set_normalization(y[None], [scene])
    flow_model.zero_grad(set_to_none=True)
    x = torch.randn((1, 10))
    t = torch.full((1, 1), 0.3)
    loss = flow_model.drift_batch(x, t, [scene], ["free"], drop_mask=torch.ones(1, dtype=torch.bool)).square().mean()
    loss.backward()
    assert flow_model.null_scene.grad.abs().sum() > 0
    assert flow_model.null_guidance.grad.abs().sum() > 0
    x = torch.randn(10)
    assert flow_model.baseline(x, 0.3).shape == x.shape
    assert not torch.allclose(flow_model.baseline(x, 0.3), flow_model.drift(x, 0.3, scene, "free"))


def test_invalid_guidance_and_group_separation(flow_model):
    scene = context(obstacles=(moving_obstacle(1.0),))
    with pytest.raises(ValueError, match="Unknown guidance"):
        flow_model.drift(torch.randn(10), 0.3, scene, "fabricated")
    with pytest.raises(ValueError, match="absent"):
        flow_model.drift(torch.randn(10), 0.3, context(), "left")
    rows = [FlowExample(context(f"group-{i}"), "free", np.zeros((5, 2)), f"group-{i}")
            for i in range(6) for _ in range(2)]
    fit, val = split_training_groups(rows, 7, 0.2)
    assert {r.group_id for r in fit}.isdisjoint({r.group_id for r in val})
    assert len(fit) + len(val) == 12


def test_frozen_paired_heun_mixing_changes_field_not_stored_paths(flow_model):
    scene = context(obstacles=(moving_obstacle(1.0),))
    y = np.stack((np.linspace(0, 2, 5), np.zeros(5)), axis=-1)
    flow_model.set_normalization(y[None], [scene])
    flow_model.freeze("test-model-v1")
    first = run_flow_sampling(flow_model, scene, "left", seed=11, steps=6, solver="heun")
    second = run_flow_sampling(flow_model, scene, "left", seed=11, steps=6, solver="heun")
    repaired = run_flow_sampling(
        flow_model, scene, "left", seed=11, steps=6, solver="heun",
        anchor="right", steering=lambda t: 1.0,
    )
    assert np.array_equal(first.reference, second.reference)
    assert first.reference.shape == (5, 2)
    assert np.isfinite(repaired.reference).all()
    # An anchor at unit strength is equivalent to sampling that field from the same x0.
    anchor = run_flow_sampling(flow_model, scene, "right", seed=11, steps=6, solver="heun")
    assert np.allclose(repaired.reference, anchor.reference, atol=1e-7)
    assert flow_model.revision == "test-model-v1"
