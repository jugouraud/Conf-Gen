"""Obstacle-local diagnostic uses expert proximity and preserves empty semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

from backend.src.control.contracts import Context
from backend.src.mpc.geometry import Obstacle
from backend.src.trajectory_flow.data import FlowExample
from backend.src.trajectory_flow.evaluate import run_flow_evaluation
from backend.src.trajectory_flow.sample import FlowSample

pytestmark = pytest.mark.unit


def test_obstacle_local_diagnostics_are_not_hidden_by_global_ade(monkeypatch):
    import backend.src.trajectory_flow.evaluate as evaluation

    target = np.array([[0., 0.], [1., 0.4], [2., 0.]])
    predicted = np.array([[0., 0.], [1., 0.1], [2., 0.]])
    obstacle = Obstacle(((1., 0.), (1., 0.), (1., 0.)), 0.1)
    context = Context("episode-1", 0, (0., 0.), (2., 0.),
                      (obstacle,), 0.4, 2, features=(0., 0., -3., 3., -3., 3.))
    example = FlowExample(context, "nominal", target, "episode-1")
    prepared = SimpleNamespace(validation=(example,))
    model = SimpleNamespace(codec=SimpleNamespace(track=None))
    monkeypatch.setattr(evaluation, "run_flow_sampling", lambda *a, **kw:
                        FlowSample(predicted, kw["seed"], "heun", 32, 0.01))

    result = run_flow_evaluation(model, prepared, seed=3, steps=32,
                                 solver="heun", footprint_radius=0.15)
    assert result["ade_m"] == pytest.approx(0.1)
    assert result["obstacle_local_ade_m"] == pytest.approx(0.3)
    assert result["obstacle_local_max_error_m"] == pytest.approx(0.3)
    assert result["obstacle_local_points"] == 1
    assert result["collision_fraction"] == 1.0
    assert result["p05_surface_clearance_m"] < 0


def test_no_obstacle_has_no_misleading_local_error(monkeypatch):
    import backend.src.trajectory_flow.evaluate as evaluation

    path = np.array([[0., 0.], [1., 0.], [2., 0.]])
    context = Context("empty", 0, (0., 0.), (2., 0.), (), 0.4, 2,
                      features=(0., 0., -3., 3., -3., 3.))
    example = FlowExample(context, "nominal", path, "empty")
    monkeypatch.setattr(evaluation, "run_flow_sampling", lambda *a, **kw:
                        FlowSample(path, kw["seed"], "heun", 32, 0.01))
    result = run_flow_evaluation(
        SimpleNamespace(codec=SimpleNamespace(track=None)),
        SimpleNamespace(validation=(example,)), seed=3, steps=32,
        solver="heun", footprint_radius=0.15,
    )
    assert result["obstacle_local_ade_m"] is None
    assert result["obstacle_local_points"] == 0
    assert result["p05_surface_clearance_m"] is None
