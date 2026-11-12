"""Endpoint anchoring, structured noise and fixed-track decoding regressions."""

import numpy as np
import pytest
import torch

from backend.src.control.contracts import Context
from backend.src.expert.track import TrackSpec, track_for
from backend.src.trajectory_flow.model import FlowArchitecture, TrajectoryFlowModel
from backend.src.trajectory_flow.sample import run_flow_sampling

pytestmark = pytest.mark.unit


def _scene(track, start_arc, goal_arc, horizon=16):
    circuit = track_for(track)
    start = circuit.point(start_arc)
    goal = circuit.point(goal_arc)
    return Context(
        "fixed-track", 0, tuple(start), tuple(goal), (), 0.4, horizon,
        features=(0.0, 0.0, -6.35, 6.35, -4.85, 4.85),
    )


def test_spline_flow_samples_are_smooth_and_keep_conditioned_endpoints():
    track = TrackSpec()
    architecture = FlowArchitecture(horizon=16, dt=0.4, width=32,
                                    depth=2, context_width=16, knots=6,
                                    track=vars(track))
    model = TrajectoryFlowModel(
        architecture, {"nominal": {"side": 0, "speed_scale": 1.0,
                                  "obstacle_index": 0}}
    )
    scene = _scene(track, 2.0, 9.0)
    baseline = model.codec._baseline(np.asarray(scene.state), np.asarray(scene.goal))
    model.set_normalization(baseline[None], [scene])
    start = model.initialize(scene, np.random.default_rng(7), None)
    assert start.shape == (4 + 2 * architecture.knots,)
    np.testing.assert_allclose(start[:4], [*scene.state, *scene.goal], atol=1e-6)
    assert torch.equal(model.drift(start, 0.5, scene, "nominal")[:4], torch.zeros(4))
    altered = start.clone()
    altered[:4] += 10.0
    torch.testing.assert_close(model.drift(start, 0.5, scene, "nominal"),
                               model.drift(altered, 0.5, scene, "nominal"))
    torch.testing.assert_close(model.baseline(start, 0.5), model.baseline(altered, 0.5))
    model.freeze("spline-test")
    predicted = run_flow_sampling(model, scene, "nominal", seed=7, steps=8)
    np.testing.assert_allclose(predicted.reference[0], scene.state, atol=1e-5)
    np.testing.assert_allclose(predicted.reference[-1], scene.goal, atol=1e-5)
    assert np.isfinite(predicted.reference).all()
    assert np.max(np.linalg.norm(np.diff(predicted.reference, axis=0), axis=1)) < 6


def test_spline_rejects_wrong_endpoints_and_supports_legacy_geometry():
    scene = _scene(TrackSpec(), 0.0, 7.0)
    model = TrajectoryFlowModel(
        FlowArchitecture(horizon=16, dt=0.4, knots=5),
        {"nominal": {"side": 0, "speed_scale": 1.0, "obstacle_index": 0}},
    )
    reference = np.linspace(scene.state, scene.goal, 17)
    model.set_normalization(reference[None], [scene])
    bad = reference.copy()
    bad[0] += 0.2
    with pytest.raises(ValueError, match="endpoints"):
        model.normalize(torch.tensor(bad[None]), [scene])
    encoded = model.normalize(torch.tensor(reference[None]), [scene])
    np.testing.assert_allclose(model.decode(encoded[0]).detach().numpy()[[0, -1]],
                               reference[[0, -1]], atol=1e-5)
