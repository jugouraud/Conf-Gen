"""Scene selection, physical diagnostics, and metre-accurate SVG geometry."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.src.control.contracts import Context
from backend.src.expert.geometry import scene_boundary_clearances, timed_clearances
from backend.src.expert.scenes import Scene
from backend.src.expert.storage import MAX_EXPERT_TRAJECTORIES, read_record
from backend.src.expert.track import track_for
from backend.src.mpc.geometry import Obstacle
from backend.src.trajectory_flow.model import TrajectoryFlowModel
from backend.src.trajectory_flow.sample import run_flow_sampling

PALETTE = ("#38bdf8", "#f472b6", "#facc15", "#a78bfa", "#4ade80", "#fb923c")
MAX_REFERENCES = 12


@dataclass(frozen=True)
class SceneCatalog:
    scenes: tuple[Scene, ...]
    footprint_radius: float
    dataset_id: str
    split: str
    horizon: int
    dt: float


@dataclass(frozen=True)
class ReferenceOverlay:
    guidance: str
    seed: int
    positions: np.ndarray  # (H+1, 2), physical metres; not an MPC rollout
    obstacle_clearance: float | None  # signed, swept disc-surface metres
    boundary_clearance: float  # signed disc-surface metres
    latency_seconds: float


def read_scene_catalog(dataset: Path, split: str = "train") -> SceneCatalog:
    """Read ready *observations* from a bounded expert-v1 manifest, not labels."""
    if split not in {"train", "reference", "future_calibration", "test"}:
        raise ValueError("Unknown expert dataset split")
    manifest = read_record(dataset / "frozen.json")
    if manifest.get("schema") != "expert-dataset-v1":
        raise ValueError("Expected expert-dataset-v1 frozen manifest")
    vehicle = manifest["config"]["vehicle"]
    if vehicle["kind"] != "velocity_disc" or vehicle["output"] != "xy":
        raise ValueError("Visualizer only supports velocity_disc/xy scenes")
    rows = manifest["scenes"]
    if not isinstance(rows, list) or len(rows) > MAX_EXPERT_TRAJECTORIES:
        raise ValueError("Unexpected scene count")
    selected = [Scene.from_record(row["raw_observation"]) for row in rows
                if row["status"] == "ready" and row["split"] == split]
    if not selected:
        raise ValueError(f"No ready scenes in {split} split")
    radius = float(vehicle["footprint_radius"])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid vehicle footprint radius")
    if len({scene.scene_id for scene in selected}) != len(selected):
        raise ValueError("Duplicate scene ID")
    planner = manifest["config"]["planner"]
    return SceneCatalog(
        tuple(selected), radius, manifest["dataset_id"], split,
        planner["horizon"], planner["dt"],
    )


def context_from_scene(scene: Scene, horizon: int, dt: float) -> Context:
    """Match the training context: physical time k=0, predicted centres i=0..H."""
    if len(scene.initial_state) != 2:
        raise ValueError("Only velocity-disc xy initial states are supported")
    times = scene.start_time + dt * np.arange(horizon + 1)
    obstacles = tuple(
        Obstacle(tuple(map(tuple, obstacle.at(times))), obstacle.radius)
        for obstacle in scene.obstacles
    )
    return Context(
        episode=scene.group_id, k=0, state=scene.initial_state,
        goal=scene.goal, obstacles=obstacles, dt=dt, horizon=horizon,
        compatibility=f"velocity_disc-xy-H{horizon}-dt{dt:.8g}",
        features=(*scene.initial_velocity, *scene.bounds),
    )


def sample_references(
    model: TrajectoryFlowModel,
    scene: Scene,
    guidance_names: tuple[str, ...],
    *,
    seed: int,
    samples_per_guidance: int,
    steps: int,
    solver: str,
    footprint_radius: float,
) -> tuple[ReferenceOverlay, ...]:
    """Sample a frozen field; same seed/noise for each guidance at each index."""
    if (type(seed) is not int or not 0 <= seed < 2**32
            or type(samples_per_guidance) is not int or not 1 <= samples_per_guidance <= 3
            or not guidance_names or len(guidance_names) * samples_per_guidance > MAX_REFERENCES
            or type(steps) is not int or not 1 <= steps <= 128
            or solver not in {"euler", "heun"}):
        raise ValueError("Sampling options exceed visualizer limits")
    if any(name not in model.guidance for name in guidance_names):
        raise ValueError("Guidance missing from frozen checkpoint")
    context = context_from_scene(scene, model.architecture.horizon, model.architecture.dt)
    times = scene.start_time + model.architecture.dt * np.arange(model.architecture.horizon + 1)
    results = []
    for index in range(samples_per_guidance):
        current_seed = (seed + index) % 2**32
        for name in guidance_names:
            sample = run_flow_sampling(
                model, context, name, seed=current_seed, steps=steps, solver=solver
            )
            obstacles = timed_clearances(times, sample.reference, scene.obstacles, footprint_radius)
            boundaries = scene_boundary_clearances(scene, sample.reference, footprint_radius)
            results.append(ReferenceOverlay(
                name, current_seed, sample.reference,
                float(obstacles.min()) if len(obstacles) else None,
                float(boundaries.min()), sample.latency_seconds,
            ))
    return tuple(results)


def render_scene_svg(
    scene: Scene,
    references: tuple[ReferenceOverlay, ...],
    *,
    footprint_radius: float,
    index: int,
    horizon: int,
    dt: float,
) -> str:
    """Draw the physical x/y plane, including moving circular obstacles at i*dt."""
    if not 0 <= index <= horizon:
        raise ValueError("Physical prediction index must lie in 0..H")
    xmin, xmax, ymin, ymax = scene.bounds
    positions = [r.positions for r in references]
    if any(p.shape != (horizon + 1, 2) or not np.isfinite(p).all() for p in positions):
        raise ValueError("Invalid sampled reference shape or coordinate")
    xvalues = [xmin, xmax, scene.initial_state[0], scene.goal[0]]
    yvalues = [ymin, ymax, scene.initial_state[1], scene.goal[1]]
    for p in positions:
        xvalues.extend(p[:, 0])
        yvalues.extend(p[:, 1])
    for obstacle in scene.obstacles:
        xvalues.extend(obstacle.centers[:, 0] + obstacle.radius)
        xvalues.extend(obstacle.centers[:, 0] - obstacle.radius)
        yvalues.extend(obstacle.centers[:, 1] + obstacle.radius)
        yvalues.extend(obstacle.centers[:, 1] - obstacle.radius)
    circuit = track_for(scene.track) if scene.track is not None else None
    if circuit is not None:
        center = circuit.points[:-1]
        normal = circuit.normal(circuit.arc[:-1])
        outer = center - circuit.spec.right_width * normal
        inner = center + circuit.spec.left_width * normal
        xvalues.extend(outer[:, 0])
        xvalues.extend(inner[:, 0])
        yvalues.extend(outer[:, 1])
        yvalues.extend(inner[:, 1])
    pad = max(0.35, footprint_radius * 2)
    left, right = min(xvalues) - pad, max(xvalues) + pad
    bottom, top = min(yvalues) - pad, max(yvalues) + pad

    def point(x: float, y: float) -> str:
        return f"{x:.5f},{-y:.5f}"

    def polyline(points: np.ndarray) -> str:
        return " ".join(point(x, y) for x, y in points)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{left:.5f} {-top:.5f} '
        f'{right - left:.5f} {top - bottom:.5f}" '
        'style="display:block;width:100%;max-height:72vh;background:#0b1220;border-radius:12px">',
        f'<rect x="{xmin:.5f}" y="{-ymax:.5f}" width="{xmax-xmin:.5f}" '
        f'height="{ymax-ymin:.5f}" fill="#16253c" stroke="#94a3b8" stroke-width="0.04"/>',
    ]
    if circuit is not None:
        def closed_path(path: np.ndarray) -> str:
            return "M" + polyline(path).replace(" ", " L") + " Z"

        parts.extend((
            f'<path data-kind="track-asphalt" d="{closed_path(outer)} '
            f'{closed_path(inner[::-1])}" fill="#334155" fill-rule="evenodd"/>',
            f'<path data-kind="track-outer" d="{closed_path(outer)}" '
            'fill="none" stroke="#94a3b8" stroke-width="0.05"/>',
            f'<path data-kind="track-inner" d="{closed_path(inner)}" '
            'fill="none" stroke="#94a3b8" stroke-width="0.05"/>',
            f'<path data-kind="track-centerline" d="{closed_path(center)}" '
            'fill="none" stroke="#d9bf7b" stroke-width="0.03" '
            'stroke-dasharray="0.2 0.15"/>',
        ))
    current_time = scene.start_time + index * dt
    for obstacle in scene.obstacles:
        track = obstacle.centers
        center = obstacle.at(np.asarray([current_time]))[0]
        parts.append(f'<polyline points="{polyline(track)}" fill="none" '
                     'stroke="#fb923c" stroke-opacity="0.65" stroke-width="0.025" '
                     'stroke-dasharray="0.10 0.08"/>')
        parts.append(f'<circle cx="{center[0]:.5f}" cy="{-center[1]:.5f}" '
                     f'r="{obstacle.radius + footprint_radius:.5f}" '
                     'fill="none" stroke="#fb923c" stroke-width="0.022" '
                     'stroke-dasharray="0.065 0.05"/>')
        parts.append(f'<circle cx="{center[0]:.5f}" cy="{-center[1]:.5f}" '
                     f'r="{obstacle.radius:.5f}" fill="#fb923c" fill-opacity="0.5" '
                     'stroke="#fdba74" stroke-width="0.025"/>')
    colors = {name: PALETTE[i % len(PALETTE)]
              for i, name in enumerate(dict.fromkeys(r.guidance for r in references))}
    for reference in references:
        color = colors[reference.guidance]
        x, y = reference.positions[index]
        parts.append(f'<polyline points="{polyline(reference.positions)}" fill="none" '
                     f'stroke="{color}" stroke-width="0.035" stroke-linejoin="round" '
                     'stroke-linecap="round" stroke-opacity="0.85"/>')
        parts.append(f'<circle cx="{x:.5f}" cy="{-y:.5f}" r="{footprint_radius:.5f}" '
                     f'fill="{color}" fill-opacity="0.18" stroke="{color}" '
                     'stroke-width="0.025"/>')
    for coordinates, color, label in (
        (scene.initial_state, "#4ade80", "START"),
        (scene.goal, "#f8fafc", "GOAL"),
    ):
        x, y = coordinates[:2]
        parts.append(f'<circle cx="{x:.5f}" cy="{-y:.5f}" r="0.09" '
                     f'fill="{color}" stroke="#0b1220" stroke-width="0.025"/>')
        parts.append(f'<text x="{x + 0.12:.5f}" y="{-y - 0.12:.5f}" fill="{color}" '
                     'font-family="sans-serif" font-size="0.18">'
                     f'{label}</text>')
    return "".join(parts) + "</svg>"
