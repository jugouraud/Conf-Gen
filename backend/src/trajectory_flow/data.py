"""Read expert-v1 training targets and hold out entire episode groups."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from backend.src.control.contracts import Context
from backend.src.expert.audit import read_jsonl
from backend.src.expert.scenes import Scene
from backend.src.expert.storage import MAX_EXPERT_TRAJECTORIES, read_record
from backend.src.mpc.geometry import Obstacle


@dataclass(frozen=True)
class FlowExample:
    context: Context
    guidance: str
    target: np.ndarray  # (H+1,2), physical metres
    group_id: str
    track: dict | None = None  # Fixed track, or None for legacy rectangular scenes.


@dataclass(frozen=True)
class PreparedTraining:
    fit: tuple[FlowExample, ...]
    validation: tuple[FlowExample, ...]
    guidance: dict[str, dict]
    dataset_id: str
    manifest_sha256: str
    fit_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    horizon: int
    dt: float
    footprint_radius: float


def example_from_target(row: dict, horizon: int, dt: float, dataset_id: str) -> FlowExample:
    if row.get("schema") != "expert-target-v1" or row.get("split") != "train" or (
        row.get("dataset_id") != dataset_id or row.get("usage") != "expert_training_target"
    ) or row.get("validation", {}).get("valid") is not True:
        raise ValueError("Not a validated training target in the selected dataset")
    if row.get("vehicle", {}).get("kind") != "velocity_disc" or row["vehicle"]["output"] != "xy":
        raise ValueError("Only velocity-disc xy expert targets are supported")
    if row["planner"]["horizon"] != horizon or not np.isclose(row["planner"]["dt"], dt):
        raise ValueError("Target horizon/dt is incompatible with trained architecture")
    scene = Scene.from_record(row["observation"])
    if scene.group_id != row["group_id"] or scene.scene_id != row["scene_id"]:
        raise ValueError("Scene identity differs from expert target")
    times = scene.start_time + dt * np.arange(horizon + 1)
    if not np.allclose(np.asarray(row["timestamps"]), times, atol=1e-9, rtol=0):
        raise ValueError("Expert target physical timestamps differ from scene")
    target = np.asarray(row["y_ref_star"], dtype=np.float64)
    if target.shape != (horizon + 1, 2) or not np.isfinite(target).all() or (
        np.linalg.norm(target[0] - scene.initial_state[:2]) > 1e-5
    ):
        raise ValueError("Invalid expert reference dimensions, start or values")
    obstacles = tuple(
        Obstacle(tuple(map(tuple, obstacle.at(times))), obstacle.radius)
        for obstacle in scene.obstacles
    )
    context = Context(
        episode=scene.group_id,
        k=0,
        state=tuple(scene.initial_state),
        goal=scene.goal,
        obstacles=obstacles,
        dt=dt,
        horizon=horizon,
        compatibility=f"velocity_disc-xy-H{horizon}-dt{dt:.8g}",
        features=(*scene.initial_velocity, *scene.bounds),
    )
    # Retain each guidance label, but ensure its meaning is geometric.
    guidance = row["guidance"]
    if set(guidance) != {"name", "side", "speed_scale", "obstacle_index"}:
        raise ValueError("Invalid geometric guidance schema")
    return FlowExample(context, guidance["name"], target, scene.group_id,
                       None if scene.track is None else asdict(scene.track))


def split_training_groups(
    examples: list[FlowExample], seed: int, fraction: float
) -> tuple[tuple[FlowExample, ...], tuple[FlowExample, ...]]:
    """Split only the expert train partition, never reserved reference/test groups."""
    if not 0 < fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    groups = sorted({row.group_id for row in examples})
    if len(groups) < 2:
        raise ValueError("At least two independent training groups are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    count = max(1, min(len(groups) - 1, round(fraction * len(groups))))
    validation_ids = set(groups[:count])
    fit = tuple(row for row in examples if row.group_id not in validation_ids)
    validation = tuple(row for row in examples if row.group_id in validation_ids)
    if not fit or not validation or {r.group_id for r in fit} & {
        r.group_id for r in validation
    }:
        raise ValueError("Invalid independent episode separation")
    return fit, validation


def _matched_guidance_rows(rows: list[dict], guidance: list[dict]) -> list[dict]:
    """Use only scenes with all three validated labels, never fill missing targets."""
    required = {"nominal": 0, "left": 1, "right": -1}
    catalogue = {item["name"]: item for item in guidance}
    if set(catalogue) != set(required) or any(
        catalogue[name]["side"] != side for name, side in required.items()
    ) or len({catalogue[name]["obstacle_index"] for name in required}) != 1 or (
        len({catalogue[name]["speed_scale"] for name in required}) != 1
    ):
        raise ValueError("Matched training requires equal-speed nominal=0, left=+1, right=-1")
    scene_labels: dict[str, set[str]] = defaultdict(set)
    observations = {}
    for row in rows:
        scene_id, label = row["scene_id"], row["guidance"]["name"]
        if label not in catalogue or row["guidance"] != catalogue[label]:
            raise ValueError("Target guidance differs from frozen matched catalogue")
        if scene_id in observations and row["observation"] != observations[scene_id]:
            raise ValueError("Matched references have different scene observations")
        observations[scene_id] = row["observation"]
        if label in scene_labels[scene_id]:
            raise ValueError("Duplicate guidance label for one matched scene")
        scene_labels[scene_id].add(label)
    complete = {scene_id for scene_id, labels in scene_labels.items()
                if labels == set(required)}
    if not complete:
        raise ValueError("No complete validated nominal/left/right scenes")
    return [row for row in rows if row["scene_id"] in complete]


def run_training_preparation(
    dataset: Path, *, seed: int, validation_fraction: float,
    require_matched_triplets: bool = False,
) -> PreparedTraining:
    """Prepare *only* splits/train.jsonl for flow training.

    Does not use raw reference/future_calibration/test labels for fitting, model
    selection, normalization or loss diagnostics. Separate downstream calibration
    partitions must be constructed after freezing this checkpoint.
    """
    from backend.src.common.storage import calculate_file_sha256

    manifest = read_record(dataset / "frozen.json")
    config = manifest["config"]
    if config["vehicle"]["kind"] != "velocity_disc" or config["vehicle"]["output"] != "xy":
        raise ValueError("Bicycle targets require a separate dynamics/tracker adapter")
    horizon, dt = config["planner"]["horizon"], config["planner"]["dt"]
    rows = read_jsonl(dataset / "splits" / "train.jsonl")
    if not rows or len(rows) > MAX_EXPERT_TRAJECTORIES:
        raise ValueError("Training partition is empty or too large")
    if len({row["target_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate expert targets")
    # Validate *all* training targets before selection; incomplete scenes must
    # not mask malformed or unsafe input records.
    examples = [example_from_target(row, horizon, dt, manifest["dataset_id"]) for row in rows]
    if require_matched_triplets:
        selected = {row["target_id"] for row in _matched_guidance_rows(rows, config["guidance"])}
        examples = [example for row, example in zip(rows, examples, strict=True)
                    if row["target_id"] in selected]
    fit, validation = split_training_groups(examples, seed, validation_fraction)
    assigned = manifest["group_splits"]
    if any(assigned[example.group_id] != "train" for example in examples):
        raise ValueError("Expert calibration/test group leaked into training")
    fitted_guidance = {row.guidance for row in fit}
    if not {row.guidance for row in validation}.issubset(fitted_guidance):
        raise ValueError("Validation guidance has no fitted training examples")
    guidance = {
        g["name"]: {key: g[key] for key in ("side", "speed_scale", "obstacle_index")}
        for g in config["guidance"] if g["name"] in fitted_guidance
    }
    if any(row.guidance not in guidance for row in examples):
        raise ValueError("Unknown guidance in training set")
    return PreparedTraining(
        fit, validation, guidance, manifest["dataset_id"],
        calculate_file_sha256(dataset / "frozen.json"),
        tuple(sorted({r.group_id for r in fit})),
        tuple(sorted({r.group_id for r in validation})), horizon, dt,
        config["vehicle"]["footprint_radius"],
    )
