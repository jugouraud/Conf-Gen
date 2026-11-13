"""Inspect expert paths and optional frozen-field predictions on one fixed circuit."""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from backend.src.common.storage import calculate_file_sha256
from backend.src.expert.audit import read_jsonl
from backend.src.expert.config import Guidance
from backend.src.expert.scenes import Scene
from backend.src.expert.storage import MAX_EXPERT_TRAJECTORIES, MAX_JSON_BYTES, parse_json, read_record
from backend.src.expert.track import TrackSpec, track_for

if TYPE_CHECKING:
    from backend.src.trajectory_flow.model import TrajectoryFlowModel

COLORS = ("#38bdf8", "#f472b6", "#facc15", "#a78bfa", "#4ade80", "#fb923c")
SPLITS = ("train", "reference", "future_calibration", "test")
MAX_OVERLAYS = 24
MAX_CATALOG_PATHS = MAX_EXPERT_TRAJECTORIES
PAGE_SIZE = 100


@dataclass(frozen=True)
class ExpertCatalogPager:
    directory: Path
    split: str
    offsets: tuple[int, ...]

    @property
    def pages(self) -> int:
        return (len(self.offsets) + PAGE_SIZE - 1) // PAGE_SIZE

    def load(self, page: int) -> "ExpertCatalog":
        if type(page) is not int or not 0 <= page < self.pages:
            raise ValueError("Invalid catalog page")
        start = page * PAGE_SIZE
        rows = []
        with (self.directory / "splits" / f"{self.split}.jsonl").open("rb") as stream:
            for offset in self.offsets[start:start + PAGE_SIZE]:
                stream.seek(offset)
                rows.append(parse_json(stream.readline(MAX_JSON_BYTES + 1)))
        return _catalog_from_rows(rows, self.split)


@dataclass(frozen=True)
class ExpertPath:
    target_id: str
    scene: Scene
    points: np.ndarray  # Shape (H+1, 2), metres, independently validated.
    guidance: Guidance


@dataclass(frozen=True)
class ExpertCatalog:
    paths: tuple[ExpertPath, ...]
    track: TrackSpec
    footprint_radius: float
    required_clearance: float
    dataset_id: str
    split: str
    horizon: int
    dt: float


def open_expert_catalog_pager(directory: Path, split: str) -> ExpertCatalogPager:
    """Index byte offsets only; trajectory JSON is parsed page-by-page on demand."""
    if split not in SPLITS:
        raise ValueError("Unknown dataset split")
    path = directory / "splits" / f"{split}.jsonl"
    offsets = []
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline(MAX_JSON_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_JSON_BYTES:
                raise ValueError("Expert target exceeds the JSON size limit")
            offsets.append(offset)
            if len(offsets) > MAX_CATALOG_PATHS:
                raise ValueError("Invalid split catalog")
    if not offsets:
        raise ValueError(f"No validated trajectories in {split} split")
    return ExpertCatalogPager(directory, split, tuple(offsets))


def _catalog_from_rows(rows: list[dict], split: str) -> ExpertCatalog:
    first = rows[0]
    vehicle, planner = first["vehicle"], first["planner"]
    paths = []
    for target in rows:
        target_id = target.get("target_id")
        if (target.get("schema") != "expert-target-v1" or target.get("split") != split
                or target.get("dataset_id") != first.get("dataset_id")
                or not isinstance(target_id, str)
                or not re.fullmatch(r"s\d{4,5}-g\d+-i\d+-r\d+", target_id)
                or target.get("validation", {}).get("valid") is not True):
            raise ValueError("Invalid or unvalidated expert target")
        scene = Scene.from_record(target["observation"])
        points = np.asarray(target["y_ref_star"], dtype=float)
        if (scene.track is None or points.shape != (planner["horizon"] + 1, 2)
                or not np.isfinite(points).all() or target["vehicle"] != vehicle
                or target["planner"] != planner):
            raise ValueError("Invalid expert trajectory page")
        paths.append(ExpertPath(target_id, scene, points, Guidance(**target["guidance"])))
    track = paths[0].scene.track
    if any(path.scene.track != track for path in paths):
        raise ValueError("Dataset contains different circuit geometry")
    return ExpertCatalog(tuple(paths), track, float(vehicle["footprint_radius"]),
                         float(planner["required_clearance"]), first["dataset_id"], split,
                         int(planner["horizon"]), float(planner["dt"]))


def read_expert_catalog(directory: Path, split: str = "train") -> ExpertCatalog:
    """Read selected validated labels, with bounded IDs and on-disk digests."""
    if split not in SPLITS:
        raise ValueError("Unknown dataset split")
    manifest = read_record(directory / "frozen.json")
    index = read_record(directory / "index.json")
    if (manifest.get("schema") != "expert-dataset-v1"
            or manifest.get("dataset_id") != index.get("dataset_id")):
        raise ValueError("Dataset manifest and index disagree")
    vehicle = manifest["config"]["vehicle"]
    planner = manifest["config"]["planner"]
    if vehicle["kind"] != "velocity_disc" or vehicle["output"] != "xy":
        raise ValueError("Only velocity_disc xy expert targets are supported")
    ids = index["target_ids"]
    if (
        not isinstance(ids, list)
        or len(ids) > MAX_CATALOG_PATHS
        or len(ids) != len(set(ids))
    ):
        raise ValueError("Invalid target index")
    indexed_ids = set(ids)
    split_path = directory / "splits" / f"{split}.jsonl"
    rows = None
    if split_path.is_file():
        relative_path = f"splits/{split}.jsonl"
        if calculate_file_sha256(split_path) != index["files"].get(relative_path):
            raise ValueError("Split digest differs from completed index")
        rows = read_jsonl(split_path)
        if not rows or len(rows) > MAX_CATALOG_PATHS:
            raise ValueError("Invalid split catalog")
    paths = []
    if rows is None:
        rows = []
        for target_id in ids:
            if not isinstance(target_id, str) or not re.fullmatch(r"s\d{4}-g\d+-i\d+-r\d+", target_id):
                raise ValueError("Invalid target ID")
            relative_path = f"targets/{target_id}.json"
            path = directory / relative_path
            if calculate_file_sha256(path) != index["files"].get(relative_path):
                raise ValueError("Target digest differs from completed index")
            rows.append(read_record(path))
    for target in rows:
        target_id = target.get("target_id")
        if (target.get("schema") != "expert-target-v1"
                or target.get("dataset_id") != index["dataset_id"]
                or not isinstance(target_id, str)
                or target_id not in indexed_ids
                or target.get("validation", {}).get("valid") is not True):
            raise ValueError("Invalid or unvalidated expert target")
        if target["split"] != split:
            continue
        scene = Scene.from_record(target["observation"])
        if scene.track is None:
            raise ValueError("Expert viewer requires a fixed circuit")
        points = np.asarray(target["y_ref_star"], dtype=float)
        if points.shape != (planner["horizon"] + 1, 2) or not np.isfinite(points).all():
            raise ValueError("Invalid expert trajectory coordinates")
        guidance = Guidance(**target["guidance"])
        paths.append(ExpertPath(target_id, scene, points, guidance))
    if not paths:
        raise ValueError(f"No validated trajectories in {split} split")
    first = paths[0].scene
    for entry in paths:
        scene = entry.scene
        if scene.track != first.track:
            raise ValueError("Dataset contains different circuit geometry")
        for obstacle in scene.obstacles:
            if not np.allclose(obstacle.centers, obstacle.centers[0], atol=1e-12, rtol=0):
                raise ValueError("Dataset contains moving obstacles")
    return ExpertCatalog(
        tuple(paths), first.track, float(vehicle["footprint_radius"]),
        float(planner["required_clearance"]), index["dataset_id"], split,
        int(planner["horizon"]), float(planner["dt"]),
    )


def load_prediction_model(
    checkpoint: Path, expected_sha256: str, catalog: ExpertCatalog
) -> TrajectoryFlowModel:
    """Require the trusted checkpoint to represent this dataset and guidance."""
    from backend.src.trajectory_flow.train import load_flow_checkpoint

    model = load_flow_checkpoint(checkpoint, expected_sha256=expected_sha256)
    metadata = read_record(checkpoint / "metadata.json")
    if metadata.get("dataset_id") != catalog.dataset_id:
        raise ValueError("Checkpoint was trained on a different expert dataset")
    if (model.architecture.horizon, model.architecture.dt) != (catalog.horizon, catalog.dt):
        raise ValueError("Checkpoint horizon/dt differs from the expert dataset")
    for entry in catalog.paths:
        expected = asdict(entry.guidance)
        name = expected.pop("name")
        if model.guidance.get(name) != expected:
            raise ValueError(f"Checkpoint guidance differs for {name}")
    return model


def sample_predicted_path(
    model: TrajectoryFlowModel, catalog: ExpertCatalog, index: int
) -> np.ndarray:
    """Use the real frozen field on the selected expert scene and guidance."""
    from backend.src.trajectory_flow.sample import run_flow_sampling
    from frontend.visualization import context_from_scene

    if type(index) is not int or not 0 <= index < len(catalog.paths):
        raise ValueError("Selected trajectory index is out of range")
    entry = catalog.paths[index]
    context = context_from_scene(entry.scene, catalog.horizon, catalog.dt)
    sample = run_flow_sampling(
        model, context, entry.guidance.name, seed=42, steps=32, solver="heun"
    )
    points = np.asarray(sample.reference, dtype=float)
    if points.shape != entry.points.shape or not np.isfinite(points).all():
        raise ValueError("Predicted trajectory has invalid coordinates")
    return points


def render_expert_svg(
    catalog: ExpertCatalog, selected: tuple[int, ...],
    predictions: dict[int, np.ndarray] | None = None,
    *,
    focus_index: int = 0,
) -> str:
    """Fixed track art; annotate the obstacle of each visible scene."""
    predictions = {} if predictions is None else predictions
    if len(selected) + len(predictions) > MAX_OVERLAYS:
        raise ValueError("Too many displayed trajectories")
    visible_indices = set(selected) | set(predictions) | {focus_index}
    if any(type(index) is not int or not 0 <= index < len(catalog.paths)
           for index in visible_indices):
        raise ValueError("Selected trajectory index is out of range")
    track = track_for(catalog.track)
    spec = track.spec
    extent = max(spec.left_width, spec.right_width) + 0.75
    x0 = spec.center[0] - spec.radii[0] - extent
    y0 = spec.center[1] - spec.radii[1] - extent
    width = 2 * (spec.radii[0] + extent)
    height = 2 * (spec.radii[1] + extent)
    arc = np.linspace(0, track.length, 512, endpoint=False)
    center = track.point(arc)
    normal = track.normal(arc)
    outer = center - spec.right_width * normal
    inner = center + spec.left_width * normal

    def xy(points: np.ndarray) -> str:
        return " ".join(f"{x:.4f},{-y:.4f}" for x, y in points)

    def shape(points: np.ndarray) -> str:
        return f"M{xy(points).replace(' ', ' L')} Z"

    # Same track drawing and viewBox for every selection; obstacle positions vary.
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.3f} {-y0-height:.3f} '
        f'{width:.3f} {height:.3f}" style="width:100%;display:block;border-radius:18px">',
        '<defs><linearGradient id="grass" x2="1" y2="1"><stop stop-color="#172d36"/>'
        '<stop offset="1" stop-color="#0b192c"/></linearGradient>'
        '<linearGradient id="asphalt" x2="0.8" y2="1"><stop stop-color="#34465c"/>'
        '<stop offset="1" stop-color="#202f42"/></linearGradient>'
        '<pattern id="texture" width="0.22" height="0.22" patternUnits="userSpaceOnUse">'
        '<circle cx="0.03" cy="0.03" r="0.012" fill="#d1e9f6" opacity="0.10"/></pattern></defs>',
        f'<rect x="{x0:.3f}" y="{-y0-height:.3f}" width="{width:.3f}" '
        'height="{height:.3f}" fill="url(#grass)"/>',
        f'<path d="{shape(outer)} {shape(inner[::-1])}" fill="url(#asphalt)" '
        'fill-rule="evenodd"/>',
        f'<path d="{shape(outer)}" fill="none" stroke="#d0e4e9" stroke-width="0.048"/>',
        f'<path d="{shape(inner)}" fill="none" stroke="#d0e4e9" stroke-width="0.048"/>',
        f'<path d="{shape(center)}" fill="none" stroke="#d9bf7b" opacity="0.65" '
        'stroke-width="0.024" stroke-dasharray="0.19 0.14"/>',
        f'<path d="{shape(outer)} {shape(inner[::-1])}" fill="url(#texture)" '
        'fill-rule="evenodd"/>',
    ]
    for index in sorted(visible_indices):
        color = COLORS[index % len(COLORS)]
        for obstacle in catalog.paths[index].scene.obstacles:
            x, y = obstacle.centers[0]
            safe_radius = (obstacle.radius + catalog.footprint_radius
                           + catalog.required_clearance)
            parts.extend((
                f'<circle data-kind="obstacle-clearance" data-index="{index}" '
                f'cx="{x:.4f}" cy="{-y:.4f}" r="{safe_radius:.4f}" '
                f'fill="{color}" fill-opacity="0.10" stroke="{color}" '
                'stroke-width="0.035" stroke-dasharray="0.09 0.06"/>',
                f'<circle data-kind="obstacle" data-index="{index}" '
                f'cx="{x:.4f}" cy="{-y:.4f}" r="{obstacle.radius:.4f}" '
                f'fill="#ea580c" stroke="{color}" stroke-width="0.042"/>',
            ))
    for index in selected:
        if type(index) is not int or not 0 <= index < len(catalog.paths):
            raise ValueError("Selected trajectory index is out of range")
        entry = catalog.paths[index]
        color = COLORS[index % len(COLORS)]
        parts.append(f'<polyline data-kind="expert" points="{xy(entry.points)}" fill="none" '
                     f'stroke="{color}" stroke-width="0.07" stroke-linecap="round" '
                     'stroke-linejoin="round" opacity="0.97"/>')
        for point, radius, fill in ((entry.points[0], 0.105, "#86efac"),
                                    (entry.points[-1], 0.105, "#f8fafc")):
            parts.append(f'<circle cx="{point[0]:.4f}" cy="{-point[1]:.4f}" '
                         f'r="{radius:.3f}" fill="{fill}" stroke="{color}" '
                         'stroke-width="0.035"/>')
    for index, points in predictions.items():
        if type(index) is not int or not 0 <= index < len(catalog.paths):
            raise ValueError("Selected trajectory index is out of range")
        points = np.asarray(points, dtype=float)
        if points.shape != catalog.paths[index].points.shape or not np.isfinite(points).all():
            raise ValueError("Invalid predicted trajectory coordinates")
        color = COLORS[index % len(COLORS)]
        parts.append(f'<polyline data-kind="predicted" points="{xy(points)}" fill="none" '
                     f'stroke="{color}" stroke-width="0.075" stroke-linecap="round" '
                     'stroke-linejoin="round" stroke-dasharray="0.12 0.08" opacity="0.95"/>')
        for point, fill in ((points[0], "#86efac"), (points[-1], "#f8fafc")):
            parts.append(f'<circle cx="{point[0]:.4f}" cy="{-point[1]:.4f}" '
                         f'r="0.075" fill="{fill}" stroke="{color}" stroke-width="0.028"/>')
    return "".join(parts) + "</svg>"


def main(argv: list[str] | None = None) -> int:
    from nicegui import run, ui  # GUI is optional for pure catalog/renderer tests.

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--checkpoint", type=Path, help="Frozen trajectory-flow checkpoint")
    parser.add_argument("--sha256", help="Trusted weight digest from the training receipt")
    parser.add_argument("--page", type=int, default=1, help="One-based trajectory page (100 paths)")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be in 1..65535")
    if (args.checkpoint is None) != (args.sha256 is None):
        parser.error("--checkpoint and --sha256 must be supplied together")
    pager = open_expert_catalog_pager(args.dataset, args.split)
    if not 1 <= args.page <= pager.pages:
        parser.error(f"--page must be in 1..{pager.pages}")
    catalog = pager.load(args.page - 1)
    model = (load_prediction_model(args.checkpoint, args.sha256, catalog)
             if args.checkpoint is not None else None)

    @ui.page("/")
    def page() -> None:
        ui.query("body").style("background:#0a1423;color:#f0f6fc")
        chosen_experts = {0}
        chosen_predictions: set[int] = set()
        predictions: dict[int, np.ndarray] = {}
        pending: set[int] = set()
        with ui.column().classes("w-full max-w-screen-xl mx-auto p-5 gap-3"):
            ui.label(f"{catalog.track.layout.upper()} / TRAJECTORY EXPLORER").classes(
                "text-2xl font-bold tracking-wider text-sky-200")
            ui.label("Fixed track · varying obstacles · solid: expert y_ref* · "
                     "dashed: predicted y_ref").classes("text-slate-400")
            with ui.card().classes("w-full max-w-3xl self-center").style("background:#142337"):
                drawing = ui.html(render_expert_svg(catalog, (0,)), sanitize=False).classes("w-full")
                ui.label("Green: start · White: goal · Orange: obstacles; "
                         "colored rings link paths").classes("text-xs text-slate-300")
            with ui.card().classes("w-full").style("background:#142337"):
                ui.label(f"Dataset {catalog.dataset_id[:12]} · {catalog.split} · "
                         f"{len(catalog.paths)} validated expert targets").classes("text-sm text-slate-300")
                details = ui.label("").classes("text-sm text-sky-200")
                selection = ui.label("").classes("text-xs text-slate-400")
                status = ui.label(
                    "Supply --checkpoint and --sha256 to enable predictions."
                    if model is None else "Predictions are sampled on first selection (Heun/32, seed 42)."
                ).classes("text-xs text-amber-200")
                ui.label("Predictions are raw model references, not validated safe paths or MPC rollouts.")\
                    .classes("text-xs text-slate-400")
                ui.label("Select trajectories to overlay").classes("text-sm font-medium text-slate-200")
                with ui.scroll_area().classes("w-full h-80 rounded border border-slate-600 p-2"):
                    trajectory_rows = ui.column().classes("w-full gap-1")

        def draw() -> None:
            index = focus_index
            entry = catalog.paths[index]
            visible = {i: predictions[i] for i in chosen_predictions if i in predictions}
            drawing.set_content(render_expert_svg(
                catalog, tuple(sorted(chosen_experts)), visible, focus_index=index
            ))
            length = float(np.linalg.norm(np.diff(entry.points, axis=0), axis=1).sum())
            detail = (f"{index + 1}/{len(catalog.paths)} · {entry.scene.scene_id} · "
                      f"{entry.guidance.name} · expert {length:.2f} m")
            if index in predictions:
                prediction = predictions[index]
                error = np.sqrt(np.mean(np.sum((prediction - entry.points) ** 2, axis=1)))
                detail += f" · predicted RMS error {error:.2f} m"
            details.set_text(detail)
            selection.set_text(f"{len(chosen_experts)} expert + "
                               f"{len(chosen_predictions)} predicted checked")

        def toggle_expert(index: int, checkbox) -> None:
            nonlocal focus_index
            focus_index = index
            if checkbox.value:
                if len(chosen_experts) + len(chosen_predictions) >= MAX_OVERLAYS:
                    ui.notify(f"At most {MAX_OVERLAYS} simultaneous paths")
                    checkbox.set_value(False)
                    return
                chosen_experts.add(index)
            else:
                chosen_experts.discard(index)
            draw()

        async def toggle_prediction(index: int, checkbox) -> None:
            nonlocal focus_index
            focus_index = index
            if model is None:
                return
            if not checkbox.value:
                chosen_predictions.discard(index)
                draw()
                return
            if len(chosen_experts) + len(chosen_predictions) >= MAX_OVERLAYS:
                ui.notify(f"At most {MAX_OVERLAYS} simultaneous paths")
                checkbox.set_value(False)
                return
            chosen_predictions.add(index)
            draw()
            if index in predictions or index in pending:
                return
            if pending:
                chosen_predictions.discard(index)
                checkbox.set_value(False)
                draw()
                return
            pending.add(index)
            checkbox.disable()
            status.set_text(f"Sampling predicted path for {catalog.paths[index].scene.scene_id}...")
            try:
                points = await run.io_bound(sample_predicted_path, model, catalog, index)
                predictions[index] = points
                status.set_text("Prediction sampled from the frozen vector field.")
            except (ValueError, RuntimeError, OSError) as error:
                chosen_predictions.discard(index)
                status.set_text(f"Prediction failed: {str(error)[:160]}")
                checkbox.set_value(False)
            finally:
                pending.remove(index)
                checkbox.enable()
                draw()

        focus_index = 0
        with trajectory_rows:
            for index, entry in enumerate(catalog.paths):
                with ui.row().classes("w-full items-center gap-3 rounded px-2 py-1 hover:bg-slate-700"):
                    ui.label(f"{index + 1:>3} · {entry.scene.scene_id} · {entry.guidance.name}") \
                        .classes("grow text-xs text-slate-300")
                    expert = ui.checkbox("Expert", value=index == 0).props("dense")
                    expert.on_value_change(
                        lambda _, index=index, checkbox=expert: toggle_expert(index, checkbox)
                    )
                    predicted = ui.checkbox("Predicted", value=False).props("dense")
                    if model is None:
                        predicted.disable()
                    else:
                        async def on_prediction_change(_, index=index, checkbox=predicted) -> None:
                            await toggle_prediction(index, checkbox)

                        predicted.on_value_change(on_prediction_change)
        draw()

    ui.run(host="127.0.0.1", port=args.port, title="Expert and predicted trajectory browser",
           reload=False, show=False)
    return 0


if __name__ in {"__main__", "__mp_main__"}:
    raise SystemExit(main())
