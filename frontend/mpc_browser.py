"""Replay a frozen trajectory-flow reference under closed-loop disc tracking MPC."""

import argparse
from pathlib import Path

import numpy as np

from backend.experiments.trajectory_flow.track import run_tracking_rollout
from backend.src.expert.config import PlannerSettings, VehicleSettings
from backend.src.expert.storage import read_record
from frontend.expert_browser import (
    SPLITS,
    load_prediction_model,
    read_expert_catalog,
    render_expert_svg,
    sample_predicted_path,
)


def render_tracking_svg(base: str, states: np.ndarray, predictions: tuple[np.ndarray, ...],
                        step: int, footprint_radius: float) -> str:
    """Add measured disc and MPC prediction to the expert browser's track SVG."""
    if (not base.endswith("</svg>") or states.ndim != 2 or states.shape[1] != 2
            or not np.isfinite(states).all() or not 0 <= step < len(states)
            or not np.isfinite(footprint_radius) or footprint_radius <= 0):
        raise ValueError("Invalid tracking visualization inputs")

    def points(path: np.ndarray) -> str:
        return " ".join(f"{x:.4f},{-y:.4f}" for x, y in path)

    path = states[:step + 1]
    x, y = path[-1]
    overlay = [
        f'<polyline data-kind="executed" points="{points(path)}" fill="none" '
        'stroke="#4ade80" stroke-width="0.085" stroke-linejoin="round"/>',
    ]
    if step < len(predictions):
        overlay.append(
            f'<polyline data-kind="mpc-prediction" points="{points(predictions[step])}" '
            'fill="none" stroke="#f8fafc" stroke-width="0.04" '
            'stroke-dasharray="0.10 0.06" opacity="0.8"/>'
        )
    overlay.append(
        f'<circle data-kind="vehicle" cx="{x:.4f}" cy="{-y:.4f}" '
        f'r="{footprint_radius:.4f}" fill="#4ade80" stroke="#f8fafc" '
        'stroke-width="0.045"/>'
    )
    return base[:-6] + "".join(overlay) + "</svg>"


def main(argv: list[str] | None = None) -> int:
    from nicegui import run, ui

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="Trusted weights digest from training receipt")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be in 1..65535")

    catalog = read_expert_catalog(args.dataset, args.split)
    model = load_prediction_model(args.checkpoint, args.sha256, catalog)
    manifest = read_record(args.dataset / "frozen.json")
    vehicle = VehicleSettings(**manifest["config"]["vehicle"])
    planner = PlannerSettings(**manifest["config"]["planner"])
    if (planner.horizon, planner.dt) != (catalog.horizon, catalog.dt):
        raise ValueError("Dataset and tracking physical grids disagree")

    def simulate(index: int):
        predicted = sample_predicted_path(model, catalog, index)
        return predicted, run_tracking_rollout(
            catalog.paths[index].scene, predicted, vehicle, planner,
            window=8, process_noise=0.0, seed=42,
        )

    @ui.page("/")
    def page() -> None:
        ui.query("body").style("background:#0a1423;color:#f0f6fc")
        reference, rollout, base = None, None, ""
        busy = False
        with ui.column().classes("w-full max-w-screen-xl mx-auto p-5 gap-3"):
            ui.label(f"{catalog.track.layout.upper()} / MPC TRACKING").classes(
                "text-2xl font-bold text-sky-200"
            )
            ui.label("Dashed colored: frozen flow y_ref · green: executed velocity-disc · "
                     "dotted white: next MPC prediction").classes("text-slate-400")
            with ui.card().classes("w-full max-w-3xl self-center").style("background:#142337"):
                drawing = ui.html(render_expert_svg(catalog, (), focus_index=0),
                                  sanitize=False).classes("w-full")
            with ui.card().classes("w-full").style("background:#142337"):
                with ui.row().classes("items-center gap-4"):
                    choice = ui.number("Scene index (0-based)", value=0, min=0,
                                       max=len(catalog.paths) - 1, step=1)
                    generate = ui.button("Sample y_ref and run MPC")
                status = ui.label("Choose a scene and simulate.").classes("text-sm text-amber-200")
                metrics = ui.label("One frozen reference; no diffusion replanning or conformal repair.")\
                    .classes("text-sm text-slate-300")
                with ui.row().classes("items-center gap-3"):
                    play = ui.button("Play")
                    pause = ui.button("Pause")
                    slider = ui.slider(min=0, max=catalog.horizon, value=0).classes("grow")
                    time_label = ui.label("k = 0").classes("text-sm text-slate-300")
                ui.label("Offline velocity-disc model, not a steering car; track containment "
                         "is checked before release, not constrained inside SLSQP.")\
                    .classes("text-xs text-slate-400")

        def draw() -> None:
            if rollout is None:
                return
            k = min(int(slider.value), len(rollout.states) - 1)
            drawing.set_content(render_tracking_svg(
                base, rollout.states, rollout.predictions, k, vehicle.footprint_radius
            ))
            time_label.set_text(f"k = {k} · t = {k * planner.dt:.1f} s")

        def advance() -> None:
            if rollout is None or int(slider.value) >= len(rollout.states) - 1:
                timer.deactivate()
                return
            slider.set_value(int(slider.value) + 1)

        timer = ui.timer(0.2, advance, active=False)
        slider.on_value_change(lambda _: draw())
        play.on_click(lambda: timer.activate() if rollout is not None else None)
        pause.on_click(timer.deactivate)

        async def generate_run() -> None:
            nonlocal reference, rollout, base, busy
            value = choice.value
            if busy or value is None or not float(value).is_integer():
                ui.notify("Select a whole-number scene index")
                return
            index = int(value)
            if not 0 <= index < len(catalog.paths):
                ui.notify("Scene index is out of range")
                return
            busy = True
            generate.disable()
            timer.deactivate()
            status.set_text("Sampling frozen flow and solving receding-horizon MPC...")
            try:
                predicted, simulated = await run.io_bound(simulate, index)
                reference, rollout = predicted, simulated
                base = render_expert_svg(catalog, (), {index: reference}, focus_index=index)
                slider.set_value(0)
                draw()
                error = np.linalg.norm(rollout.states - reference[:len(rollout.states)], axis=1)
                metrics.set_text(
                    f"{len(rollout.controls)}/{planner.horizon} steps · "
                    f"position RMS {np.sqrt(np.mean(error**2)):.3f} m · "
                    f"goal distance {np.linalg.norm(rollout.states[-1] - catalog.paths[index].scene.goal):.3f} m"
                )
                status.set_text(f"Simulation status: {rollout.status}")
            except (ValueError, RuntimeError, OSError) as error:
                status.set_text(f"Simulation failed: {str(error)[:160]}")
            finally:
                busy = False
                generate.enable()

        generate.on_click(generate_run)

    ui.run(host="127.0.0.1", port=args.port, title="Flow reference to tracking MPC",
           reload=False, show=False)
    return 0


if __name__ in {"__main__", "__mp_main__"}:
    raise SystemExit(main())
