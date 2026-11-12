"""Run a localhost-only NiceGUI explorer for a trusted frozen trajectory field."""

import argparse
from pathlib import Path

from nicegui import run, ui

from backend.src.trajectory_flow.train import load_flow_checkpoint
from frontend.visualization import (
    MAX_REFERENCES,
    PALETTE,
    read_scene_catalog,
    render_scene_svg,
    sample_references,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="Trusted SHA-256 from the training receipt")
    parser.add_argument("--dataset", type=Path, required=True, help="Completed expert dataset directory")
    parser.add_argument("--split", choices=("train", "reference", "future_calibration", "test"),
                        default="train")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be in 1..65535")

    # Load once and verify the supplied trusted digest; never create random weights.
    model = load_flow_checkpoint(args.checkpoint, expected_sha256=args.sha256)
    catalog = read_scene_catalog(args.dataset, args.split)
    scenes = {scene.scene_id: scene for scene in catalog.scenes}
    horizon, dt = model.architecture.horizon, model.architecture.dt
    if (catalog.horizon, catalog.dt) != (horizon, dt):
        raise ValueError("Expert dataset horizon/dt differs from frozen model")

    def compatible_guidance(scene):
        return [name for name, spec in model.guidance.items()
                if spec["side"] == 0 or spec["obstacle_index"] < len(scene.obstacles)]

    @ui.page("/")
    def visualizer() -> None:
        ui.query("body").style("background:#090f1c;color:#e2e8f0")
        first_scene = catalog.scenes[0]
        chosen_names = compatible_guidance(first_scene)
        references = ()  # Per-browser-page visual state; no shared mutable samples.
        busy = False

        with ui.column().classes("w-full max-w-screen-2xl mx-auto p-4 gap-4"):
            ui.label("TRAJECTORY FLOW / CIRCUIT EXPLORER").classes(
                "text-2xl font-bold tracking-wide text-sky-300"
            )
            ui.label("Frozen vector-field references · physical x/y in metres · "
                     "moving obstacles at prediction time i·Δt").classes("text-slate-400")
            with ui.row().classes("w-full items-start gap-4 flex-wrap"):
                with ui.card().classes("w-72 shrink-0").style(
                    "background:#142137;color:#e2e8f0"
                ):
                    ui.label("SCENE & SAMPLING").classes("font-bold text-sky-300")
                    selector = ui.select(
                        {scene.scene_id: f"{scene.scene_id} · {scene.map_id}"
                         for scene in catalog.scenes},
                        label="Expert scene", value=first_scene.scene_id,
                    ).classes("w-full")
                    guidance = ui.select(chosen_names, value=chosen_names[0] if chosen_names else None,
                                         label="Conditioning guidance").classes("w-full")
                    compare = ui.checkbox("Compare all compatible guidance", value=False)
                    seed = ui.number("Noise seed", value=42, min=0, max=2**32 - 1,
                                     precision=0).classes("w-full")
                    count = ui.number("Samples per guidance", value=1, min=1,
                                      max=3, precision=0).classes("w-full")
                    solver = ui.select(["heun", "euler"], value="heun",
                                       label="Integrator").classes("w-full")
                    steps = ui.number("Integration steps", value=32, min=1,
                                      max=128, precision=0).classes("w-full")
                    generate_button = ui.button("Generate references", icon="route").classes(
                        "w-full"
                    )
                    ui.label(f"Checkpoint: {model.revision[:12]} · Dataset: "
                             f"{catalog.dataset_id[:12]} · Split: {catalog.split}").classes(
                        "text-xs text-slate-400 break-all"
                    )
                    ui.label("These are reference proposals, not MPC-tracked paths "
                             "or certified-safe trajectories.").classes("text-xs text-amber-300")
                with ui.column().classes("flex-1 min-w-80 gap-3"):
                    with ui.card().classes("w-full").style(
                        "background:#142137;color:#e2e8f0"
                    ):
                        status = ui.label("Choose a scene and generate a reference.").classes(
                            "text-sm text-slate-300"
                        )
                        # Only fixed SVG tags and finite numeric coordinates enter this HTML.
                        plot = ui.html("", sanitize=False).classes("w-full")
                        ui.label("START: green · GOAL: white · obstacle: orange · "
                                 "dashed ring: obstacle + vehicle footprint").classes(
                            "text-xs text-slate-400"
                        )
                        with ui.row().classes("w-full items-center"):
                            time_label = ui.label("Prediction index i = 0").classes(
                                "w-48 text-sm text-slate-300"
                            )
                            timeline = ui.slider(min=0, max=horizon, value=0, step=1).classes(
                                "flex-1"
                            )
                    with ui.card().classes("w-full").style(
                        "background:#142137;color:#e2e8f0"
                    ):
                        ui.label("GENERATED REFERENCES").classes("font-bold text-sky-300")
                        legend = ui.column().classes("w-full gap-1")
                        ui.label("Clearances are signed metres; negative means geometric overlap. "
                                 "They do not account for MPC tracking error or uncertainty.").classes(
                            "text-xs text-slate-400"
                        )

        def draw() -> None:
            scene = scenes[selector.value]
            index = int(timeline.value)
            plot.set_content(render_scene_svg(
                scene, references, footprint_radius=catalog.footprint_radius,
                index=index, horizon=horizon, dt=dt,
            ))
            time_label.set_text(f"i = {index} · +{index * dt:.2f} s")

        def change_scene() -> None:
            nonlocal references
            references = ()
            scene = scenes[selector.value]
            choices = compatible_guidance(scene)
            guidance.set_options(choices, value=choices[0] if choices else None)
            timeline.set_value(0)
            status.set_text(f"{scene.scene_id}: scene loaded; generate to sample the field.")
            legend.clear()
            draw()

        async def generate() -> None:
            nonlocal references, busy
            if busy:
                return
            busy = True
            generate_button.disable()
            scene_id = selector.value
            scene = scenes[scene_id]
            names = (tuple(compatible_guidance(scene)) if compare.value
                     else (guidance.value,))
            try:
                # Values from browser widgets are untrusted even with min/max props.
                if (seed.value is None or count.value is None or steps.value is None
                        or not float(seed.value).is_integer()
                        or not float(count.value).is_integer()
                        or not float(steps.value).is_integer()):
                    raise ValueError("Seed, samples and steps must be whole numbers")
                if len(names) * int(count.value) > MAX_REFERENCES:
                    raise ValueError(f"At most {MAX_REFERENCES} references per request")
                requested_solver, requested_steps = solver.value, int(steps.value)
                result = await run.io_bound(
                    sample_references, model, scene, names,
                    seed=int(seed.value), samples_per_guidance=int(count.value),
                    steps=requested_steps, solver=requested_solver,
                    footprint_radius=catalog.footprint_radius,
                )
                if selector.value != scene_id:
                    return  # Discard an old scene's result, never mislabel a plot.
                references = result
                legend.clear()
                colors = {name: PALETTE[i % len(PALETTE)]
                          for i, name in enumerate(dict.fromkeys(r.guidance for r in references))}
                with legend:
                    for reference in references:
                        obs = ("none" if reference.obstacle_clearance is None
                               else f"{reference.obstacle_clearance:+.3f} m")
                        with ui.row().classes("items-center gap-2"):
                            ui.label("●").style(f"color:{colors[reference.guidance]}")
                            ui.label(
                                f"{reference.guidance} · seed {reference.seed} · "
                                f"obstacle {obs} · boundary "
                                f"{reference.boundary_clearance:+.3f} m · "
                                f"{reference.latency_seconds*1000:.0f} ms"
                            ).classes("text-sm")
                status.set_text(f"{scene_id}: {len(references)} raw reference(s) sampled "
                                f"with {requested_solver}/{requested_steps} steps.")
                draw()
            except (ValueError, KeyError, TypeError) as error:
                status.set_text(f"Cannot generate: {error}")
            finally:
                busy = False
                generate_button.enable()

        selector.on_value_change(lambda _: change_scene())
        timeline.on_value_change(lambda _: draw())
        generate_button.on_click(generate)
        draw()
        if chosen_names:
            ui.timer(0.1, generate, once=True)

    ui.run(host="127.0.0.1", port=args.port, title="Conf-Gen trajectory explorer",
           reload=False, show=False)
    return 0


if __name__ in {"__main__", "__mp_main__"}:
    raise SystemExit(main())
