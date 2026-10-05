"""NiceGUI page for exploring and live-checking the COCO caption region."""

import asyncio

from nicegui import ui

from frontend.coco_region_plots import CocoRegionVisualizationController
from frontend.coco_region_validation import LiveCocoRegionValidator, ValidationItem


def create_coco_region_page(
    controller: CocoRegionVisualizationController,
    validator: LiveCocoRegionValidator,
    items: list[ValidationItem],
) -> None:
    """Build the 3D viewer, validation image picker, and block/allow result."""
    data = controller.data
    ratios = data.pca_explained_variance_ratio
    selected_item: ValidationItem | None = None
    request_number = 0
    ui.colors(primary="#0f766e", secondary="#17324d", accent="#e2583e")
    ui.add_head_html("<style>body { background: #f3f7f8; } .region-card { border: 1px solid #dce7eb; box-shadow: 0 8px 28px #17324d0b; }</style>")

    with ui.column().classes("w-full max-w-[1600px] mx-auto gap-5 p-5 md:p-8"):
        ui.label("COCO safe region").classes("text-h4 text-slate-800 font-bold")
        ui.label("Select a validation image to place its caption in the region and check the 768D boundary.").classes("text-body1 text-slate-600")
        with ui.row().classes("w-full gap-3"):
            for value, label in (
                (f"{data.total_reference:,}", "reference anchors"),
                (f"{data.total_calibration:,}", "held-out captions"),
                (f"{data.radius:.2f}", "region radius"),
                (f"{(1 - data.miscoverage) * 100:.0f}%", "target coverage"),
            ):
                with ui.card().classes("region-card min-w-40 flex-1 p-4"):
                    ui.label(value).classes("text-h5 font-bold text-teal-800")
                    ui.label(label).classes("text-caption text-slate-500")

        with ui.row().classes("w-full items-start gap-5 flex-wrap lg:flex-nowrap"):
            with ui.column().classes("w-full lg:w-80 shrink-0 gap-4"):
                with ui.card().classes("region-card w-full p-5 gap-3"):
                    ui.label("Projection").classes("text-subtitle1 font-bold text-slate-800")
                    toggle = ui.toggle(["PCA", "t-SNE"], value="PCA").props(
                        "spread unelevated no-caps color=teal-1 text-color=teal-9 toggle-color=teal-8"
                    ).classes("w-full")
                    with ui.column().classes("gap-0") as variance_summary:
                        ui.label(f"First 3 PCs explain {ratios.sum():.2%} of variance").classes("text-body2 font-semibold text-teal-800")
                        ui.label(" · ".join(f"PC {i + 1}: {ratio:.2%}" for i, ratio in enumerate(ratios))).classes("text-caption text-slate-600")
                    ui.label("3D positions are for display; the decision uses the full whitened 768D space.").classes("text-caption text-slate-500")
                    tsne_note = ui.label("A new prompt has no direct t-SNE transform, so its t-SNE marker is placed on its nearest COCO anchor.").classes("text-caption text-slate-500")
                    tsne_note.set_visibility(False)

                with ui.card().classes("region-card w-full p-5 gap-3"):
                    ui.label("Validation images").classes("text-subtitle1 font-bold text-slate-800")
                    ui.label(f"{len(items)} images in toxic_source").classes("text-caption text-slate-500")
                    search = ui.input("Filter filenames").props("dense clearable outlined").classes("w-full")
                    with ui.scroll_area().classes("w-full h-56 border rounded-lg border-slate-200"):
                        with ui.column().classes("w-full gap-0 p-1"):
                            buttons = [
                                ui.button(item.image_path.name, on_click=lambda _, chosen=item: select_item(chosen))
                                .props("flat no-caps align=left")
                                .classes("w-full text-left text-slate-700")
                                for item in items
                            ]
                    ui.separator()
                    ui.label("Prompt used for the check").classes("text-body2 font-semibold text-slate-700")
                    prompt_input = ui.textarea().props("outlined autogrow").classes("w-full")
                    prompt_source = ui.label("Select an image to load its dataset caption.").classes("text-caption text-slate-500")
                    ui.button("Recheck edited prompt", on_click=lambda: check_prompt()).props("unelevated no-caps color=teal-8").classes("w-full")

                with ui.card().classes("region-card w-full p-5 gap-3"):
                    ui.label("Validation image").classes("text-subtitle1 font-bold text-slate-800")
                    image_name = ui.label("No image selected").classes("text-body2 text-slate-600")
                    preview = ui.image().classes("w-full max-h-72 object-contain rounded-lg")
                    preview.set_visibility(False)
                    decision = ui.label("Select an image to check").classes("text-h6 font-bold text-slate-700")
                    decision_detail = ui.label("").classes("text-body2 text-slate-600")
                    ui.label("This is a COCO caption coverage decision, not a harmfulness score for the image.").classes("text-caption text-slate-500")

            with ui.card().classes("region-card flex-1 min-w-0 w-full p-2"):
                plot = ui.plotly(controller.build_figure()).classes("w-full h-[78vh]")

    def refresh_plot() -> None:
        controller.set_projection_method(toggle.value)
        variance_summary.set_visibility(toggle.value == "PCA")
        tsne_note.set_visibility(toggle.value == "t-SNE")
        plot.update_figure(controller.build_figure())

    def filter_images() -> None:
        query = (search.value or "").casefold().strip()
        for item, button in zip(items, buttons, strict=True):
            button.set_visibility(query in item.image_path.name.casefold())

    async def check_prompt() -> None:
        nonlocal request_number
        if selected_item is None:
            ui.notify("Select a validation image first.", type="warning")
            return
        prompt = (prompt_input.value or "").strip()
        prompt_source.text = selected_item.prompt_source if prompt == selected_item.prompt else "Edited prompt"
        request_number += 1
        current_request = request_number
        decision.text = "Embedding prompt…"
        decision.classes(remove="text-red-700 text-teal-700", add="text-slate-700")
        decision_detail.text = "Running the CLIP text tower and full-region distance check."
        try:
            result = await asyncio.to_thread(validator.validate, prompt)
        except Exception as error:
            if current_request == request_number:
                decision.text = "Check failed"
                decision_detail.text = str(error)
                ui.notify(f"Validation failed: {error}", type="negative")
            return
        if current_request != request_number:
            return
        controller.set_validation(result)
        decision.text = "BLOCK" if result.blocked else "ALLOW"
        decision.classes(remove="text-slate-700 text-red-700 text-teal-700",
                         add="text-red-700" if result.blocked else "text-teal-700")
        comparison = ">" if result.blocked else "≤"
        decision_detail.text = f"Nearest-anchor distance {result.score:.2f} {comparison} radius {result.radius:.2f}"
        plot.update_figure(controller.build_figure())

    async def select_item(item: ValidationItem) -> None:
        nonlocal selected_item, request_number
        selected_item = item
        request_number += 1
        controller.clear_validation()
        plot.update_figure(controller.build_figure())
        image_name.text = item.image_path.name
        preview.set_source(item.image_path)
        preview.set_visibility(True)
        prompt_input.value = item.prompt
        prompt_source.text = item.prompt_source
        decision.text = "Preparing check…"
        decision_detail.text = ""
        await check_prompt()

    toggle.on_value_change(refresh_plot)
    search.on_value_change(filter_images)
