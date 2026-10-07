"""Compact NiceGUI view for live checks against the COCO caption region."""

import asyncio

from nicegui import ui

from frontend.coco_region_plots import CocoRegionVisualizationController
from frontend.coco_region_validation import LiveCocoRegionValidator, ValidationItem


def create_coco_region_page(
    controller: CocoRegionVisualizationController,
    validator: LiveCocoRegionValidator,
    items: list[ValidationItem],
) -> None:
    selected_item: ValidationItem | None = None
    request_number = 0
    ui.colors(primary="#0f766e", secondary="#17324d", accent="#e2583e")
    ui.add_head_html("<style>html, body, #q-app { height: 100%; overflow: hidden; } "
                     "body { background: #f3f7f8; } .region-card { border: 1px solid #dce7eb; "
                     "box-shadow: 0 8px 28px #17324d0b; }</style>")

    with ui.column().classes("w-full h-[100dvh] min-h-0 overflow-hidden p-3 gap-2"):
        with ui.row().classes("w-full items-center justify-between shrink-0 gap-2"):
            ui.label("COCO prompt region").classes("text-h5 font-bold text-slate-800")
            ui.button("Validation analysis", icon="analytics",
                      on_click=lambda: ui.navigate.to("/analysis")).props("unelevated no-caps color=teal-8")

        with ui.row().classes("w-full flex-1 min-h-0 gap-3 flex-nowrap"):
            with ui.column().classes("w-80 shrink-0 h-full min-h-0 gap-2"):
                with ui.card().classes("region-card w-full p-3 gap-1 shrink-0"):
                    ui.label("Projection").classes("text-subtitle2 font-bold text-slate-800")
                    toggle = ui.toggle(["PCA", "t-SNE"], value="PCA").props(
                        "spread unelevated no-caps color=teal-1 text-color=teal-9 toggle-color=teal-8"
                    ).classes("w-full")
                    projection_note = ui.label(
                        f"3 PCs explain {controller.data.pca_explained_variance_ratio.sum():.1%} of variance"
                    ).classes("text-caption text-slate-500")

                with ui.card().classes("region-card w-full flex-1 min-h-0 p-3 gap-2"):
                    ui.label(f"Validation prompts · {len(items):,}").classes("text-subtitle2 font-bold text-slate-800")
                    search = ui.input("Find prompt or category").props("dense clearable outlined").classes("w-full")
                    with ui.scroll_area().classes("w-full flex-1 min-h-0 border rounded border-slate-200"):
                        prompt_buttons = ui.column().classes("w-full gap-0 p-1")
                    prompt_input = ui.textarea("Prompt").props("outlined dense rows=2").classes("w-full shrink-0")
                    with ui.row().classes("w-full items-center justify-between gap-1 shrink-0"):
                        prompt_source = ui.label("Select a prompt").classes("text-caption text-slate-500 truncate flex-1")
                        ui.button("Check", on_click=lambda: check_prompt()).props("unelevated dense no-caps color=teal-8")

                with ui.card().classes("region-card w-full p-3 gap-1 shrink-0"):
                    decision = ui.label("Select a prompt to check").classes("text-subtitle1 font-bold text-slate-700")
                    decision_detail = ui.label("").classes("text-caption text-slate-600")
                    order1_decision = ui.label("Order-1 W1 · awaiting prompt").classes("text-caption text-slate-600")
                    orderinf_decision = ui.label("Adaptive order-∞ · awaiting prompt").classes("text-caption text-slate-600")

            with ui.card().classes("region-card flex-1 min-w-0 h-full min-h-0 p-1 overflow-hidden"):
                plot = ui.plotly(controller.build_figure()).classes("w-full h-full")

    def refresh_plot() -> None:
        controller.set_projection_method(toggle.value)
        projection_note.text = (
            f"3 PCs explain {controller.data.pca_explained_variance_ratio.sum():.1%} of variance"
            if toggle.value == "PCA" else "New prompts appear at their nearest COCO anchor"
        )
        plot.update_figure(controller.build_figure())

    def filter_prompts() -> None:
        query = (search.value or "").casefold().strip()
        matches = [item for item in items if query in item.prompt.casefold() or query in item.categories.casefold()]
        prompt_buttons.clear()
        with prompt_buttons:
            for item in matches[:80]:
                ui.button(f"{item.row_number}: {item.prompt[:68]}",
                          on_click=lambda _, chosen=item: select_item(chosen)).props(
                              "flat dense no-caps align=left"
                          ).classes("w-full text-left text-slate-700")
            if len(matches) > 80:
                ui.label(f"First 80 of {len(matches):,} matches · refine search").classes("text-caption text-slate-500")

    async def check_prompt() -> None:
        nonlocal request_number
        prompt = (prompt_input.value or "").strip()
        if not prompt:
            ui.notify("Enter or select a prompt.", type="warning")
            return
        request_number += 1
        current_request = request_number
        prompt_source.text = (
            f"CSV row {selected_item.row_number} · {selected_item.categories}"
            if selected_item is not None and prompt == selected_item.prompt else "Edited prompt"
        )
        decision.text = "Checking prompt…"
        decision_detail.text = ""
        order1_decision.text = "Order-1 W1 · checking…"
        orderinf_decision.text = "Adaptive order-∞ · checking…"
        try:
            result = await asyncio.to_thread(validator.validate, prompt)
        except Exception as error:
            if current_request == request_number:
                decision.text = "Check failed"
                decision_detail.text = str(error)
                order1_decision.text = "Order-1 W1 · unavailable"
                orderinf_decision.text = "Adaptive order-∞ · unavailable"
            return
        if current_request != request_number:
            return
        controller.set_validation(result)
        decision.text = "BLOCK" if result.blocked else "ALLOW"
        decision.classes(remove="text-slate-700 text-red-700 text-teal-700",
                         add="text-red-700" if result.blocked else "text-teal-700")
        decision_detail.text = f"Nearest anchor: {result.score:.2f} / {result.radius:.2f}"
        order1_decision.text = (
            f"Order-1 W1: {'BLOCK' if result.order1_blocked else 'ALLOW'} · "
            f"{result.order1_score:.4f} / {result.order1_radius:.4f}"
            if result.order1_score is not None else "Order-1 W1: unavailable"
        )
        orderinf_decision.text = (
            f"Adaptive order-∞: {'BLOCK' if result.orderinf_blocked else 'ALLOW'} · "
            f"{result.orderinf_score:.2f} / {result.orderinf_radius:.2f}"
            if result.orderinf_score is not None else "Adaptive order-∞: unavailable"
        )
        plot.update_figure(controller.build_figure())

    async def select_item(item: ValidationItem) -> None:
        nonlocal selected_item, request_number
        selected_item = item
        request_number += 1
        controller.clear_validation()
        plot.update_figure(controller.build_figure())
        prompt_input.value = item.prompt
        prompt_source.text = f"CSV row {item.row_number} · {item.categories}"
        await check_prompt()

    toggle.on_value_change(refresh_plot)
    search.on_value_change(filter_prompts)
    filter_prompts()
