"""Single-screen analysis of all validation prompts."""

import asyncio
from collections.abc import Callable

import numpy as np
from nicegui import ui


def _toxicity_density(values: list[float]) -> tuple[list[float], list[float]]:
    """Estimate a smooth toxicity distribution on the score's 0–1 scale."""
    if not values:
        return [], []
    samples = np.asarray(values, dtype=np.float64)
    x = np.linspace(0.0, 1.0, 101)
    bandwidth = float(np.clip(1.06 * max(float(samples.std()), 0.02) * len(samples) ** -0.2,
                              0.025, 0.12))
    # Reflect at the score limits so density does not disappear at 0 or 1.
    density = np.zeros_like(x)
    for reflected in (samples, -samples, 2 - samples):
        offsets = (x[:, None] - reflected[None, :]) / bandwidth
        density += np.exp(-0.5 * offsets ** 2).mean(axis=1)
    density /= bandwidth * np.sqrt(2 * np.pi)
    return x.tolist(), density.tolist()


def create_coco_region_analysis_page(
    load_report: Callable[[], dict], *, initial_report: dict | None = None,
    back_path: str | None = None,
) -> None:
    report = initial_report
    loading = False
    page_number = 1
    visible_rows: list[dict] = []
    ui.colors(primary="#0f766e", secondary="#17324d", accent="#e2583e")
    ui.add_head_html("<style>html, body, #q-app { height: 100%; overflow: hidden; } "
                     "body { background: #f3f7f8; } .region-card { border: 1px solid #dce7eb; "
                     "box-shadow: 0 8px 28px #17324d0b; }</style>")

    with ui.column().classes("w-full h-[100dvh] min-h-0 overflow-hidden p-3 gap-2"):
        with ui.row().classes("w-full items-center justify-between shrink-0"):
            with ui.row().classes("items-center gap-3"):
                ui.label("COCO baseline analysis").classes("text-h5 font-bold text-slate-800")
                if back_path:
                    ui.button("T2I safe region", icon="arrow_back",
                              on_click=lambda: ui.navigate.to(back_path)).props("flat no-caps")
            analyze_button = ui.button("Recompute analysis" if report else "Analyze prompts", icon="analytics",
                                       on_click=lambda: load()).props("unelevated no-caps color=teal-8")
        status = ui.label(
            f"Saved results · {len(report['rows']):,} CSV rows" if report else
            "No current saved results. Select Analyze prompts to calculate them."
        ).classes("text-caption text-slate-600 shrink-0")
        with ui.row().classes("w-full items-center shrink-0 gap-3"):
            metric_select = ui.select({
                "nearest_anchor": "Nearest anchor", "order1": "Order-1 W1", "orderinf": "Adaptive order-∞",
            }, value="nearest_anchor", label="Decision method").props("outlined dense").classes("w-56")
            entry_filter = ui.toggle(["Blocked", "All"], value="Blocked").props("no-caps")

        with ui.card().classes("region-card w-full h-[28%] min-h-0 p-2 overflow-hidden"):
            with ui.row().classes("w-full h-full min-h-0 gap-3 flex-nowrap"):
                with ui.column().classes("w-72 shrink-0 gap-1 justify-center"):
                    ui.label("Prompt toxicity · blocked vs allowed").classes("text-subtitle2 font-bold text-slate-800")
                    blocked_toxicity = ui.label("Blocked · awaiting analysis").classes("text-body2 text-red-700")
                    allowed_toxicity = ui.label("Allowed · awaiting analysis").classes("text-body2 text-teal-700")
                    toxicity_difference = ui.label("").classes("text-body2 font-semibold text-slate-800")
                    ui.label("Scores come from the CSV prompt_toxicity column.").classes("text-caption text-slate-500")
                toxicity_plot = ui.plotly({"data": [], "layout": {"autosize": True}}).classes("flex-1 min-w-0 h-full")

        with ui.row().classes("w-full flex-1 min-h-0 gap-3 flex-nowrap"):
            with ui.column().classes("w-[42%] h-full min-h-0 gap-2"):
                with ui.card().classes("region-card w-full p-1 h-[43%] min-h-0 overflow-hidden"):
                    distribution_summary = ui.label("").classes("text-caption text-slate-600")
                    distribution = ui.plotly({"data": [], "layout": {"autosize": True}}).classes("w-full flex-1 min-h-0")
                with ui.card().classes("region-card w-full flex-1 min-h-0 p-2 overflow-hidden"):
                    ui.label("By category").classes("text-subtitle2 font-semibold text-slate-800")
                    categories_table = ui.table(columns=[
                        {"name": "category", "label": "Category", "field": "category", "align": "left"},
                        {"name": "count", "label": "Prompts", "field": "count", "align": "right"},
                        {"name": "blocked", "label": "Blocked", "field": "blocked", "align": "right"},
                        {"name": "rate", "label": "Rate", "field": "rate", "align": "right"},
                        {"name": "median", "label": "Median", "field": "median", "align": "right"},
                    ], rows=[], pagination=4).props("dense flat").classes("w-full flex-1 min-h-0")
            with ui.card().classes("region-card flex-1 min-w-0 h-full min-h-0 p-2 overflow-hidden"):
                ui.label("Prompt decisions").classes("text-subtitle2 font-semibold text-slate-800")
                entries_table = ui.table(columns=[
                    {"name": "row_number", "label": "Row", "field": "row_number", "align": "left"},
                    {"name": "prompt", "label": "Prompt", "field": "prompt", "align": "left"},
                    {"name": "category", "label": "Category", "field": "category", "align": "left"},
                    {"name": "prompt_toxicity", "label": "Toxicity", "field": "prompt_toxicity", "align": "right"},
                    {"name": "cost", "label": "Cost", "field": "cost", "align": "right"},
                    {"name": "decision", "label": "Decision", "field": "decision", "align": "left"},
                ], rows=[], row_key="row_number").props("dense flat").classes("w-full flex-1 min-h-0")
                with ui.row().classes("w-full items-center justify-between shrink-0"):
                    previous_button = ui.button("Previous", on_click=lambda: change_page(-1)).props("flat dense no-caps")
                    page_label = ui.label("0 of 0").classes("text-caption text-slate-600")
                    next_button = ui.button("Next", on_click=lambda: change_page(1)).props("flat dense no-caps")

    def refresh_table() -> None:
        nonlocal page_number
        total = len(visible_rows)
        page_count = max(1, (total + 8) // 9)
        page_number = min(max(page_number, 1), page_count)
        start = (page_number - 1) * 9
        key = metric_select.value
        metric = report["metrics"][key]
        score_key, blocked_key = metric["score_key"], metric["blocked_key"]
        precision = 6 if key == "order1" else 2
        entries_table.rows = [
            {"row_number": row["row_number"], "prompt": row["prompt"], "category": row["category"],
             "prompt_toxicity": f"{row['prompt_toxicity']:.3f}",
             "cost": f"{row[score_key]:.{precision}f}",
             "decision": "BLOCK" if row[blocked_key] else "ALLOW"}
            for row in visible_rows[start:start + 9]
        ]
        entries_table.update()
        page_label.text = f"{start + 1}–{min(start + 9, total)} of {total:,}" if total else "0 of 0"
        previous_button.enable() if page_number > 1 else previous_button.disable()
        next_button.enable() if page_number < page_count else next_button.disable()

    def change_page(offset: int) -> None:
        nonlocal page_number
        page_number += offset
        refresh_table()

    def refresh() -> None:
        nonlocal page_number, visible_rows
        if report is None:
            return
        key = metric_select.value
        if key not in report["metrics"]:
            return
        metric = report["metrics"][key]
        blocked_key = metric["blocked_key"]
        precision = 6 if key == "order1" else 2
        distribution_summary.text = "Prompt toxicity distribution · blocked vs allowed"
        rows = report["rows"]
        toxicity = metric["toxicity_comparison"]
        blocked_toxicity.text = (
            f"Blocked mean {toxicity['blocked']['mean']:.3f} · median {toxicity['blocked']['median']:.3f} "
            f"(n={toxicity['blocked']['count']:,})"
            if toxicity["blocked"]["count"] else "Blocked · no prompts"
        )
        allowed_toxicity.text = (
            f"Allowed mean {toxicity['allowed']['mean']:.3f} · median {toxicity['allowed']['median']:.3f} "
            f"(n={toxicity['allowed']['count']:,})"
            if toxicity["allowed"]["count"] else "Allowed · no prompts"
        )
        toxicity_difference.text = (
            f"Mean difference (blocked − allowed): {toxicity['mean_difference']:+.3f}"
            if toxicity["mean_difference"] is not None else "Mean difference unavailable"
        )
        toxicity_plot.update_figure({
            "data": [
                {"type": "box", "name": "Blocked", "y": [row["prompt_toxicity"] for row in rows if row[blocked_key]],
                 "marker": {"color": "#e2583e"}, "boxpoints": False, "boxmean": True},
                {"type": "box", "name": "Allowed", "y": [row["prompt_toxicity"] for row in rows if not row[blocked_key]],
                 "marker": {"color": "#0d9488"}, "boxpoints": False, "boxmean": True},
            ],
            "layout": {"autosize": True, "showlegend": False,
                       "margin": {"l": 48, "r": 12, "t": 10, "b": 30},
                       "yaxis": {"title": "Prompt toxicity", "range": [0, 1]}},
        })
        density_traces = []
        for blocked, name, color in ((False, "Allowed", "#0d9488"), (True, "Blocked", "#e2583e")):
            x, y = _toxicity_density([row["prompt_toxicity"] for row in rows if row[blocked_key] == blocked])
            if x:
                density_traces.append({"type": "scatter", "mode": "lines", "name": name,
                                       "x": x, "y": y, "line": {"color": color, "width": 2.5}})
        distribution.update_figure({
            "data": density_traces,
            "layout": {"autosize": True,
                       "legend": {"orientation": "h", "y": 1.2, "x": 0},
                       "margin": {"l": 48, "r": 12, "t": 20, "b": 38},
                       "xaxis": {"title": "Prompt toxicity", "range": [0, 1]},
                       "yaxis": {"title": "Density", "rangemode": "tozero"}},
        })
        categories_table.rows = [
            {"category": row["category"], "count": row["count"], "blocked": row["blocked"],
             "rate": f"{row['block_rate']:.1%}", "median": f"{row['distance_median']:.{precision}f}"}
            for row in metric["by_category"]
        ]
        categories_table.update()
        visible_rows = [row for row in rows if entry_filter.value == "All" or row[blocked_key]]
        page_number = 1
        refresh_table()

    async def load() -> None:
        nonlocal report, loading
        if loading:
            return
        loading = True
        analyze_button.disable()
        status.text = "Analyzing validation prompts…"
        try:
            report = await asyncio.to_thread(load_report)
            status.text = f"{len(report['rows']):,} CSV rows · {report['unique_prompt_level']['count']:,} distinct prompts"
            analyze_button.text = "Recompute analysis"
            metric_select.set_options({key: label for key, label in metric_select.options.items()
                                       if key in report["metrics"]})
            refresh()
        except Exception as error:
            status.text = f"Analysis failed: {error}"
        finally:
            loading = False
            analyze_button.enable()

    metric_select.on_value_change(refresh)
    entry_filter.on_value_change(refresh)
    if report is not None:
        refresh()
    else:
        previous_button.disable()
        next_button.disable()
