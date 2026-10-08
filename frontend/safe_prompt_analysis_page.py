"""NiceGUI dashboard for the DiffusionDB safe-region experiment."""

from __future__ import annotations

import numpy as np
from nicegui import ui

from backend.validation.safe_prompt_analysis import SPLIT_LABELS

COLORS = {
    "safe_test": "#0f766e",
    "toxic_test_i2p": "#d97706",
    "toxic_test_t2i_risky": "#c2410c",
}
METHOD_LABELS = {
    "nearest_anchor": "Nearest anchor",
    "order1": "Order-1 W1",
    "orderinf": "Adaptive order-∞ · 5 task-nearest",
}


def _distribution_figure(analysis: dict, method: str) -> dict:
    distributions = analysis["distributions"][method]
    all_scores = np.concatenate([np.asarray(scores) for scores in distributions.values()])
    radius = analysis["methods"][method]["radius"]
    minimum = min(float(all_scores.min()), radius)
    maximum = max(float(all_scores.max()), radius)
    padding = max((maximum - minimum) * 0.03, abs(radius) * 0.005, 1e-8)
    bins = np.linspace(minimum - padding, maximum + padding, 36)
    centers = ((bins[:-1] + bins[1:]) / 2).tolist()
    traces = []
    for split, scores in distributions.items():
        counts, _ = np.histogram(scores, bins=bins)
        traces.append({
            "type": "scatter", "mode": "lines", "name": SPLIT_LABELS[split],
            "x": centers, "y": (counts / len(scores) * 100).tolist(),
            "line": {"color": COLORS[split], "width": 3},
        })
    return {
        "data": traces,
        "layout": {
            "autosize": True,
            "margin": {"l": 55, "r": 18, "t": 20, "b": 55},
            "legend": {"orientation": "h", "y": 1.15, "x": 0},
            "xaxis": {"title": f"{METHOD_LABELS[method]} score"},
            "yaxis": {"title": "Share of each set (%)", "rangemode": "tozero"},
            "shapes": [{"type": "line", "x0": radius, "x1": radius,
                        "y0": 0, "y1": 1, "yref": "paper",
                        "line": {"color": "#334155", "width": 2, "dash": "dash"}}],
            "annotations": [{"x": radius, "y": 1, "yref": "paper", "text": f"Budget {radius:.4f}",
                             "showarrow": False, "xanchor": "left", "yanchor": "bottom"}],
        },
    }


def _harmfulness_box_figure(rows: list[dict], metric: str, method: str = "nearest_anchor") -> dict:
    if metric not in {"prompt_toxicity", "inappropriate_percentage"}:
        raise ValueError(f"Unsupported I2P score: {metric}")
    groups = ((False, "Accepted", "#0f766e"), (True, "Rejected", "#c2410c"))
    traces = []
    for rejected, name, color in groups:
        values = [row[metric] for row in rows
                  if row["split"] == "toxic_test_i2p"
                  and row.get("rejections", {}).get(method, row["rejected"]) == rejected
                  and row.get(metric) is not None]
        traces.append({
            "type": "box", "name": f"{name} (n={len(values):,})", "y": values,
            "marker": {"color": color}, "line": {"color": color},
            "boxpoints": False, "boxmean": True,
        })
    is_toxicity = metric == "prompt_toxicity"
    return {
        "data": traces,
        "layout": {
            "autosize": True,
            "showlegend": False,
            "margin": {"l": 72, "r": 20, "t": 20, "b": 45},
            "yaxis": {"title": "Prompt toxicity (0–1)" if is_toxicity
                      else "Inappropriate outputs (%)",
                      "range": [0, 1] if is_toxicity else [0, 100]},
        },
    }


def _metric_card(title: str, value: str, detail: str, color: str):
    with ui.card().classes("flex-1 min-w-56 p-4 gap-1 border border-slate-200 shadow-sm"):
        ui.label(title).classes("text-sm font-medium text-slate-600")
        value_label = ui.label(value).classes(f"text-3xl font-bold {color}")
        detail_label = ui.label(detail).classes("text-xs text-slate-500")
    return value_label, detail_label


def create_safe_prompt_analysis_page(analysis: dict) -> None:
    methods = analysis["methods"]
    initial_method = "orderinf"
    splits = methods[initial_method]["splits"]
    safe = splits["safe_test"]
    i2p = splits["toxic_test_i2p"]
    risky = splits["toxic_test_t2i_risky"]
    rows = analysis["rows"]
    page_number = 1
    page_size = 10

    ui.colors(primary="#0f766e", secondary="#17324d", accent="#c2410c")
    ui.add_head_html("<style>body { background: #f4f7f8; } "
                     ".safe-region-card { border: 1px solid #dce7eb; "
                     "box-shadow: 0 8px 28px #17324d0b; }</style>")
    with ui.column().classes("w-full min-h-screen max-w-[1600px] mx-auto p-4 md:p-6 gap-4"):
        with ui.row().classes("w-full items-start justify-between gap-3"):
            with ui.column().classes("gap-1"):
                ui.label("T2I safe-region analysis").classes("text-3xl font-bold text-slate-800")
                ui.label(
                    f"{analysis['safe_count']:,} DiffusionDB safe prompts · "
                    f"{analysis['risky_count']:,} risky evaluation prompts"
                ).classes("text-sm text-slate-600")
            with ui.row().classes("items-center gap-2 flex-wrap"):
                method_select = ui.select(METHOD_LABELS, value=initial_method,
                                          label="Decision method").props("outlined dense").classes("w-72")
                ui.button("COCO baseline", icon="compare_arrows",
                          on_click=lambda: ui.navigate.to("/coco")).props(
                              "outline no-caps color=teal-8"
                          )

        with ui.row().classes("w-full gap-3 flex-wrap"):
            safe_value, safe_detail = _metric_card(
                "Held-out safe coverage", f"{safe['acceptance_rate']:.1%}",
                f"{safe['accepted']:,} / {safe['count']:,} accepted", "text-teal-800",
            )
            i2p_value, i2p_detail = _metric_card(
                "I2P rejection", f"{i2p['rejected'] / i2p['count']:.1%}",
                f"{i2p['rejected']:,} / {i2p['count']:,} rejected", "text-amber-700",
            )
            risky_value, risky_detail = _metric_card(
                "T2I-RiskyPrompt rejection", f"{risky['rejected'] / risky['count']:.1%}",
                f"{risky['rejected']:,} / {risky['count']:,} rejected", "text-orange-700",
            )

        with ui.card().classes("safe-region-card w-full p-4 gap-2"):
            ui.label("Method comparison").classes("text-lg font-semibold text-slate-800")
            ui.label("Each budget is calibrated on the same safe calibration set; rates use held-out prompts.").classes(
                "text-sm text-slate-600"
            )
            comparison_rows = []
            for method, details in methods.items():
                stats = details["splits"]
                comparison_rows.append({
                    "method": METHOD_LABELS[method],
                    "budget": f"{details['radius']:.6g}",
                    "coverage": f"{stats['safe_test']['acceptance_rate']:.1%}",
                    "i2p": f"{stats['toxic_test_i2p']['rejected'] / stats['toxic_test_i2p']['count']:.1%}",
                    "t2i": f"{stats['toxic_test_t2i_risky']['rejected'] / stats['toxic_test_t2i_risky']['count']:.1%}",
                })
            ui.table(columns=[
                {"name": "method", "label": "Decision method", "field": "method", "align": "left"},
                {"name": "budget", "label": "Budget", "field": "budget", "align": "right"},
                {"name": "coverage", "label": "Safe coverage", "field": "coverage", "align": "right"},
                {"name": "i2p", "label": "I2P rejection", "field": "i2p", "align": "right"},
                {"name": "t2i", "label": "T2I rejection", "field": "t2i", "align": "right"},
            ], rows=comparison_rows, pagination=3).props("dense flat hide-bottom").classes("w-full")

        with ui.card().classes("safe-region-card w-full p-4 gap-2"):
            ui.label("Score distributions").classes("text-lg font-semibold text-slate-800")
            ui.label("Prompts to the right of the dashed budget are rejected. Each curve uses its own set size.").classes(
                "text-sm text-slate-600"
            )
            distance_plot = ui.plotly(_distribution_figure(analysis, initial_method)).classes("w-full h-80")

        with ui.card().classes("safe-region-card w-full p-4 gap-2"):
            with ui.row().classes("w-full items-center justify-between gap-3 flex-wrap"):
                ui.label("I2P score by region decision").classes("text-lg font-semibold text-slate-800")
                harmfulness_select = ui.select({
                    "prompt_toxicity": "Prompt toxicity",
                    "inappropriate_percentage": "Inappropriate outputs (%)",
                }, value="prompt_toxicity", label="I2P measure").props("outlined dense").classes("w-64")
            ui.label(
                "Box plots compare I2P prompts accepted and rejected by the selected method. "
                "Prompt toxicity scores the text; inappropriate-output percentage describes generated images. "
                "These measures are not pooled with DiffusionDB NSFW scores or T2I-RiskyPrompt categories."
            ).classes("text-sm text-slate-600")
            harmfulness_plot = ui.plotly(_harmfulness_box_figure(
                rows, "prompt_toxicity", initial_method
            )).classes("w-full h-80")

        with ui.row().classes("w-full gap-3 flex-wrap"):
            with ui.card().classes("safe-region-card flex-1 min-w-80 p-4 gap-2"):
                ui.label("Experiment splits").classes("text-lg font-semibold text-slate-800")
                split_rows = [
                    {"split": "Safe reference", "role": "Fit geometry",
                     "count": f"{analysis['reference_count']:,}"},
                    {"split": "Safe calibration", "role": "Set radius",
                     "count": f"{analysis['calibration_count']:,}"},
                    {"split": "Held-out safe", "role": "Coverage evaluation",
                     "count": f"{safe['count']:,}"},
                    {"split": "I2P", "role": "Risky evaluation",
                     "count": f"{i2p['count']:,}"},
                    {"split": "T2I-RiskyPrompt", "role": "Risky evaluation",
                     "count": f"{risky['count']:,}"},
                ]
                ui.table(columns=[
                    {"name": "split", "label": "Set", "field": "split", "align": "left"},
                    {"name": "role", "label": "Use", "field": "role", "align": "left"},
                    {"name": "count", "label": "Prompts", "field": "count", "align": "right"},
                ], rows=split_rows, pagination=5).props("dense flat hide-bottom").classes("w-full")
            with ui.card().classes("safe-region-card flex-1 min-w-80 p-4 gap-2"):
                ui.label("Method").classes("text-lg font-semibold text-slate-800")
                ui.label("CLIP final-layer EOS · safe-reference PPCA Mahalanobis metric").classes(
                    "text-sm text-slate-700"
                )
                method_note = ui.label(
                    "Order-∞ uses exact token-cloud W1 to select five anchors from a fixed "
                    "60-anchor safe-reference subset; its score is the zero-cut MST bottleneck."
                ).classes("text-sm text-slate-600")
                radius_note = ui.label(
                    f"The region has {analysis['reference_count']:,} safe anchors. "
                    f"Only {analysis['calibration_count']:,} safe calibration prompts set its radius "
                    f"({methods[initial_method]['radius']:.2f}). Held-out safe and risky prompts are evaluation only."
                ).classes("text-sm text-slate-600")
                total_rejected = i2p["rejected"] + risky["rejected"]
                total_risky = i2p["count"] + risky["count"]
                combined_note = ui.label(
                    f"Combined risky rejection: {total_rejected:,}/{total_risky:,} "
                    f"({total_rejected / total_risky:.1%})."
                ).classes("text-sm font-medium text-slate-800")

        with ui.card().classes("safe-region-card w-full p-4 gap-3"):
            with ui.row().classes("w-full items-center justify-between gap-2"):
                ui.label("Held-out prompt decisions").classes("text-lg font-semibold text-slate-800")
                result_count = ui.label().classes("text-sm text-slate-600")
            with ui.row().classes("w-full items-end gap-3 flex-wrap"):
                source_filter = ui.select(
                    {"all": "All sets", **{key: label for key, label in SPLIT_LABELS.items()}},
                    value="all", label="Set",
                ).props("outlined dense").classes("w-48")
                decision_filter = ui.select(
                    {"all": "All decisions", "rejected": "Rejected", "accepted": "Accepted"},
                    value="all", label="Decision",
                ).props("outlined dense").classes("w-44")
                search = ui.input("Search prompt or category").props("outlined dense clearable").classes(
                    "min-w-64 flex-1"
                )
            prompt_table = ui.table(columns=[
                {"name": "row_number", "label": "#", "field": "row_number", "align": "left"},
                {"name": "source", "label": "Set", "field": "source", "align": "left"},
                {"name": "prompt", "label": "Prompt", "field": "prompt", "align": "left"},
                {"name": "category", "label": "Category", "field": "category", "align": "left"},
                {"name": "score", "label": "Score", "field": "score", "align": "right"},
                {"name": "decision", "label": "Decision", "field": "decision", "align": "left"},
            ], rows=[], row_key="row_number").props("dense flat wrap-cells hide-bottom").classes("w-full")
            with ui.row().classes("w-full items-center justify-end gap-2"):
                previous = ui.button("Previous", on_click=lambda: change_page(-1)).props("flat dense no-caps")
                page_label = ui.label().classes("text-sm text-slate-600")
                following = ui.button("Next", on_click=lambda: change_page(1)).props("flat dense no-caps")

    def matching_rows() -> list[dict]:
        query = (search.value or "").strip().casefold()
        selected_source = source_filter.value
        selected_decision = decision_filter.value
        method = method_select.value
        return [row for row in rows
                if (selected_source == "all" or row["split"] == selected_source)
                and (selected_decision == "all" or row["rejections"][method] == (selected_decision == "rejected"))
                and (not query or query in row["prompt"].casefold() or query in row["category"].casefold())]

    def refresh(reset_page: bool = True) -> None:
        nonlocal page_number
        if reset_page:
            page_number = 1
        matching = matching_rows()
        pages = max(1, (len(matching) + page_size - 1) // page_size)
        page_number = min(max(page_number, 1), pages)
        start = (page_number - 1) * page_size
        prompt_table.rows = [
            {"row_number": row["row_number"], "source": row["source"],
             "prompt": row["prompt"], "category": row["category"],
             "score": f"{row['scores'][method_select.value]:.4f}" if method_select.value == "order1"
             else f"{row['scores'][method_select.value]:.2f}",
             "decision": "REJECT" if row["rejections"][method_select.value] else "ACCEPT"}
            for row in matching[start:start + page_size]
        ]
        prompt_table.update()
        result_count.text = f"{len(matching):,} matching prompts"
        page_label.text = f"{page_number} of {pages}"
        previous.enable() if page_number > 1 else previous.disable()
        following.enable() if page_number < pages else following.disable()

    def change_page(offset: int) -> None:
        nonlocal page_number
        page_number += offset
        refresh(reset_page=False)

    def refresh_method() -> None:
        method = method_select.value
        details = methods[method]
        split_stats = details["splits"]
        safe_stats = split_stats["safe_test"]
        i2p_stats = split_stats["toxic_test_i2p"]
        risky_stats = split_stats["toxic_test_t2i_risky"]
        safe_value.text = f"{safe_stats['acceptance_rate']:.1%}"
        safe_detail.text = f"{safe_stats['accepted']:,} / {safe_stats['count']:,} accepted"
        i2p_value.text = f"{i2p_stats['rejected'] / i2p_stats['count']:.1%}"
        i2p_detail.text = f"{i2p_stats['rejected']:,} / {i2p_stats['count']:,} rejected"
        risky_value.text = f"{risky_stats['rejected'] / risky_stats['count']:.1%}"
        risky_detail.text = f"{risky_stats['rejected']:,} / {risky_stats['count']:,} rejected"
        method_note.text = {
            "nearest_anchor": "The score is distance to the closest of all safe reference anchors.",
            "order1": "Order-1 W1 sums Mahalanobis distances to all M safe anchors and divides by M(M+1).",
            "orderinf": "Order-∞ uses exact token-cloud W1 to select five anchors from a fixed "
                        "60-anchor safe-reference subset; its score is the zero-cut MST bottleneck.",
        }[method]
        radius_note.text = (
            f"The region has {analysis['reference_count']:,} safe anchors. "
            f"Only {analysis['calibration_count']:,} safe calibration prompts set this method's "
            f"budget ({details['radius']:.4f}). Held-out safe and risky prompts are evaluation only."
        )
        rejected = i2p_stats["rejected"] + risky_stats["rejected"]
        total = i2p_stats["count"] + risky_stats["count"]
        combined_note.text = f"Combined risky rejection: {rejected:,}/{total:,} ({rejected / total:.1%})."
        distance_plot.update_figure(_distribution_figure(analysis, method))
        harmfulness_plot.update_figure(_harmfulness_box_figure(rows, harmfulness_select.value, method))
        refresh()

    source_filter.on_value_change(lambda _: refresh())
    decision_filter.on_value_change(lambda _: refresh())
    search.on_value_change(lambda _: refresh())
    method_select.on_value_change(lambda _: refresh_method())
    harmfulness_select.on_value_change(
        lambda _: harmfulness_plot.update_figure(_harmfulness_box_figure(
            rows, harmfulness_select.value, method_select.value
        ))
    )
    refresh()
