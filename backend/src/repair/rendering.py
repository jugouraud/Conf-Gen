"""Repair statistics and presentation artifacts."""

import html
import io
import json
from collections import defaultdict

import numpy as np
from matplotlib.figure import Figure

from backend.src.inference.policies import SHIELDGEMMA_IMAGE_POLICIES
from backend.src.repair.definitions import REPAIR_METHOD_LABELS
from backend.src.repair.steering import SteeringSchedule


def render_scored_repair_gallery(pairs, summaries, image_prefix, threshold):
    """Render every scored image with its method, policy scores, and paired change."""
    escape = html.escape
    groups = defaultdict(list)
    for row in pairs:
        groups[
            (row["category"], row["split"], row["id"], row["mode"], row["seed"])
        ].append(row)
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Repair techniques and image toxicity</title>",
        "<style>body{font:16px/1.5 system-ui,sans-serif;color:#172638;background:#f1f5f9;margin:0}"
        "header,main{max-width:1600px;margin:auto;padding:24px}header{background:white}"
        "a{color:#075aab}section{background:white;padding:22px;margin:24px 0;border-radius:12px}"
        ".images{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:16px}"
        "figure{margin:0;border:1px solid #cad5e2;border-radius:8px;overflow:hidden}"
        "img{width:100%;height:auto;display:block}figcaption{padding:14px}h3{margin:0 0 8px}"
        "dl{display:grid;grid-template-columns:1fr 1fr;margin:10px 0}dd{margin:0;text-align:right}"
        ".score{font-size:22px;font-weight:700}.high{color:#a12622}.low{color:#17623a}"
        ".meta{color:#526174;overflow-wrap:anywhere}.prompt{white-space:pre-wrap}"
        "table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}"
        ".scroll{overflow:auto}nav{display:flex;gap:18px;flex-wrap:wrap}small{overflow-wrap:anywhere}"
        "@media print{section,figure{break-inside:avoid}header,main{padding:0}}</style>",
        "<header><h1>Repair techniques and image toxicity</h1>",
        f"<p>{len(groups)} prompt/mode/seed comparisons · {len(pairs)} repair pairs · threshold {threshold:.2f}</p>",
        "<p>Each group uses the same prompt and initial noise. Toxicity is the maximum of the sexual, "
        "dangerous, and violence policy violation probabilities (0–1). Lower scores indicate lower "
        "model-rated toxicity; they do not measure image quality or semantic preservation.</p>",
        '<p><a href="pairs.csv">All image scores (CSV)</a> · <a href="comparison.md">Summary</a></p><nav>',
    ]
    categories = sorted({key[0] for key in groups})
    for index, category in enumerate(categories):
        parts.append(f'<a href="#category-{index}">{escape(category)}</a>')
    parts.append(
        '</nav></header><main><section><h2>Technique comparison</h2><div class="scroll"><table>'
        "<tr><th>Technique</th><th>Mode</th><th>Pairs</th><th>Original toxicity</th>"
        "<th>Repaired toxicity</th><th>Reduction</th></tr>"
    )
    for row in summaries:
        if row["split"] == "all":
            parts.append(
                f"<tr><td>{escape(row['method'])} · {escape(REPAIR_METHOD_LABELS[row['method']])}</td>"
                f"<td>{escape(row['mode'].replace('_', ' '))}</td><td>{row['pairs']}</td>"
                f"<td>{row['mean_original_toxicity']:.4f}</td><td>{row['mean_repaired_toxicity']:.4f}</td>"
                f"<td>{row['mean_toxicity_reduction']:+.4f}</td></tr>"
            )
    parts.append("</table></div></section>")
    for category_index, category in enumerate(categories):
        parts.append(f'<h2 id="category-{category_index}">{escape(category)}</h2>')
        for key, rows in sorted(groups.items()):
            if key[0] != category:
                continue
            _, split, prompt_id, mode, seed = key
            rows = sorted(
                rows,
                key=lambda row: (row["method"], row["variant"], str(row["budget"])),
            )
            parts.append(
                f"<section><h2>{escape(mode.replace('_', ' '))} · seed {seed}</h2>"
                f'<p class="prompt">{escape(rows[0]["prompt"])}</p>'
                f'<p class="meta">{escape(split)} · prompt {escape(prompt_id)}</p><div class="images">'
            )
            cards = [("R0", "Original baseline", "original", rows[0])]
            cards.extend(
                (row["method"], REPAIR_METHOD_LABELS[row["method"]], "repaired", row)
                for row in rows
            )
            for method, label, role, row in cards:
                toxicity = row[f"{role}_toxicity"]
                css = "high" if toxicity >= threshold else "low"
                src = escape(image_prefix + row[f"{role}_image"], quote=True)
                parts.append(
                    f'<figure><a href="{src}"><img loading="lazy" src="{src}" '
                    f'alt="{escape(method)} · {escape(label)} · toxicity {toxicity:.4f}"></a>'
                    f"<figcaption><h3>{escape(method)} · {escape(label)}</h3>"
                    f'<div class="score {css}">Toxicity {toxicity:.4f}</div><dl>'
                )
                for policy in SHIELDGEMMA_IMAGE_POLICIES:
                    parts.append(
                        f"<dt>{policy.title()}</dt><dd>{row[f'{role}_{policy}']:.4f}</dd>"
                    )
                parts.append("</dl>")
                if role == "repaired":
                    parts.append(
                        f"<p>Change from original: <strong>{row['toxicity_delta']:+.4f}</strong></p>"
                        f'<p class="meta">Variant: {escape(row["variant"])}<br>'
                        f"Budget: {escape(str(row['budget']))}<br>"
                        f"Repair status: {escape(str(row['search_status']))}<br>"
                        f"Identical image bytes: {row['identical_image_bytes']}</p>"
                    )
                parts.append(
                    f"<small>{escape(row[f'{role}_image'])}</small></figcaption></figure>"
                )
            parts.append("</div></section>")
    parts.append("</main></html>")
    return "".join(parts)


def build_manual_review_pairs(records):
    baselines = {
        (row["split"], row["id"], row["mode"], row["seed"]): row
        for row in records
        if row["method"] == "R0"
    }
    pairs = []
    for row in records:
        if not (
            row["flagged"]
            and row["split"].startswith("unsafe")
            and row["method"] != "R0"
            and row["budget"] == "minimum"
        ):
            continue
        baseline = baselines[(row["split"], row["id"], row["mode"], row["seed"])]
        pairs.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "category": row["category"],
                "mode": row["mode"],
                "method": row["method"],
                "variant": row["variant"],
                "selected_schedule_variant": row.get(
                    "selected_schedule_variant", row["variant"]
                ),
                "seed": row["seed"],
                "anchor_id": row["anchor_id"],
                "original_image": baseline["image"],
                "corrected_image": row["image"],
                "repair_status": row["search_status"],
                "frozen_probe_admissible": row["frozen_probe"]["admissible"],
                "safe_score": row["frozen_probe"]["safe_score"],
                "unsafe_score": row["frozen_probe"]["unsafe_score"],
                "schedule": json.dumps(row["schedule"]),
                "manual_image_assessment": "",
            }
        )
    return pairs


def render_manual_review_html(pairs):
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Manual harmful-intent review</title>',
        "<style>body{font:16px sans-serif;margin:2em}section{margin:2em 0}article{display:flex;gap:1em}figure{margin:0}img{width:min(40vw,512px)}pre{white-space:pre-wrap}</style>",
        "<h1>Original and repaired generations</h1>",
        (
            "<p>Flagged harmful intents only. Images share their initial noise. No automated image evaluator was used. "
            "Probe admission is separate from image safety. Unresolved repairs retain the original output.</p>"
        ),
    ]
    for pair in pairs:
        parts.extend(
            [
                f"<section><h2>{html.escape(pair['category'])} · {html.escape(pair['variant'])}</h2>",
                f"<pre>{html.escape(pair['prompt'])}</pre>",
                (
                    f"<p>{html.escape(pair['mode'])} · seed {pair['seed']} · "
                    f"{html.escape(pair['repair_status'])} · field admissible={pair['frozen_probe_admissible']}</p><article>"
                ),
            ]
        )
        for label, key in (
            ("Original, unmodified", "original_image"),
            ("Repaired output", "corrected_image"),
        ):
            parts.append(
                f'<figure><figcaption>{label}</figcaption><img loading="lazy" '
                f'src="../{html.escape(pair[key], quote=True)}"></figure>'
            )
        parts.append("</article></section>")
    return "".join(parts)


def calculate_prompt_clustered_interval(rows, value, seed=1729):
    """Bootstrap prompts, retaining all seeds for each sampled prompt."""
    clusters = defaultdict(list)
    for row in rows:
        result = value(row)
        if result is not None:
            clusters[(row["split"], row["id"])].append(float(result))
    if not clusters:
        return {"mean": None, "ci95": None, "prompts": 0}
    totals = np.array([sum(values) for values in clusters.values()])
    counts = np.array([len(values) for values in clusters.values()])
    mean = float(totals.sum() / counts.sum())
    if len(clusters) < 2:
        return {"mean": mean, "ci95": None, "prompts": len(clusters)}
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(clusters), (1000, len(clusters)))
    sampled = totals[indexes].sum(axis=1) / counts[indexes].sum(axis=1)
    return {
        "mean": mean,
        "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "prompts": len(clusters),
    }


def summarize_repair_comparison(records):
    baselines = {
        (r["split"], r["id"], r["mode"], r["seed"]): r
        for r in records
        if r["method"] == "R0"
    }
    groups = defaultdict(list)
    for row in records:
        groups[
            (row["mode"], row["variant"], row["budget"], row["split"], row["category"])
        ].append(row)
        groups[
            (row["mode"], row["variant"], row["budget"], row["split"], "all")
        ].append(row)
    summaries = []
    for (mode, variant, budget, split, category), rows in sorted(groups.items()):
        flagged = [row for row in rows if row["flagged"]]
        unsafe_baselines = []
        for row in rows:
            baseline = baselines[(row["split"], row["id"], row["mode"], row["seed"])]
            if baseline["image_evaluation"].get("unsafe") is True:
                unsafe_baselines.append(row)
        metrics = {
            "field_repair_rate": calculate_prompt_clustered_interval(
                flagged, lambda r: r["frozen_probe"]["admissible"]
            ),
            "safe_admission": calculate_prompt_clustered_interval(
                rows, lambda r: r["frozen_probe"]["safe_admission"]
            ),
            "unsafe_exclusion": calculate_prompt_clustered_interval(
                rows, lambda r: r["frozen_probe"]["unsafe_exclusion"]
            ),
            "unsafe_image_rate": calculate_prompt_clustered_interval(
                rows, lambda r: r["image_evaluation"].get("unsafe")
            ),
            "paired_image_repair_rate": calculate_prompt_clustered_interval(
                unsafe_baselines,
                lambda r: (
                    None
                    if r["image_evaluation"].get("unsafe") is None
                    else not r["image_evaluation"]["unsafe"]
                ),
            ),
            "semantic_preservation": calculate_prompt_clustered_interval(
                rows, lambda r: r["image_evaluation"].get("semantic_preservation")
            ),
            "quality": calculate_prompt_clustered_interval(
                rows, lambda r: r["image_evaluation"].get("quality")
            ),
            "artifacts": calculate_prompt_clustered_interval(
                rows, lambda r: r["image_evaluation"].get("artifacts")
            ),
            "integrated_strength": calculate_prompt_clustered_interval(
                rows, lambda r: r["schedule"]["integrated_strength"]
            ),
            "mean_strength": calculate_prompt_clustered_interval(
                rows, lambda r: r["generation"]["mean_strength"]
            ),
            "modified_step_fraction": calculate_prompt_clustered_interval(
                rows, lambda r: r["generation"]["modified_step_fraction"]
            ),
            "field_perturbation": calculate_prompt_clustered_interval(
                rows, lambda r: r["generation"]["integrated_field_perturbation"]
            ),
            "inference_seconds": calculate_prompt_clustered_interval(
                rows, lambda r: r["generation"]["inference_seconds"]
            ),
            "model_evaluations": calculate_prompt_clustered_interval(
                rows, lambda r: r["generation"]["model_evaluations"]
            ),
            "diagnostic_model_evaluations": calculate_prompt_clustered_interval(
                rows, lambda r: r["generation"]["diagnostic_model_evaluations"]
            ),
            "query_probe_seconds": calculate_prompt_clustered_interval(
                rows, lambda r: r["search"]["query_probe"]["seconds"]
            ),
            "query_probe_model_evaluations": calculate_prompt_clustered_interval(
                rows, lambda r: r["search"]["query_probe"]["model_evaluations"]
            ),
            "verification_model_evaluations": calculate_prompt_clustered_interval(
                rows,
                lambda r: r["frozen_probe"]["direct_verification_model_evaluations"],
            ),
            "search_seconds": calculate_prompt_clustered_interval(
                rows, lambda r: r["search"].get("seconds", 0.0)
            ),
            "search_evaluations": calculate_prompt_clustered_interval(
                rows, lambda r: r["search_evaluations"]
            ),
            "direct_verification_seconds": calculate_prompt_clustered_interval(
                rows, lambda r: r["generation"].get("direct_verification_seconds")
            ),
            "peak_cuda_bytes": calculate_prompt_clustered_interval(
                rows, lambda r: r["generation"].get("peak_cuda_bytes")
            ),
            "pixel_rmse": calculate_prompt_clustered_interval(
                rows, lambda r: r["image_similarity"]["pixel_rmse"]
            ),
        }
        summaries.append(
            {
                "mode": mode,
                "method": variant.split("_")[0],
                "variant": variant,
                "budget": budget,
                "split": split,
                "category": category,
                "generation_records": len(rows),
                "flagged_prompts": len({r["id"] for r in flagged}),
                "unresolved_prompts": len(
                    {r["id"] for r in flagged if not r["frozen_probe"]["admissible"]}
                ),
                "unnecessary_intervention_rate": float(
                    np.mean([r["schedule"]["integrated_strength"] > 0 for r in rows])
                )
                if split == "additional_safe_test"
                else None,
                **metrics,
            }
        )
    return summaries


def render_svg_curves(title, curves, xlabel="Flow time", ylabel="Value"):
    """Standalone exportable scientific plot without a GUI backend."""
    figure = Figure(figsize=(9, 4), layout="constrained")
    axis = figure.subplots()
    for label, xs, ys in curves:
        axis.plot(xs, ys, label=label, marker="." if len(xs) == 1 else None)
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.grid(alpha=0.2)
    if curves:
        axis.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    stream = io.StringIO()
    figure.savefig(stream, format="svg")
    return stream.getvalue()


def build_repair_figure_artifacts(records, summaries):

    artifacts = {}
    representatives = {}
    for row in records:
        representatives.setdefault(row["category"], row["id"])
    for category, prompt_id in representatives.items():
        rows = [
            r
            for r in records
            if r["id"] == prompt_id
            and r["budget"] == "minimum"
            and r["mode"] == records[0]["mode"]
            and r["seed"] == records[0]["seed"]
        ]
        curves = []
        for row in rows:
            schedule = row["schedule"]
            sampler = SteeringSchedule(
                schedule["name"], tuple(schedule["times"]), tuple(schedule["strengths"])
            )
            times = np.linspace(0, 1, 201)
            curves.append((row["variant"], times, [sampler(t) for t in times]))
        artifacts[f"schedules_{category}.svg"] = render_svg_curves(
            f"Steering schedules: {category}", curves, ylabel="Lambda"
        )
    profiles = []
    for split in (
        "unsafe_test",
        "additional_safe_test",
        "safe_test",
        "unsafe_validation",
        "safe_validation",
    ):
        unique = {
            r["id"]: r
            for r in records
            if r["split"] == split and r["deviation_profiles"]
        }
        if unique:
            times = next(iter(unique.values()))["probe_times"]
            for statistic in ("max", "top10", "top25"):
                values = np.mean(
                    [r["deviation_profiles"][statistic] for r in unique.values()],
                    axis=0,
                )
                profiles.append((f"{split} {statistic}", times, values))
    artifacts["deviations.svg"] = render_svg_curves(
        "Normalized temporal deviations", profiles, ylabel="Safe-normalized deviation"
    )
    points = [
        r
        for r in summaries
        if r["category"] == "all"
        and r["split"].startswith("unsafe")
        and r["unsafe_image_rate"]["mean"] is not None
    ]
    tradeoff = [
        (
            r["variant"],
            [r["semantic_preservation"]["mean"]],
            [r["unsafe_image_rate"]["mean"]],
        )
        for r in points
        if r["semantic_preservation"]["mean"] is not None
    ]
    if tradeoff:
        artifacts["safety_preservation.json"] = json.dumps(tradeoff)
        artifacts["safety_preservation.svg"] = render_svg_curves(
            "Generated-image safety vs preservation",
            tradeoff,
            xlabel="Semantic preservation",
            ylabel="Unsafe image rate",
        )
    efficiency = [
        (
            r["variant"],
            [r["integrated_strength"]["mean"]],
            [r["field_repair_rate"]["mean"]],
        )
        for r in summaries
        if r["category"] == "all" and r["field_repair_rate"]["mean"] is not None
    ]
    artifacts["field_intervention_efficiency.svg"] = render_svg_curves(
        "Frozen-probe repair efficiency",
        efficiency,
        xlabel="Integrated steering",
        ylabel="Frozen-probe repair rate",
    )
    image_efficiency = [
        (
            r["variant"],
            [r["integrated_strength"]["mean"]],
            [1 - r["unsafe_image_rate"]["mean"]],
        )
        for r in points
    ]
    if image_efficiency:
        artifacts["image_intervention_efficiency.svg"] = render_svg_curves(
            "Generated-image intervention efficiency",
            image_efficiency,
            xlabel="Integrated steering",
            ylabel="Safe image rate",
        )
    artifacts["intervention_efficiency.json"] = json.dumps(
        [
            {
                "variant": r["variant"],
                "mode": r["mode"],
                "budget": r["budget"],
                "split": r["split"],
                "integrated_strength": r["integrated_strength"]["mean"],
                "frozen_probe_repair_rate": r["field_repair_rate"]["mean"],
                "unsafe_image_rate": r["unsafe_image_rate"]["mean"],
            }
            for r in summaries
            if r["category"] == "all"
        ]
    )
    return artifacts


def build_repair_image_gallery(records):
    groups = defaultdict(list)
    for row in records:
        if row["budget"] == "minimum":
            groups[(row["split"], row["id"], row["mode"], row["seed"])].append(row)
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Paired repairs</title>',
        "<style>body{font:15px sans-serif;margin:2em}section{display:flex;overflow:auto;gap:1em}figure{margin:0}img{width:200px}figcaption{max-width:200px}</style>",
        "<h1>Paired generations</h1><p>Identical initial noise within each group. Field admission is separate from image safety.</p>",
    ]
    for (split, prompt_id, mode, seed), rows in groups.items():
        parts.append(
            f"<h2>{html.escape(split)} / {html.escape(prompt_id)} / {html.escape(mode)} / seed {seed}</h2>"
            f"<p>{html.escape(rows[0]['prompt'])}</p><section>"
        )
        for row in rows:
            parts.append(
                f'<figure><img loading="lazy" src="../{html.escape(row["image"], quote=True)}"><figcaption>'
                f"<strong>{html.escape(row['method'])} · {html.escape(REPAIR_METHOD_LABELS[row['method']])}</strong><br>"
                f"{html.escape(row['variant'])}; field admissible={row['frozen_probe']['admissible']}; "
                f"image unsafe={row['image_evaluation'].get('unsafe')}; toxicity: unmeasured (see scored report)</figcaption></figure>"
            )
        parts.append("</section>")
    return "".join(parts)
