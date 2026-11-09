"""Draw schematic R1--R5 definitions using the implemented schedule functions.

Heights, adaptive centers, and R5 values are illustrative. No results are read
and no detector feasibility is claimed. Run with the research Python environment.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

DESTINATION = Path(__file__).resolve().parent
KAPPA = 0.65  # Illustration only; the repair search chooses this per prompt.
WIDTH = 0.25  # One configured candidate width, used for illustration.
PEAK_TIME = 0.4375  # Illustrative adaptive center on the probe time grid.
R4_CENTERS = (0.1875, 0.8125)
METHODS = {
    "R1": ("Constant steering", "#2454A6"),
    "R2": ("Fixed temporal window", "#008879"),
    "R3": ("Peak-deviation window", "#B4780B"),
    "R4": ("Contiguous-deviation windows", "#8851AD"),
    "R5": ("Optimized smooth schedule", "#C64252"),
}
DESCRIPTIONS = {
    "R1": "Strength: one searched value $\\kappa$\nDuration: the full interval [0, 1]",
    "R2": "Strength: searched $\\kappa$; width: chosen from preset $w$\nPosition: early, middle, or late",
    "R3": "Strength: searched $\\kappa$; width: chosen from preset $w$\nPosition: largest normalized probe deviation",
    "R4": "Strength: one searched $\\kappa$ for the combined shape\nUp to two selected centers; each window has width $w$",
    "R5": "Strength: nine optimized knot values $\\theta_k$\nActive intervals follow the optimized knot values",
}


def window(name, centers, width):
    from backend.src.repair.steering import build_window_steering_schedule

    return build_window_steering_schedule(name, centers, width, 0.2).scale_strength(
        KAPPA
    )


def mark_width(ax, center, width, color):
    left = min(max(0, center - width / 2), 1 - width)
    right = left + width
    ax.annotate(
        "",
        xy=(left, 0.14),
        xytext=(right, 0.14),
        arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.3},
    )
    ax.text((left + right) / 2, 0.17, "$w$", color=color, ha="center", fontsize=12)


def draw(ax, method, schedules, times, values):
    title, color = METHODS[method]
    schedule = schedules[method]
    if method == "R2":
        for center in (WIDTH / 2, 1 - WIDTH / 2):
            alternate = window("alternative", [center], WIDTH)
            ax.plot(
                times,
                [alternate(t) for t in times],
                color=color,
                ls="--",
                alpha=0.40,
                lw=1.6,
            )
        ax.text(
            0.5,
            0.04,
            "Dashed: other preset positions",
            ha="center",
            fontsize=9,
            color=color,
        )
    ax.plot(times, values[method], color=color, linewidth=2.5)
    if method in ("R1", "R2", "R3", "R4"):
        x = {"R1": 0.86, "R2": 0.51, "R3": PEAK_TIME, "R4": R4_CENTERS[-1]}[method]
        ax.text(x, KAPPA + 0.04, "$\\kappa$", ha="center", color=color, fontsize=14)
    if method == "R2":
        mark_width(ax, 0.5, WIDTH, color)
    if method == "R3":
        mark_width(ax, PEAK_TIME, WIDTH, color)
        ax.axvline(PEAK_TIME, ymin=0, ymax=0.66 / 1.1, color=color, ls=":", alpha=0.65)
        ax.annotate(
            "$t^*$: deviation peak",
            xy=(PEAK_TIME, KAPPA),
            xytext=(0.72, 0.44),
            fontsize=9,
            ha="center",
            arrowprops={"arrowstyle": "->", "color": color},
        )
    if method == "R4":
        for center in R4_CENTERS:
            mark_width(ax, center, WIDTH, color)
    if method == "R5":
        ax.scatter(
            schedule.times,
            schedule.strengths,
            s=42,
            facecolors="white",
            edgecolors=color,
            linewidths=1.5,
            zorder=4,
        )
        for k in (0, 2, 4, 6, 8):
            ax.annotate(
                f"$\\theta_{k}$",
                (schedule.times[k], schedule.strengths[k]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=10,
            )
    ax.set(
        xlim=(-0.015, 1.015),
        ylim=(-0.025, 1.05),
        xticks=[0, 0.25, 0.5, 0.75, 1],
        yticks=[0, 0.5, 1],
        xlabel="Normalized generation time $t$",
        ylabel=r"Steering strength $\lambda(t)$",
    )
    ax.grid(alpha=0.16)
    ax.set_axisbelow(True)
    ax.set_title(f"{method}  |  {title}", loc="left", fontsize=12, pad=12)
    ax.text(
        0.025,
        0.97,
        DESCRIPTIONS[method],
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        linespacing=1.5,
    )


def run_steering_figure_reporting(destination: Path = DESTINATION) -> dict:
    """Write schematic curves and numerical/provenance tables to an explicit folder."""
    from backend.src.repair.steering import (
        SteeringSchedule,
        build_constant_steering_schedule,
    )

    destination.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    schedules = {
        "R1": build_constant_steering_schedule(KAPPA, "R1"),
        "R2": window("R2_middle", [0.5], WIDTH),
        "R3": window("R3_peak", [PEAK_TIME], WIDTH),
        "R4": window("R4_contiguous", R4_CENTERS, WIDTH),
        "R5": SteeringSchedule(
            "R5",
            tuple(np.linspace(0, 1, 9)),
            (0.10, 0.30, 0.70, 0.65, 0.20, 0.15, 0.40, 0.60, 0.10),
        ),
    }
    times = np.linspace(0, 1, 4001)
    values = {
        method: np.array([schedule(t) for t in times])
        for method, schedule in schedules.items()
    }
    for method, curve in values.items():
        assert np.all((0 <= curve) & (curve <= 1))
        area = np.sum(np.diff(times) * (curve[:-1] + curve[1:]) / 2)
        assert abs(area - schedules[method].integrate_strength()) < 1e-6
    assert np.allclose(values["R1"], KAPPA, rtol=0, atol=1e-14)
    assert all(schedules["R2"](t) == 0 for t in (0.0, 0.25, 0.75, 1.0))
    assert len(schedules["R5"].times) == 9

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), layout="constrained")
    for ax, method in zip(axes.flat, METHODS):
        draw(ax, method, schedules, times, values)
    key = axes.flat[-1]
    key.axis("off")
    key.text(
        0.02,
        0.97,
        "How strength and duration are defined",
        fontsize=13,
        weight="bold",
        va="top",
    )
    key.text(
        0.02,
        0.84,
        "$t=0$: initial noise; $t=1$: final image\n\n"
        "$\\lambda=0$: original velocity; $\\lambda=1$: safe velocity\n\n"
        "R1-R4: search amplitude $\\kappa$ per prompt.\n"
        "R2-R4: test widths $w$ = 0.25, 0.50, 0.75.\n"
        "Each isolated window uses 20% ramps\n"
        "at its interior start and end.\n\n"
        "R5: optimize nine equally spaced knot values;\n"
        "there is no separate window-width search.\n\n"
        "All transitions use smoothstep interpolation.\n"
        "Curves illustrate definitions; heights and\n"
        "adaptive centers are illustrative.",
        va="top",
        fontsize=10.5,
        linespacing=1.3,
    )
    fig.suptitle(
        "How R1-R5 define steering strength and application windows\n"
        "Schematic schedules: method definitions, not measured results",
        fontsize=16,
        weight="bold",
    )
    for extension in ("png", "svg", "pdf"):
        fig.savefig(destination / f"steering-schedules.{extension}", dpi=180)
    plt.close(fig)
    for method in METHODS:
        fig, ax = plt.subplots(figsize=(6.8, 4.6), layout="constrained")
        draw(ax, method, schedules, times, values)
        fig.suptitle(
            "Schematic method definition; illustrative curve",
            fontsize=10,
            color="#555555",
        )
        for extension in ("png", "svg"):
            fig.savefig(destination / f"steering-{method.lower()}.{extension}", dpi=180)
        plt.close(fig)

    with (destination / "steering-schedules.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "illustrative_normalized_time",
                *[f"{method}_illustrative_lambda" for method in METHODS],
            ]
        )
        writer.writerows(zip(times, *(values[method] for method in METHODS)))
    provenance = {
        "kind": "schematic_method_definitions",
        "notice": "Amplitudes and adaptive centers are illustrative; no measured results are plotted or feasibility claimed.",
        "source_results": None,
        "schedule_implementation": "backend/src/repair/steering.py",
        "schedule_implementation_sha256": hashlib.sha256(
            (ROOT / "backend/src/repair/steering.py").read_bytes()
        ).hexdigest(),
        "configured_candidate_widths": [0.25, 0.5, 0.75],
        "ramp_fraction": 0.2,
        "r5_knots": 9,
        "illustrative_schedules": {
            method: schedule.to_record() for method, schedule in schedules.items()
        },
    }
    (destination / "steering-schedules.provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print("Created five schematic method plots and combined PNG/SVG/PDF.")
    print(
        "Verified strength bounds, window support, knot count, and numerical plot integrals."
    )
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    return run_steering_figure_reporting(parser.parse_args(argv).output)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
