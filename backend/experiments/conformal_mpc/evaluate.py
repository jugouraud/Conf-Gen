"""Paired independent episode simulation with actual executed-state diagnostics."""

import time
from collections import Counter

import numpy as np

from backend.experiments.conformal_mpc.prepare import PreparedExperiment
from backend.experiments.conformal_mpc.synthetic import Scene
from backend.src.control.contracts import Detector, Mode, PreviousReference
from backend.src.control.pipeline import run_control_pipeline
from backend.src.mpc.geometry import Obstacle, minimum_clearance

MODES = ("nominal", "safe_only", "joint", "repaired")


def simulate_episode(
    experiment: PreparedExperiment, detector: Detector, scene: Scene, mode: Mode
) -> dict:
    cfg, tracking = experiment.config, experiment.tracking
    physical = np.asarray(scene.initial)
    previous = None
    rng = np.random.default_rng(scene.seed + 100000)
    disturbances = rng.normal(0, cfg["process_noise"], (cfg["physical_steps"], 2))
    records, executed = [], [physical.copy()]
    planned_clearance, executed_clearance = np.inf, np.inf
    reference_score_clearance = np.inf
    started = time.perf_counter()
    for k in range(cfg["physical_steps"]):
        observation = scene.observe(physical, k, cfg["horizon"], cfg["physical_dt"])
        result = run_control_pipeline(
            observation,
            experiment.model,
            detector,
            experiment.schedules,
            tracking,
            guidance=scene.guidance,
            mode=mode,
            seed=scene.seed + k,
            sampling=experiment.sampling,
            deadline_seconds=cfg["deadline_seconds"],
            previous=previous,
        )
        clearance = None
        proposal_clearance = None
        if result.sampled_reference is not None:
            proposal_clearance = minimum_clearance(
                result.sampled_reference,
                observation.obstacles,
                tracking.footprint_radius,
            )
            planned_clearance = min(planned_clearance, proposal_clearance)
        if result.reference is not None:
            clearance = minimum_clearance(
                result.reference, observation.obstacles, tracking.footprint_radius
            )
            planned_clearance = min(planned_clearance, clearance)
            reference_score_clearance = min(reference_score_clearance, clearance)
        else:
            # Keep all episodes in the target population, including no-output cases.
            reference_score_clearance = -np.inf
        if result.released:
            control = result.control
            previous = PreviousReference(observation, result.reference)
        else:
            control = np.zeros(2)  # Explicit simulator stop, not a certified fallback.
        following = (
            tracking.dynamics.rollout(physical, control[None])[1] + disturbances[k]
        )
        segment_obstacles = tuple(
            Obstacle(o.centers[:2], o.radius) for o in observation.obstacles
        )
        actual_clearance = minimum_clearance(
            np.stack((physical, following)),
            segment_obstacles,
            tracking.footprint_radius,
        )
        executed_clearance = min(executed_clearance, actual_clearance)
        error = (
            float(np.linalg.norm(following - result.reference[1]))
            if result.reference is not None
            else None
        )
        records.append(
            {
                "k": k,
                "released": result.released,
                "failure": result.failure,
                "fallback_reason": result.fallback_reason,
                "flagged": result.decision.flagged if result.decision else None,
                "safe_flag": result.decision.safe_flag if result.decision else None,
                "unsafe_flag": result.decision.unsafe_flag if result.decision else None,
                "d_safe": result.decision.d_safe if result.decision else None,
                "d_unsafe": result.decision.d_unsafe if result.decision else None,
                "repair_status": result.repair.status if result.repair else None,
                "steering": result.repair.schedule.integrate_strength()
                if result.repair
                else 0.0,
                "planned_clearance": clearance,
                "proposal_clearance": proposal_clearance,
                "executed_clearance": actual_clearance,
                "tracking_error": error,
                "latency_seconds": result.latency,
                "control": control.tolist(),
                "state": physical.tolist(),
            }
        )
        physical = following
        executed.append(physical.copy())
        if np.linalg.norm(physical - np.asarray(scene.goal)) <= cfg["goal_tolerance"]:
            break
    return {
        "episode": scene.episode,
        "mode": mode,
        "seed": scene.seed,
        "goal_completed": bool(
            np.linalg.norm(physical - np.asarray(scene.goal)) <= cfg["goal_tolerance"]
        ),
        "planned_minimum_clearance": None
        if planned_clearance == np.inf
        else planned_clearance,
        "executed_minimum_clearance": executed_clearance,
        "reference_score": tracking.required_clearance - reference_score_clearance,
        "executed_score": tracking.required_clearance - executed_clearance,
        "planned_collision": bool(planned_clearance < 0),
        "executed_collision": bool(executed_clearance < 0),
        "executed_positions": np.asarray(executed).tolist(),
        "steps": records,
        "episode_seconds": time.perf_counter() - started,
    }


def summarize(episodes: list[dict], labels: dict[str, bool], mode: str) -> dict:
    steps = [step for episode in episodes for step in episode["steps"]]
    errors = [s["tracking_error"] for s in steps if s["tracking_error"] is not None]
    latencies = [s["latency_seconds"] for s in steps]
    # Detection rates use the independent first query of each episode, never
    # count correlated replans as independent conformal calibration observations.
    by_class = {}
    for safe, name in (
        (True, "safe_false_positive_rate"),
        (False, "unsafe_detection_rate"),
    ):
        selected = [e["steps"][0] for e in episodes if labels[e["episode"]] == safe]
        flags = [
            (s["safe_flag"] if mode == "safe_only" else s["flagged"]) for s in selected
        ]
        by_class[name] = float(np.mean(flags)) if flags else None
    return {
        **by_class,
        "episodes": len(episodes),
        "physical_steps": len(steps),
        "released_fraction": float(np.mean([s["released"] for s in steps])),
        "repair_failures": sum(
            s["repair_status"] in {"repair_infeasible", "missing_anchor"} for s in steps
        ),
        "failure_counts": dict(Counter(s["failure"] for s in steps if s["failure"])),
        "fallback_uses": sum(
            s["released"] and s["fallback_reason"] is not None for s in steps
        ),
        "planned_collision_rate": float(
            np.mean([e["planned_collision"] for e in episodes])
        ),
        "executed_collision_rate": float(
            np.mean([e["executed_collision"] for e in episodes])
        ),
        "minimum_planned_clearance": min(
            (
                e["planned_minimum_clearance"]
                for e in episodes
                if e["planned_minimum_clearance"] is not None
            ),
            default=None,
        ),
        "minimum_executed_clearance": min(
            e["executed_minimum_clearance"] for e in episodes
        ),
        "goal_completion_rate": float(np.mean([e["goal_completed"] for e in episodes])),
        "tracking_rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        "mean_steering": float(np.mean([s["steering"] for s in steps])),
        "mean_latency_seconds": float(np.mean(latencies)),
        "p95_latency_seconds": float(np.quantile(latencies, 0.95)),
        "maximum_latency_seconds": max(latencies),
        "detector_used_for_release": mode != "nominal",
    }


def run_simulation_evaluation(
    experiment: PreparedExperiment, detector: Detector, *, split: str
) -> dict:
    if split not in {"output_calibration", "test"}:
        raise ValueError("Evaluation uses only fresh output-calibration or test scenes")
    labels = {row.context.episode: row.safe for row in experiment.groups[split]}
    results = {}
    for mode in MODES:
        episodes = [
            simulate_episode(experiment, detector, scene, mode)
            for scene in experiment.scenes[split]
        ]
        results[mode] = {
            "episodes": episodes,
            "metrics": summarize(episodes, labels, mode),
        }
    return results
