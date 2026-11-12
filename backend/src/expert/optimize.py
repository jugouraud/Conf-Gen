"""One bounded expert attempt, independent of the deployed diffusion generator."""

import time
from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from backend.src.expert.config import ExpertConfig, Guidance
from backend.src.expert.problem import (
    expert_cost,
    initial_controls,
    physical_times,
    planning_constraints,
)
from backend.src.expert.scenes import Scene
from backend.src.expert.validate import run_expert_validation, validate_initial_scene
from backend.src.expert.vehicle import Vehicle


def run_expert_optimization(
    scene: Scene,
    guidance: Guidance,
    config: ExpertConfig,
    *,
    guess: int,
    seed: int,
    retry: int = 0,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    started = clock()
    p, v = config.planner, config.vehicle
    result = {
        "status": "pending",
        "solver_success": False,
        "feasible": False,
        "eligible": False,
        "guess": guess,
        "retry": retry,
        "seed": seed,
        "states": None,
        "controls": None,
        "timestamps": None,
        "y_ref_star": None,
        "validation": None,
        "cost": None,
    }
    reasons = validate_initial_scene(scene, config)
    if guidance.side and guidance.obstacle_index >= len(scene.obstacles):
        reasons.append("guidance_obstacle_missing")
    if reasons:
        return dict(
            result,
            status="invalid_scene_or_guidance",
            validation={"valid": False, "reasons": reasons},
            runtime_seconds=clock() - started,
        )
    vehicle = Vehicle(v, p.dt)
    initial = np.asarray(scene.initial_state)
    deadline = started + p.solver_timeout

    def check_deadline() -> None:
        if clock() >= deadline:
            raise TimeoutError("expert_timeout")

    def rollout(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        check_deadline()
        controls = flat.reshape((p.horizon, 2))
        return controls, vehicle.rollout(initial, controls)

    def cost(flat: np.ndarray) -> float:
        controls, states = rollout(flat)
        return expert_cost(scene, controls, states, config)

    def constraints(flat: np.ndarray) -> np.ndarray:
        controls, states = rollout(flat)
        return planning_constraints(scene, guidance, states, controls, config, vehicle)

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            controls = initial_controls(
                scene, guidance, config, vehicle, guess, seed, retry
            )
            bounds = list(
                zip(
                    np.tile(v.input_lower, p.horizon),
                    np.tile(v.input_upper, p.horizon),
                    strict=True,
                )
            )
            solved = minimize(
                cost,
                controls.ravel(),
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": p.max_iterations, "ftol": p.ftol},
            )
            check_deadline()
            controls, states = rollout(solved.x)
            timestamps = physical_times(scene, config)
            reference = vehicle.output(states)
            validation = run_expert_validation(
                scene, guidance, config, states, controls, timestamps, reference
            )
            feasible = validation["valid"]
            eligible = feasible and (solved.success or not p.require_convergence)
            result.update(
                status="validated"
                if eligible
                else "validation_failed"
                if not feasible
                else "solver_not_converged",
                solver_success=bool(solved.success),
                solver_code=int(solved.status),
                iterations=int(solved.nit),
                feasible=feasible,
                eligible=eligible,
                cost=expert_cost(scene, controls, states, config),
                states=states.tolist(),
                controls=controls.tolist(),
                timestamps=timestamps.tolist(),
                y_ref_star=reference.tolist(),
                validation=validation,
            )
            check_deadline()
    except TimeoutError:
        result.update(status="solver_timeout", eligible=False)
    except (
        ValueError,
        RuntimeError,
        FloatingPointError,
        OverflowError,
        np.linalg.LinAlgError,
    ):
        result.update(status="numerical_or_solver_failure", eligible=False)
    result["runtime_seconds"] = clock() - started
    return result
