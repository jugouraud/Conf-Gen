"""Offline vehicle/output maps; the default reuses the online linear dynamics."""

import numpy as np

from backend.src.expert.config import VehicleSettings
from backend.src.mpc.dynamics import LinearDynamics


class Vehicle:
    def __init__(self, settings: VehicleSettings, dt: float):
        self.settings = settings
        self.dt = dt
        self.state_dimension = 2 if settings.kind == "velocity_disc" else 5
        self.output_dimension = 2 if settings.output == "xy" else 4
        self.linear = (
            LinearDynamics(np.eye(2), dt * np.eye(2), np.eye(2), dt)
            if settings.kind == "velocity_disc"
            else None
        )

    def derivative(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        _, _, heading, speed, steering = state
        return np.array(
            [
                speed * np.cos(heading),
                speed * np.sin(heading),
                speed * np.tan(steering) / self.settings.wheelbase,
                control[0],
                control[1],
            ]
        )

    def advance(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """RK4 bicycle substep, or exact constant-velocity disc translation."""
        if self.linear is not None:
            return np.asarray(state) + dt * np.asarray(control)
        k1 = self.derivative(state, control)
        k2 = self.derivative(state + dt * k1 / 2, control)
        k3 = self.derivative(state + dt * k2 / 2, control)
        k4 = self.derivative(state + dt * k3, control)
        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        if self.linear is not None:
            return self.linear.a @ state + self.linear.b @ control
        result = np.asarray(state).copy()
        for _ in range(self.settings.substeps):
            result = self.advance(result, control, self.dt / self.settings.substeps)
        return result

    def rollout(self, initial: np.ndarray, controls: np.ndarray) -> np.ndarray:
        if self.linear is not None:
            return self.linear.rollout(initial, controls)
        states = [np.asarray(initial)]
        for control in controls:
            states.append(self.step(states[-1], control))
        return np.stack(states)

    def output(self, states: np.ndarray) -> np.ndarray:
        if self.settings.output == "xy":
            return states[:, :2].copy()
        return np.column_stack(
            (states[:, :2], np.sin(states[:, 2]), np.cos(states[:, 2]))
        )

    def dense_path(
        self, states: np.ndarray, controls: np.ndarray, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.linear is not None:
            return times, states[:, :2]
        dense_times, positions = [times[0]], [states[0, :2]]
        for i, control in enumerate(controls):
            current = states[i].copy()
            for j in range(self.settings.substeps):
                current = self.advance(
                    current, control, self.dt / self.settings.substeps
                )
                dense_times.append(
                    times[i] + (j + 1) * self.dt / self.settings.substeps
                )
                positions.append(current[:2])
        return np.asarray(dense_times), np.stack(positions)

    def curve_margin(self) -> float:
        if self.linear is not None:
            return 0.0
        v = self.settings
        acceleration_bound = (
            v.max_acceleration + v.max_speed**2 * np.tan(v.max_steering) / v.wheelbase
        )
        return float(
            acceleration_bound * (self.dt / v.substeps) ** 2 / 8 + v.integration_margin
        )
