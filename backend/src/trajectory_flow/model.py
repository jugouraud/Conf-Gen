"""Trainable conditional trajectory flow with the existing frozen Torch adapter ABI.

The physical context remains immutable. The trainable representation is computed
inside drift, so encoder gradients are retained during supervised training.
"""

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from backend.src.control.contracts import Context


@dataclass(frozen=True)
class FlowArchitecture:
    horizon: int = 16
    dt: float = 0.4
    width: int = 128
    depth: int = 3
    context_width: int = 64

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or not 1 <= self.horizon <= 128 or (
            isinstance(self.dt, bool) or not isinstance(self.dt, (int, float))
            or not np.isfinite(self.dt) or not 0 < self.dt <= 60
        ):
            raise ValueError("Invalid physical horizon or dt")
        if type(self.width) is not int or not 16 <= self.width <= 1024 or (
            type(self.depth) is not int or not 1 <= self.depth <= 8
        ):
            raise ValueError("Invalid flow-network capacity")
        if type(self.context_width) is not int or not 8 <= self.context_width <= 512:
            raise ValueError("Invalid context width")

    def to_record(self) -> dict:
        return asdict(self)


def ego_features(context: Context) -> np.ndarray:
    """15 physical features: xy, goal, displacement, velocity, map, dt, H, count.

    Map bounds are expressed relative to ego position; coordinates are metres.
    """
    if len(context.state) != 2 or len(context.features) != 6:
        raise ValueError("Disc requires xy state and (vx,vy,xmin,xmax,ymin,ymax)")
    start = np.asarray(context.state, dtype=np.float64)
    goal = np.asarray(context.goal, dtype=np.float64)
    velocity = np.asarray(context.features[:2], dtype=np.float64)
    xmin, xmax, ymin, ymax = context.features[2:]
    map_relative = np.asarray([xmin - start[0], xmax - start[0],
                               ymin - start[1], ymax - start[1]])
    return np.concatenate((start, goal, goal - start, velocity, map_relative,
                           [context.dt, context.horizon, len(context.obstacles)]))


class TrajectoryFlowModel(nn.Module):
    """Direct position-trajectory FM model; drift coordinates are normalized xy.

    Model uses a fixed horizon and output shape (H+1,2). Obstacle counts vary;
    padded entries do not contribute to the permutation-invariant mean encoder.
    Guidance names map to *geometric* descriptors, never scene-specific IDs.
    """

    def __init__(
        self,
        architecture: FlowArchitecture,
        guidance: Mapping[str, Mapping[str, object]],
        *,
        revision: str = "untrained-trajectory-flow-v1",
    ) -> None:
        super().__init__()
        if not revision or not guidance or len(guidance) > 32:
            raise ValueError("Model revision and bounded guidance catalogue required")
        self.architecture = architecture
        self.revision = revision
        descriptions = {}
        for name, entry in guidance.items():
            if not name or not isinstance(name, str) or set(entry) != {
                "side", "speed_scale", "obstacle_index"
            }:
                raise ValueError("Guidance must have side, speed_scale, obstacle_index")
            side, speed, index = entry["side"], entry["speed_scale"], entry["obstacle_index"]
            if type(side) is not int or side not in {-1, 0, 1} or (
                isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0 < speed <= 1
            ) or type(index) is not int or not 0 <= index < 32:
                raise ValueError("Invalid geometric guidance")
            descriptions[name] = MappingProxyType(dict(
                side=side, speed_scale=float(speed), obstacle_index=index
            ))
        self.guidance = MappingProxyType(descriptions)
        h = architecture
        d = 2 * (h.horizon + 1)
        w = h.context_width
        self.obstacle_net = nn.Sequential(nn.Linear(d + 1, w), nn.SiLU(), nn.Linear(w, w))
        self.scene_net = nn.Sequential(nn.Linear(15 + w, w), nn.SiLU(), nn.Linear(w, w))
        self.guidance_net = nn.Sequential(nn.Linear(5, w), nn.SiLU(), nn.Linear(w, w))
        self.null_scene = nn.Parameter(torch.zeros(w))
        self.null_guidance = nn.Parameter(torch.zeros(w))
        layers: list[nn.Module] = [nn.Linear(d + 5 + 2 * w, h.width), nn.SiLU()]
        for _ in range(h.depth - 1):
            layers.extend((nn.Linear(h.width, h.width), nn.SiLU()))
        layers.append(nn.Linear(h.width, d))
        self.flow_net = nn.Sequential(*layers)
        self.register_buffer("target_mean", torch.zeros(2))
        self.register_buffer("target_scale", torch.ones(2))
        self.register_buffer("ego_mean", torch.zeros(15))
        self.register_buffer("ego_scale", torch.ones(15))
        self.register_buffer("length_scale", torch.ones(()))
        self._frozen_versions: tuple | None = None

    def set_normalization(self, targets: np.ndarray, contexts: Sequence[Context]) -> None:
        """Fit on generator-training groups only, excluding held-out validation."""
        if not len(targets) or len(targets) != len(contexts) or self._frozen_versions:
            raise ValueError("Normalization needs nonempty unfrozen fit data")
        if targets.shape[1:] != (self.architecture.horizon + 1, 2):
            raise ValueError("Wrong reference shape for normalization")
        flat = targets.reshape(-1, 2)
        ego = np.stack([ego_features(c) for c in contexts])
        if not np.isfinite(flat).all() or not np.isfinite(ego).all():
            raise ValueError("Nonfinite normalization data")
        # Constant dimensions get scale 1 rather than amplification by epsilon.
        def scale(values: np.ndarray) -> np.ndarray:
            std = values.std(axis=0)
            return np.where(std >= 1e-6, std, 1.0)

        with torch.no_grad():
            self.target_mean.copy_(torch.as_tensor(flat.mean(axis=0), dtype=torch.float32))
            self.target_scale.copy_(torch.as_tensor(scale(flat), dtype=torch.float32))
            self.ego_mean.copy_(torch.as_tensor(ego.mean(axis=0), dtype=torch.float32))
            self.ego_scale.copy_(torch.as_tensor(scale(ego), dtype=torch.float32))
            self.length_scale.copy_(torch.as_tensor(max(float(np.mean(scale(flat))), 0.1)))

    def encode(self, obs: Context) -> Context:
        if not isinstance(obs, Context) or obs.horizon != self.architecture.horizon or (
            not np.isclose(obs.dt, self.architecture.dt, atol=1e-12, rtol=0)
        ):
            raise ValueError("Context horizon/dt does not match trained model")
        ego_features(obs)
        return obs  # Owned, frozen physical context; neural encoding occurs in drift.

    def _context_tensors(self, contexts: Sequence[Context], names: Sequence[str]):
        if len(contexts) != len(names):
            raise ValueError("Context and guidance batch mismatch")
        device = self.target_mean.device
        scalar = float(self.length_scale)
        egos, guidance, obstacles = [], [], []
        for c, name in zip(contexts, names, strict=True):
            self.encode(c)
            if name not in self.guidance:
                raise ValueError("Unknown guidance for frozen training catalogue")
            spec = self.guidance[name]
            if spec["side"] and spec["obstacle_index"] >= len(c.obstacles):
                raise ValueError("Guidance-selected obstacle is absent")
            start = np.asarray(c.state)
            entries = []
            for obstacle in c.obstacles:
                centers = (np.asarray(obstacle.centers) - start) / scalar
                entries.append(np.r_[centers.ravel(), obstacle.radius / scalar])
            obstacles.append(entries)
            selected = np.zeros(3)
            if spec["side"] and spec["obstacle_index"] < len(c.obstacles):
                chosen = c.obstacles[spec["obstacle_index"]]
                midpoint = np.asarray(chosen.centers)[self.architecture.horizon // 2]
                selected = np.r_[(midpoint - start) / scalar, chosen.radius / scalar]
            guidance.append([spec["side"], spec["speed_scale"], *selected])
            egos.append(ego_features(c))
        maximum = max((len(rows) for rows in obstacles), default=0)
        features = np.zeros((len(contexts), maximum, 2 * (self.architecture.horizon + 1) + 1), dtype=np.float32)
        masks = np.zeros((len(contexts), maximum), dtype=bool)
        for i, rows in enumerate(obstacles):
            if rows:
                features[i, :len(rows)] = rows
                masks[i, :len(rows)] = True
        ego = (torch.as_tensor(np.stack(egos), dtype=torch.float32, device=device) - self.ego_mean) / self.ego_scale
        return (ego,
                torch.as_tensor(features, device=device),
                torch.as_tensor(masks, device=device),
                torch.as_tensor(guidance, dtype=torch.float32, device=device))

    def _embedded_condition(self, contexts: Sequence[Context], names: Sequence[str]):
        ego, obstacles, mask, guidance = self._context_tensors(contexts, names)
        if obstacles.shape[1]:
            encoded = self.obstacle_net(obstacles)
            pooled = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)
        else:
            pooled = ego.new_zeros((len(contexts), self.architecture.context_width))
        return self.scene_net(torch.cat((ego, pooled), -1)), self.guidance_net(guidance)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor,
        contexts: Sequence[Context], names: Sequence[str],
        *, drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.drift_batch(x, t, contexts, names, drop_mask=drop_mask)

    def normalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        return ((trajectory - self.target_mean) / self.target_scale).reshape(*trajectory.shape[:-2], -1)

    def decode(self, x_final: torch.Tensor) -> torch.Tensor:
        return x_final.reshape(*x_final.shape[:-1], self.architecture.horizon + 1, 2) * self.target_scale + self.target_mean

    def drift_batch(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        contexts: Sequence[Context],
        names: Sequence[str],
        *,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict dX/dt, with dropout replacing both scene and guidance by NULL."""
        if x.ndim != 2 or x.shape[1] != 2 * (self.architecture.horizon + 1):
            raise ValueError("Expected flat normalized trajectory batch")
        if t.shape != (len(x), 1) or len(contexts) != len(x):
            raise ValueError("Invalid per-item time or context batch")
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(t).all()) or (
            not bool(((t >= 0) & (t <= 1)).all())
        ):
            raise ValueError("Flow inputs must be finite and time in [0,1]")
        scene, guide = self._embedded_condition(contexts, names)
        if drop_mask is not None:
            if drop_mask.shape != (len(x),):
                raise ValueError("Dropout mask must be one boolean per example")
            scene = torch.where(drop_mask[:, None], self.null_scene, scene)
            guide = torch.where(drop_mask[:, None], self.null_guidance, guide)
        angles = 2 * torch.pi * t
        clock = torch.cat((t, torch.sin(angles), torch.cos(angles),
                           torch.sin(2 * angles), torch.cos(2 * angles)), -1)
        return self.flow_net(torch.cat((x, clock, scene, guide), -1))

    def _single(self, x: torch.Tensor, t: float, c: Context | None, a: str | None) -> torch.Tensor:
        if x.numel() != 2 * (self.architecture.horizon + 1):
            raise ValueError("Latent dimension does not match horizon")
        if not np.isfinite(t) or not 0 <= t <= 1:
            raise ValueError("Flow time must be in [0,1]")
        batch = x.reshape(1, -1)
        moment = x.new_full((1, 1), float(t))
        if c is None and a is None:
            # Build exactly the same clock features as the conditional branch.
            angle = 2 * torch.pi * moment
            clock = torch.cat((moment, torch.sin(angle), torch.cos(angle),
                               torch.sin(2 * angle), torch.cos(2 * angle)), -1)
            condition = torch.cat((batch, clock, self.null_scene[None], self.null_guidance[None]), -1)
            return self.flow_net(condition).reshape_as(x)
        if c is None or a is None:
            raise ValueError("Context and guidance must be jointly present or NULL")
        return self.drift_batch(batch, moment, [c], [a]).reshape_as(x)

    def drift(self, x: torch.Tensor, t: float, c: Context, a: str) -> torch.Tensor:
        return self._single(x, t, c, a)

    def baseline(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return self._single(x, t, None, None)

    def initialize(self, c: Context, rng: np.random.Generator, warm: torch.Tensor | None) -> torch.Tensor:
        self.encode(c)
        d = 2 * (self.architecture.horizon + 1)
        noise = rng.standard_normal(d).astype(np.float32)
        x0 = torch.as_tensor(noise, device=self.target_mean.device)
        if warm is not None:
            if warm.shape != (self.architecture.horizon + 1, 2):
                raise ValueError("Shifted warm reference shape mismatch")
            # Explicitly different initializer distribution; requires separate calibration.
            x0 = x0 + 0.1 * self.normalize(warm)
        return x0

    def noise(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return torch.zeros_like(x)  # deterministic flow matching; G=0.

    def assert_frozen(self) -> None:
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise ValueError("Model must be frozen before probe construction")
        current = tuple((name, value._version) for name, value in (
            *self.named_parameters(), *self.named_buffers()
        ))
        if self._frozen_versions is not None and current != self._frozen_versions:
            raise ValueError("Frozen model parameters or buffers changed")

    def freeze(self, revision: str) -> None:
        if not revision:
            raise ValueError("Frozen model revision required")
        self.eval().requires_grad_(False)
        self.revision = revision
        self._frozen_versions = tuple((name, value._version) for name, value in (
            *self.named_parameters(), *self.named_buffers()
        ))
