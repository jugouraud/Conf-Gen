"""Trajectory-model contract and adapter for an application-supplied Torch model."""

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from backend.src.common.storage import calculate_file_sha256
from backend.src.control.contracts import Context


class TrajectoryModel(Protocol):
    revision: str

    def assert_frozen(self) -> None: ...
    def encode(self, obs: Context) -> Context: ...
    def drift(self, x: np.ndarray, t: float, c: Context, a: str) -> np.ndarray: ...
    def baseline(self, x: np.ndarray, t: float) -> np.ndarray: ...
    def decode(self, x_final: np.ndarray) -> np.ndarray: ...
    def initialize(
        self, c: Context, rng: np.random.Generator, warm: np.ndarray | None
    ) -> np.ndarray: ...
    def noise(self, x: np.ndarray, t: float) -> np.ndarray: ...


class TorchTrajectoryAdapter:
    """Freeze a trusted, locally constructed module before any calibration.

    The module implements the contract above with Torch tensors for latent
    inputs/outputs; encode returns the immutable Context. Time convention and
    native normalization belong to that module. No SD3 image checkpoint is
    presented as a trajectory model. The sampler is run_reference_sampling.
    """

    def __init__(
        self, module: torch.nn.Module, revision: str, *, device: str = "cpu"
    ) -> None:
        if not revision:
            raise ValueError("Pinned checkpoint revision required")
        self.module = module.to(device).eval().requires_grad_(False)
        self.device = device
        self.revision = revision
        self.versions = self._versions()

    def _versions(self) -> tuple:
        return tuple(
            (name, id(value), value._version)
            for name, value in (
                *self.module.named_parameters(),
                *self.module.named_buffers(),
            )
        )

    @classmethod
    def from_checkpoint(
        cls,
        module: torch.nn.Module,
        path: Path,
        expected_sha256: str,
        *,
        device: str = "cpu",
    ) -> "TorchTrajectoryAdapter":
        """Load weights only; expected digest must come from a trusted source."""
        if calculate_file_sha256(path) != expected_sha256:
            raise ValueError("Checkpoint hash mismatch")
        module.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True), strict=True
        )
        return cls(module, expected_sha256, device=device)

    def assert_frozen(self) -> None:
        if (
            any(m.training for m in self.module.modules())
            or any(p.requires_grad for p in self.module.parameters())
            or self._versions() != self.versions
        ):
            raise ValueError("Model changed after freezing; recalibration required")

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        parameter = next(self.module.parameters(), None)
        dtype = parameter.dtype if parameter is not None else torch.float64
        return torch.tensor(value, dtype=dtype, device=self.device)

    def _call(self, method: str, *args: object) -> np.ndarray:
        self.assert_frozen()
        with torch.inference_mode():
            result = getattr(self.module, method)(*args)
        self.assert_frozen()
        return result.detach().cpu().double().numpy().copy()

    def encode(self, obs: Context) -> Context:
        self.assert_frozen()
        with torch.inference_mode():
            context = self.module.encode(obs)
        self.assert_frozen()
        if not isinstance(context, Context):
            raise ValueError("Encoder must return immutable Context")
        return context

    def drift(self, x: np.ndarray, t: float, c: Context, a: str) -> np.ndarray:
        return self._call("drift", self._tensor(x), t, c, a)

    def baseline(self, x: np.ndarray, t: float) -> np.ndarray:
        return self._call("baseline", self._tensor(x), t)

    def decode(self, x_final: np.ndarray) -> np.ndarray:
        return self._call("decode", self._tensor(x_final))

    def initialize(
        self, c: Context, rng: np.random.Generator, warm: np.ndarray | None
    ) -> np.ndarray:
        # All random input is explicit and paired; modules must not use global RNG.
        return self._call(
            "initialize", c, rng, None if warm is None else self._tensor(warm)
        )

    def noise(self, x: np.ndarray, t: float) -> np.ndarray:
        return self._call("noise", self._tensor(x), t)
