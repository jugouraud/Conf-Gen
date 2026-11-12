"""Conditional flow-matching optimization and versioned weight-only checkpoint."""

import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from backend.src.common.storage import calculate_file_sha256, write_json
from backend.src.trajectory_flow.data import FlowExample, PreparedTraining
from backend.src.trajectory_flow.model import FlowArchitecture, TrajectoryFlowModel


@dataclass(frozen=True)
class TrainingSettings:
    seed: int = 20261112
    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    unconditional_probability: float = 0.15
    validation_fraction: float = 0.2
    patience: int = 35
    threads: int = 2
    device: str = "cpu"

    def __post_init__(self) -> None:
        if any(type(value) is not int or not low <= value <= high for value, low, high in (
            (self.seed, 0, 2**32 - 1), (self.epochs, 1, 100000),
            (self.batch_size, 1, 4096), (self.patience, 1, 100000),
            (self.threads, 1, 32),
        )):
            raise ValueError("Invalid integer training budget")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or (
            not np.isfinite(value) or not low <= value <= high
        ) for value, low, high in (
            (self.learning_rate, 1e-8, 1.0), (self.weight_decay, 0.0, 1.0),
            (self.grad_clip, 1e-8, 1000.0),
            (self.unconditional_probability, 0.01, 0.9),
            (self.validation_fraction, 0.01, 0.49),
        )):
            raise ValueError("Invalid learning rate/dropout/gradient settings")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Supported training devices are cpu and cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")


def _batch_loss(
    model: TrajectoryFlowModel,
    rows: tuple[FlowExample, ...],
    generator: torch.Generator,
    probability: float,
    *,
    training: bool,
) -> torch.Tensor:
    """Independent x0/t per item, straight conditional FM coupling, MSE."""
    device = model.target_mean.device
    y = torch.as_tensor(np.stack([row.target for row in rows]), dtype=torch.float32, device=device)
    x1 = model.normalize(y)
    # CPU generator explicitly controls the random stream on CPU and GPU.
    x0 = torch.randn(x1.shape, generator=generator).to(device)
    t = torch.rand((len(rows), 1), generator=generator).to(device)
    xt = (1 - t) * x0 + t * x1
    velocity = x1 - x0
    drop = (torch.rand((len(rows),), generator=generator) < probability).to(device) if training else None
    prediction = model.drift_batch(
        xt, t, [row.context for row in rows], [row.guidance for row in rows],
        drop_mask=drop,
    )
    return torch.mean((prediction - velocity).square())


def _epoch_loss(
    model: TrajectoryFlowModel,
    rows: tuple[FlowExample, ...],
    settings: TrainingSettings,
    generator: torch.Generator,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(rows), generator=generator).tolist() if training else list(range(len(rows)))
    weighted, count = 0.0, 0
    for index in range(0, len(order), settings.batch_size):
        batch = tuple(rows[j] for j in order[index:index + settings.batch_size])
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(model, batch, generator, settings.unconditional_probability, training=True)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip, error_if_nonfinite=True)
            optimizer.step()
        else:
            with torch.no_grad():
                loss = _batch_loss(model, batch, generator, settings.unconditional_probability, training=False)
        weighted += float(loss.detach()) * len(batch)
        count += len(batch)
    return weighted / count


def _publish_weights(path: Path, state: dict) -> None:
    """A complete state_dict only; never serialize Python model objects."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".weights.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_flow_training(
    prepared: PreparedTraining,
    architecture: FlowArchitecture,
    settings: TrainingSettings,
    output: Path,
) -> dict:
    """Train and publish a complete immutable run; calibration episodes unused.

    Output owns weights.pt (strict safe state_dict), metadata.json, history.json.
    A failed run does not publish metadata.json and cannot be loaded as complete.
    """
    if (architecture.horizon, architecture.dt) != (prepared.horizon, prepared.dt):
        raise ValueError("Architecture differs from expert physical indexing")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be new/empty; no implicit resume")
    output.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).parent
    source_names = ("model.py", "data.py", "train.py", "sample.py")
    fingerprints = {name: calculate_file_sha256(source_dir / name) for name in source_names}
    torch.set_num_threads(settings.threads)
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)
    model = TrajectoryFlowModel(architecture, prepared.guidance).to(settings.device)
    model.set_normalization(
        np.stack([row.target for row in prepared.fit]),
        [row.context for row in prepared.fit],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
    generator = torch.Generator(device="cpu").manual_seed(settings.seed + 1)
    history, best_loss, best_epoch, best_state, stale = [], float("inf"), 0, None, 0
    started = time.perf_counter()
    print(
        "Training blank trajectory flow field: "
        f"{len(prepared.fit)} fit targets, {len(prepared.validation)} validation targets, "
        f"up to {settings.epochs} epochs on {settings.device}.",
        flush=True,
    )
    for epoch in range(1, settings.epochs + 1):
        training_loss = _epoch_loss(model, prepared.fit, settings, generator, optimizer)
        # The same held-out (x0,t) stream each epoch makes selection comparable.
        validation_generator = torch.Generator(device="cpu").manual_seed(settings.seed + 2)
        validation_loss = _epoch_loss(model, prepared.validation, settings, validation_generator, None)
        if not np.isfinite(validation_loss):
            raise ValueError("Nonfinite held-out flow-matching loss")
        history.append({"epoch": epoch, "train_mse": training_loss, "validation_mse": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        schedule.step()
        print(
            f"Flow epoch {epoch}/{settings.epochs}: train_mse={training_loss:.6g}, "
            f"validation_mse={validation_loss:.6g}, best={best_loss:.6g} "
            f"(epoch {best_epoch}, stale {stale}/{settings.patience}, "
            f"elapsed {time.perf_counter() - started:.1f}s).",
            flush=True,
        )
        if stale >= settings.patience:
            print("Flow training stopped early: validation patience exhausted.", flush=True)
            break
    if best_state is None:
        raise ValueError("Training produced no valid checkpoint")
    model.load_state_dict(best_state, strict=True)
    weights = output / "weights.pt"
    _publish_weights(weights, best_state)
    digest = calculate_file_sha256(weights)
    if any(calculate_file_sha256(source_dir / name) != checksum
           for name, checksum in fingerprints.items()):
        raise ValueError("Model source changed during training")
    revision = hashlib.sha256(json.dumps({
        "weights": digest, "architecture": architecture.to_record(),
        "guidance": prepared.guidance, "source_sha256": fingerprints,
    }, sort_keys=True).encode()).hexdigest()
    model.freeze(revision)
    metadata = {
        "schema": "trajectory-flow-checkpoint-v1",
        "model_revision": revision,
        "weights_sha256": digest,
        "architecture": architecture.to_record(),
        "guidance": prepared.guidance,
        "dataset_id": prepared.dataset_id,
        "expert_manifest_sha256": prepared.manifest_sha256,
        "fit_groups": prepared.fit_groups,
        "validation_groups": prepared.validation_groups,
        "fit_targets": len(prepared.fit),
        "validation_targets": len(prepared.validation),
        "settings": asdict(settings),
        "seed": settings.seed,
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "epochs_run": len(history),
        "flow_coupling": "x0~N(0,I), t~U(0,1), xt=(1-t)*x0+t*x1, u=x1-x0; G=0",
        "normalization": "fit-groups only; pooled xy moments; buffers in state_dict",
        "source_sha256": fingerprints,
        "torch_version": torch.__version__,
    }
    # Full metadata is written last, acting as a completion marker.
    write_json(output / "history.json", history)
    write_json(output / "metadata.json", metadata)
    print(
        "Trajectory flow training complete: "
        f"best epoch {best_epoch}, validation_mse={best_loss:.6g}, "
        f"{len(history)} epochs, {time.perf_counter() - started:.1f}s elapsed.",
        flush=True,
    )
    return dict(metadata, model=model)


def load_flow_checkpoint(directory: Path, *, expected_sha256: str, device: str = "cpu") -> TrajectoryFlowModel:
    """Reconstruct a trusted model and verify weights without unpickling modules."""
    path = directory / "metadata.json"
    with path.open("rb") as stream:
        raw = stream.read(128 * 1024 + 1)
    if len(raw) > 128 * 1024:
        raise ValueError("Checkpoint metadata exceeds 128 KiB")
    metadata = json.loads(raw.decode("utf-8"))
    if metadata.get("schema") != "trajectory-flow-checkpoint-v1" or (
        metadata.get("weights_sha256") != expected_sha256 or len(expected_sha256) != 64
    ):
        raise ValueError("Checkpoint identity mismatch")
    weights = directory / "weights.pt"
    if calculate_file_sha256(weights) != expected_sha256:
        raise ValueError("Checkpoint weight checksum mismatch")
    fingerprints = metadata["source_sha256"]
    source_dir = Path(__file__).parent
    if set(fingerprints) != {"model.py", "data.py", "train.py", "sample.py"} or any(
        calculate_file_sha256(source_dir / name) != checksum
        for name, checksum in fingerprints.items()
    ):
        raise ValueError("Model implementation changed; recalibrate and retrain")
    revision = hashlib.sha256(json.dumps({
        "weights": expected_sha256, "architecture": metadata["architecture"],
        "guidance": metadata["guidance"], "source_sha256": fingerprints,
    }, sort_keys=True).encode()).hexdigest()
    if metadata.get("model_revision") != revision:
        raise ValueError("Model revision mismatch")
    model = TrajectoryFlowModel(
        FlowArchitecture(**metadata["architecture"]), metadata["guidance"], revision=revision
    )
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.to(device).freeze(revision)
    return model
