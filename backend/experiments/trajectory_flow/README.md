# Conditional trajectory flow matching (velocity-disc research implementation)

This adds a **new, randomly initialized** PyTorch spline-trajectory vector field.
The flow transports Gaussian spline coefficients, with deterministic endpoint
metadata. It decodes them relative to the fixed-track centerline; it never
samples independent Gaussian noise for all 97 physical waypoints.
No SD3/image checkpoint is loaded or reused. The supervised source is the existing
[`expert_trajectories`](../expert_trajectories/README.md) generator, which creates
validated `(observation, guidance, y_ref_star)` tuples using SciPy constrained
shooting. The trajectory network starts **untrained** and must be trained before
any detector calibration or MPC reference release. Neither generator training
nor successful sampling certifies obstacle avoidance or tracking safety.

## Generate a larger training dataset

From repository root using Python 3.12 and `uv sync --locked`:

```bash
# The configured 10,000 scenes use one nominal shortest-route guidance label.
# Its 50% train split is sized for roughly 5k validated train targets; rejected
# scenes or validation failures reduce the final count.
uv run python -m backend.experiments.expert_trajectories.pipeline generate \
  --config backend/experiments/trajectory_flow/expert_training.toml \
  --output .cache/expert-many
uv run python -m backend.experiments.expert_trajectories.pipeline validate \
  --output .cache/expert-many

uv run python -m backend.experiments.trajectory_flow.pipeline \
  --dataset .cache/expert-many \
  --config backend/experiments/trajectory_flow/config.toml \
  --output .cache/trained-flow
```

For a very small **functional smoke test**, replace `.cache/expert-many` with
`backend/experiments/expert_trajectories/example_dataset`. That dataset contains
only **12 train-split targets** and is insufficient for unseen-obstacle safety or
generalization claims. Do **not** train on combined `dataset.jsonl` or the
`reference`, `future_calibration` or `test` splits.

## Input and numerical conventions

The input is the independently validated `expert-target-v1` JSONL split. The
reader uses the generator's `frozen.json`, exact episode `group_id`, physical
`timestamps`, immutable `Scene` and obstacle predictions. Only `velocity_disc`
with `output=xy`, `N=96`, `dt=0.4` is supported by the default TOML. For a
different disc horizon change the model TOML **to match the generated dataset**.
Output targets are `(N+1,2)` in metres. For fixed-track generation the decoder
uses a forward centerline and a smooth cubic-spline residual; old rectangular
scenes use an endpoint interpolation baseline. The latent has four fixed endpoint
coordinates plus `2*min(knots,N-1)` normalized spline coefficients. Only the
coefficient coordinates are sampled from Gaussian noise and flow-matched.
Flow time `t in [0,1]` is artificial and distinct from physical step `k` and MPC
prediction index `i`.
The train-only normalization is freshly fitted after holding out full validation
groups, so even internal model selection cannot affect fitted statistics.

The scene encoder receives ego xy/velocity, goal, displacement, map bounds,
physical horizon/time-step/obstacle-count features and all obstacle positions at **every** horizon
index. Obstacle centres are represented relative to ego; DeepSets mean pooling
uses a mask, so padding and obstacle order do not change the embedding. The
separate guidance encoder uses signed passage side, speed scale and the live
selected obstacle's midpoint/radius. An invalid selected obstacle is an error.
The model is intentionally fixed-horizon. The current online `Context` needs
`features=(vx,vy,xmin,xmax,ymin,ymax)` for this model; the synthetic analytic
`conformal_mpc` context omits these and must not be silently substituted.

## Objective and learned baseline

For each training pair:

```text
q1 = normalized_spline_coefficients(y_ref_star, context)
q0 ~ N(0,I),  t ~ Uniform(0,1)
xt = [start_xy, goal_xy, (1-t)*q0 + t*q1]
u = [0, 0, q1-q0]
L = mean_elementwise(||drift(xt,t,c,a)[4:] - (q1-q0)||²)
```

Time/noise are independently sampled for every trajectory and batch. With the
configured probability, **both** context and guidance are replaced by learnable
NULL embeddings. This trains the same network's unconditional branch; the
`baseline(x,t)` used by fixed-probe conformal detection is genuinely learned,
not an arbitrary zero field. A deterministic spline decoder enforces endpoints,
but there is no hard obstacle, track-boundary or vehicle-dynamics certificate.
Training uses AdamW, global gradient clipping, cosine learning-rate
scheduling, a deterministic group holdout, a reproducible fixed-noise validation
loss, best-epoch selection, and early stopping. The checkpoint contains a
strict `state_dict`, fitted normalization buffers, model architecture, geometric
guidance semantics, dataset/source identifiers, group memberships and seeds.

**Determinism:** CPU is the default; fixed seeds and thread caps make repeated
runs comparable in the same runtime, not bitwise-identical across Torch/BLAS
versions or GPU hardware. A changed seed, warm-start policy, guidance catalogue,
solver, checkpoint or scene distribution requires recalibration downstream.

## Inference and integration

`run_flow_sampling(model, current_context, guidance, seed=..., steps=32,
solver="heun")` integrates the actual drift from 0 to 1 and denormalizes once.
`euler` is also supported. Passing `anchor` and `steering` integrates their
**mixed drift** at every step with the **same current context** and `x0`. This
is not interpolation of saved expert trajectories. Paired sampling reuses the
seed, and warm starts are explicitly a changed initializer distribution.

The new `TrajectoryFlowModel` directly implements the method contract expected
by `backend.src.control.model.TorchTrajectoryAdapter`: `encode`, `drift`,
`baseline`, `initialize`, `decode`, `noise`, `assert_frozen`. An example load:

```python
from pathlib import Path
from backend.src.control.model import TorchTrajectoryAdapter
from backend.src.trajectory_flow.train import load_flow_checkpoint

# Obtain the expected SHA-256 from a trusted training receipt, not untrusted input.
model = load_flow_checkpoint(Path(".cache/trained-flow"), expected_sha256=TRUSTED_SHA256)
adapter = TorchTrajectoryAdapter(model, model.revision)
# adapter is now ready for fixed probes, field mixing and calibration on fresh scenes.
```

The existing `control.sample.integrate_field` uses Euler for ODE integration;
`trajectory_flow.sample.run_flow_sampling` provides optional Heun. Deploying
Heun requires a corresponding frozen sampling policy in the conformal/MPC
orchestrator; do not calibrate with one sampler and deploy another. The control
pipeline's current nominal `guidance` string must be present in the checkpoint's
frozen catalogue. Safe anchor guidance must be reevaluated under the **current**
scene, not copied from a stored trajectory. `TorchTrajectoryAdapter` only handles
frozen inference; **do not** train through it.

The default linear MPC is a velocity-disc tracker, not a nonlinear bicycle MPC.
Bicycle expert targets (`xy_sincos`) are deliberately rejected. On-road dynamic
obstacle deployment additionally needs sensing, prediction uncertainty, a
physically justified tracking margin, hard deadline enforcement and independently
validated fallback. Neural training loss is **not** a collision-safety certificate.

## Files and tests

- `backend/src/trajectory_flow/model.py`: model, learned baseline, normalization.
- `data.py`: expert train-split ingestion and grouped validation split.
- `train.py`: optimizer, checkpoint receipt, strict weight-only reload.
- `sample.py`: deterministic Euler/Heun and paired drift-mixed generation.
- `evaluate.py`: validation ADE/FDE, geometric clearance, timing diagnostics.
- `backend/experiments/trajectory_flow/pipeline.py`: CLI orchestrator.

```bash
uv run pytest tests/unit/test_trajectory_flow.py \
  tests/integration/test_trajectory_flow_training.py -q
uv run ruff check backend/src/trajectory_flow \
  backend/experiments/trajectory_flow tests/unit/test_trajectory_flow.py \
  tests/integration/test_trajectory_flow_training.py
```

The packed expert dataset is intentionally tiny; use more independently sampled
scene groups and an untouched final-test set before making performance claims.
Subsequent detector calibration, repaired-output calibration and MPC evaluation
must operate on outputs of the frozen *learned* policy, never on expert labels.
