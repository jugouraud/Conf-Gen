# Conditional trajectory flow matching (velocity-disc research implementation)

This adds a **new, randomly initialized** PyTorch spline-trajectory vector field.
The flow transports Gaussian spline coefficients, with deterministic endpoint
metadata. It decodes them relative to the forward Silverstone d* reference; it never
samples independent Gaussian noise for all 97 physical waypoints.
No SD3/image checkpoint is loaded or reused. The supervised source is the existing
[`expert_trajectories`](../expert_trajectories/README.md) generator, which creates
validated `(observation, guidance, y_ref_star)` tuples using SciPy constrained
shooting. The trajectory network starts **untrained** and must be trained before
any detector calibration or MPC reference release. Neither generator training
nor successful sampling certifies obstacle avoidance or tracking safety.

## Three-phase Silverstone experiment (fresh weights, then verified transfers)

The decoder must use `d*` in **all** phases; on the centerline this is the
zero-offset special case. Changing the decoder invalidates historical model
source fingerprints, so **do not** warm-start from the previous centerline
checkpoints. Generate a new centerline dataset and Phase 1 model with this code.
All outputs below must be new directories. The orchestrator uses generation's
per-target checks; run the full standalone dataset audit explicitly when needed.

Run all three phases, including the verified handoffs and optional Phase 3
visualization, with:

```bash
uv run --with-requirements frontend/requirements.txt \
  python -m backend.experiments.trajectory_flow.orchestrator
```

Use `--no-browser` for a non-interactive run. The explicit commands below add a
full standalone audit and are useful when inspecting a completed phase before
starting the next one.

The orchestrator defaults Phase 1 to `.cache/three-phase-centerline-nominal-20k`,
so it does not reuse the interrupted `.cache/three-phase-centerline` generation.

```bash
# Phase 1: centerline-only FM, 20,000 nominal scenes / attempts.
uv run python -m backend.experiments.expert_trajectories.pipeline generate \
  --config backend/experiments/trajectory_flow/expert_training.toml \
  --output .cache/three-phase-centerline
uv run python -m backend.experiments.expert_trajectories.pipeline validate \
  --output .cache/three-phase-centerline
uv run python -m backend.experiments.trajectory_flow.pipeline \
  --dataset .cache/three-phase-centerline \
  --config backend/experiments/trajectory_flow/phase1_fm.toml \
  --output .cache/three-phase-1-fm

# Copy the trusted sha256 printed by the completed training receipt (64 hex chars).
PHASE1_SHA="PASTE_64_HEX_SHA_FROM_PHASE1_RECEIPT"
# Phase 2: same scenes, continue from Phase 1; fresh optimizer and lower LR.
uv run python -m backend.experiments.trajectory_flow.pipeline \
  --dataset .cache/three-phase-centerline \
  --config backend/experiments/trajectory_flow/phase2_mixed.toml \
  --initial-checkpoint .cache/three-phase-1-fm --initial-sha256 "$PHASE1_SHA" \
  --output .cache/three-phase-2-mixed

# Phase 3: 48,000 nominal scenes / attempts (below the current 50k cap).
# 80% arbitrary feasible lateral endpoints, 20% centerline *before acceptance*.
uv run python -m backend.experiments.expert_trajectories.pipeline generate \
  --config backend/experiments/trajectory_flow/expert_phase3_dstar.toml \
  --output .cache/three-phase-dstar
uv run python -m backend.experiments.expert_trajectories.pipeline validate \
  --output .cache/three-phase-dstar
PHASE2_SHA="PASTE_64_HEX_SHA_FROM_PHASE2_RECEIPT"
uv run python -m backend.experiments.trajectory_flow.pipeline \
  --dataset .cache/three-phase-dstar \
  --config backend/experiments/trajectory_flow/phase3_dstar.toml \
  --initial-checkpoint .cache/three-phase-2-mixed --initial-sha256 "$PHASE2_SHA" \
  --output .cache/three-phase-3-dstar
```

Only generator `train` groups enter fitting and internal validation; the
`reference`, `future_calibration`, and `test` groups remain reserved. Phase 3
now replays 20% **actual Phase 1 fitting targets** alongside its Phase 3 fit
partition, with a shorter, lower-LR fine-tune. No Phase 1 validation target
enters replay. The Phase 2 weights and normalization coordinate system are
preserved; each stage has a fresh optimizer and a SHA-verified handoff. Do not
compare runs without controlling optimizer updates and evaluation scenes.

Read `evaluation.json` for per-regime ADE/collision diagnostics (centerline
versus arbitrary endpoints), in addition to aggregate metrics. The 20% figure
is the **pre-validation scene sampling rate**; actual accepted-target proportions
can differ. This experiment uses stationary obstacles and velocity-disc dynamics;
it does not establish dynamic-obstacle handling or safety at deployment. For a
stage-3 performance comparison, also train a mixed-loss model from scratch on
exactly the same Phase 3 dataset and evaluate with the same sampler and seed.

See [REVISED_EXPERIMENT.md](REVISED_EXPERIMENT.md) for paired checkpoint
comparisons, the Phase 3 replay command, obstacle-local diagnostics, and the
higher-clearance feasibility pilot. Never use a Phase 3 validation result as a
claim about the reserved `test` or conformal-calibration partitions.

## Generate a larger training dataset

From repository root using Python 3.12 and `uv sync --locked`:

```bash
# The configured 20,000 scenes request one nominal avoidance trajectory each.
# One attempt per scene produces at most 20,000 trajectories.
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

To generate the configured Silverstone dataset, train its vector field, and
launch the local browser in one command, use:

```bash
uv run --with-requirements frontend/requirements.txt \
  python -m backend.experiments.trajectory_flow.orchestrator
```

The default outputs are new-only `.cache/silverstone-matched-20k` and
`.cache/trained-silverstone-matched-20k`; choose different `--*-output` paths
for another run. Use `--no-browser` when running without a local UI.

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
Output targets are `(N+1,2)` in metres. The fixed-track decoder uses the shared
forward d* baseline (minimum-jerk interpolation of endpoint lateral offsets)
and a cubic-spline residual. For centerline endpoints d* reduces to the centerline;
rectangular scenes use an endpoint interpolation baseline. The latent has four fixed endpoint
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

The nominal regime uses the same scene distribution (start, goal, obstacle,
track) with `side=0`: the shortest safe candidate across both signed passage
sides. This is **not** an unsafe label; it is unconstrained passage subject to
the same independent collision and track-containment validation.

For each training pair:

```text
q1 = normalized_spline_coefficients(y_ref_star, context)
q0 ~ N(0,I),  t ~ Uniform(0,1)
xt = [start_xy, goal_xy, (1-t)*q0 + t*q1]
u = [0, 0, q1-q0]
L = mean_elementwise(||drift(xt,t,c,a)[4:] - (q1-q0)||²)
```

For a conditioned (non-dropped) example the auxiliary objective adds
`focus_weight * local_path_MSE + clearance_weight * swept_clearance_hinge`,
using the differentiable one-step estimate `q1_hat = qt + (1-t)*v_theta`.
`local_path_MSE` weights physical XY reconstruction by Gaussian proximity to
obstacles along the expert route (scale `focus_sigma_m`). The clearance hinge
uses closest approach of relative vehicle/obstacle segments, obstacle and car
radii, and `clearance_margin_m`. The FM term remains evaluated on every sample;
auxiliary terms are **excluded** from classifier-free NULL-conditioning batches
to avoid forcing incompatible left/right targets into the unconditional branch.
A sampled reference can still be unsafe: this is neither exact trajectory
integration nor a hard road-boundary/vehicle-dynamics constraint. Ablate the
weights and monitor collision rate, ADE, per-side success and spline error.

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
