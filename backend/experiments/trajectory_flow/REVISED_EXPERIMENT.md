# Revised Silverstone nominal-only experiment

Base branch: `mpc_integration` at `86813daca3f2d02e89ed2cdee804ab2a209e2e38`.
These changes respond to the existing three-phase result: 33.39% Phase 2 and
45.65% Phase 3 open-loop collision rates, measured on *different* validation
scenes. That is not a controlled demonstration of forgetting or failed d*.

## 1. Diagnose before more training

First run the independent dataset audits (the generation receipts alone record
`validation.status=not_run`):

```bash
uv run python -m backend.experiments.expert_trajectories.pipeline validate \
  --output .cache/three-phase-centerline-nominal-20k
uv run python -m backend.experiments.expert_trajectories.pipeline validate \
  --output .cache/three-phase-dstar-nominal-48k
```

Pair **existing** Phase 2 and Phase 3 checkpoints on the same Phase 3 held-out
episodes, with the identical Heun-32 initialization seeds and reference labels:

```bash
uv run python -m backend.experiments.trajectory_flow.compare \
  --dataset .cache/three-phase-dstar-nominal-48k \
  --checkpoint .cache/three-phase-2-mixed-nominal-20k 56e472e713c35f179966432dc82f5ddb18a50b11604ac850a9a95e2b947220e9 \
  --checkpoint .cache/three-phase-3-dstar-nominal-48k 96ff0f5d2529d74392e953587e7f6faf973a2d06617501b06472949b4b17f443 \
  --output .cache/three-phase-cross-eval-original.json
```

The experiment reads **only `splits/train.jsonl`** and selects the same internal
validation groups as the original training (`seed=20261112`, holdout 20%).
It never fits, recalibrates, or uses `reference`, `future_calibration`, or `test`.
Compare collisions, ADE, obstacle-local ADE/max error (expert surface distance
within 0.5 m), and the 5th percentile/minimum predicted obstacle clearance.
Obstacle-local ADE pools physical timesteps in the selected window; it is not
an individual-trajectory safety guarantee. Both runs must use identical code,
sampling solver, precision, and hardware for meaningful latency comparisons.

## 2. Conservative Phase 3 with genuine replay

Phase 3 now uses 40 epochs, learning rate `2e-5`, patience 20. During each epoch,
20% of *accepted fitting targets* come from the original Phase 1 centerline
training partition; they are deterministically sampled without replacement.
The other 80% come from the original Phase 3 arbitrary/centerline mixture.
The original Phase 1 **validation** groups are excluded from replay; Phase 3
validation and reserved partitions are never trained on. The checkpoint receipt
records both frozen dataset IDs, replay group IDs, and the replay fraction.

Reuse the already generated 48k and 20k datasets rather than regenerating them:

```bash
uv run python -m backend.experiments.trajectory_flow.pipeline \
  --dataset .cache/three-phase-dstar-nominal-48k \
  --replay-dataset .cache/three-phase-centerline-nominal-20k \
  --replay-fraction 0.2 \
  --config backend/experiments/trajectory_flow/phase3_dstar.toml \
  --initial-checkpoint .cache/three-phase-2-mixed-nominal-20k \
  --initial-sha256 56e472e713c35f179966432dc82f5ddb18a50b11604ac850a9a95e2b947220e9 \
  --output .cache/three-phase-3-dstar-replay-40ep
```

Compare the new checkpoint with Phase 2 on **Phase 3 validation** using
`compare.py`, then repeat on **Phase 1 validation** to test retention. Supply
the new checkpoint SHA printed by the training receipt. Select using held-out
collision fraction and obstacle-local error, not FM MSE alone. The new run has
fewer total optimizer steps than the previous 100-epoch Phase 3; it is a
*conservative fine-tune candidate*, not a compute-matched ablation.

## 3. Higher-clearance target feasibility: pilot before full regeneration

The validated expert trajectories have approximately 0.14–0.15 m *mean* surface
clearance against a 0.12 m required margin. A larger margin may reduce the
exposure of generated paths to errors, but narrow-track geometry can make some
scenes unsatisfiable. Do **not** silently relabel or reject failures as unsafe.

```bash
uv run python -m backend.experiments.expert_trajectories.pipeline generate \
  --config backend/experiments/trajectory_flow/expert_clearance_pilot.toml \
  --output .cache/clearance-018-pilot-2k
uv run python -m backend.experiments.expert_trajectories.pipeline validate \
  --output .cache/clearance-018-pilot-2k
uv run python -m backend.experiments.expert_trajectories.pipeline summary \
  --output .cache/clearance-018-pilot-2k
```

This new 2,000-scene nominal-only dataset uses the same track, model, obstacle
family and arbitrary/centerline endpoint mixture, but `required_clearance=0.18`
metres and an independent scene seed. Its result is a **feasibility/acceptance
study only**. If adequate independent targets remain, generate a full 0.18 m
training dataset and align the training `clearance_margin_m` with that change in
a separately versioned experiment. Do not train with stricter labels on old
expert trajectories or combine different minimum-clearance definitions silently.

No open-loop metric is a physical collision guarantee. Deployed results still
require closed-loop MPC tests, fresh calibration, and dynamic-obstacle validation.
