# Verified expert generation runs

Run date: 2026-11-09. Windows 11, Intel Core i7-8550U (4 cores / 8 logical CPUs),
Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1. Two worker processes, one native thread
each. These are small functional demonstrations, not a throughput benchmark or
a representative safety study. Other repository tests ran on the same machine;
timings therefore include shared-machine load.

The delivered artifacts are [example_dataset](example_dataset/summary.json) and
[bicycle_dataset](bicycle_dataset/summary.json), approximately 0.91 MB and 0.26 MB
including all attempts, targets, split exports, validation and provenance.
Both completed the standalone `validate` command after generation. The default
run also completed `summary` and a one-worker `--resume` run: all 40 committed
attempts were reused, none rerun, with identical scientific targets.

| Measurement | Default disc | Bicycle speed profiles | Supplied scenes |
| --- | ---: | ---: | ---: |
| Scene records / initial rejections | 12 / 2 | 4 / 0 | 6 / 1 |
| Solver attempts / eligible attempts | 40 / 40 | 8 / 8 | 20 / 20 |
| Validated targets / eligible scene-guidance pairs | 20 / 20 | 8 / 8 | 10 / 10 |
| Attempt acceptance | 100% | 100% | 100% |
| Scenes yielding targets / submitted scene rows | 83.3% | 100% | 83.3% |
| Minimum surface clearance (m) | 0.1593421 | null: no obstacles | 0.1590029 |
| Mean minimum clearance (m) | 0.1656429 | null: no obstacles | 0.1663052 |
| Maximum goal error (m) | 0.0010603 | 0.0031289 | 0.0007620 |
| Mean goal error (m) | 0.0008180 | 0.0022128 | 0.0006942 |
| Maximum constraint violation | 5.32e-9 | 7.29e-16 | 3.78e-9 |
| Dynamics residual | 0 | 0 | 0 |
| Scenes with distinct alternatives | 10 | 4 | 5 |
| Minimum pairwise positional RMS (m) | 0.5824613 | 0.1197453 | 0.5756237 |
| Mean pairwise positional RMS (m) | 0.6323518 | 0.1242096 | 0.5891181 |
| Generation wall time (s) | 53.629 | 20.887 | 26.770 |
| Sum of optimizer times (s) | 45.993 | 28.783 | 22.950 |
| Mean optimizer time (s) | 1.150 | 3.598 | 1.148 |
| Worker p95 time incl. startup (s) | 3.267 | 6.303 | 3.379 |

Generation wall time covers the worker/receipt stage, excluding preparation,
export and final audit. Worker/solver sums overlap across processes. Residual
maxima aggregate mixed physical units; named per-constraint values are stored
for interpretation. All accepted clearances exceed the required 0.12 m, and all
accepted goal errors are below 0.08 m. No accepted reference collision was found
under the declared footprint/interpolation model. These are not executed-rollout
collision measurements.

The default run records three static and nine moving obstacle paths, including
the two deliberately invalid initial scenes (`scene-0000` and `scene-0005`).
Both were rejected for `initial_obstacle_clearance`. Valid scenes yield ten left
and ten right targets. Group allocations are 6/2/2/2 for
train/reference/future_calibration/test; accepted target counts are 12/2/4/2.
Rejection did not trigger group reassignment. The maximum speed is
1.5000000027 m/s, within the explicit numerical validation tolerance of the
1.5 m/s constraint. Independent recomputation found zero discrete dynamics
residual in both delivered datasets.

The bicycle run yields four conservative and four aggressive trajectories, with
one independent group and two targets per split. Maximum conservative speed is
0.6000000000 m/s; maximum aggressive speed is 0.7689639240 m/s. The two modes use
speed caps of 0.6 and 1.5 m/s, respectively; the objective need not saturate the
higher cap. All four pairs exceed the 0.08 m diversity threshold. They are straight
speed-profile examples without obstacles, so no obstacle-clearance claim is made.
Turning, crowded scenes and physical integration-error bounds remain unverified.

The supplied-scene run is reproducible from `user_scenes.json`; output is retained
locally at `.cache/expert-user-scenes`. It completed standalone validation. Its
original scene, related goal variant and rejected out-of-map variant remain in
one group (`future_calibration` in this run). The row-level rejection is recorded
without leaking its related accepted targets into another split. This dataset
is not duplicated in the delivery bundle.

Scientific identities:

- Default: `0957c9689afd11a2ffa5a23f1e0f7bab64c692d276cf721bf2d9c566930ac31d`
- Bicycle: `8a29e6e2516ef2e67c40d5e7b586677d204939aaaf0da1f69d8eb8dc7d77880e`
- Supplied: `cfda99ac0a802dccf35de8d0c50bbd91f7406f51954d3abfd8625e4994a46ef1`

Each `frozen.json` records all source and input/config hashes and runtime versions.
Timestamps and numerical targets are deterministic for these inputs and runtime;
latencies and finite-deadline outcomes are not universal constants. Source files
must remain unchanged during a run. A development test run hit
its provenance guard when edits overlapped generation; subsequent verification
used frozen source files, without weakening that check.

Final verification: `python -m pytest -q` passed **261 tests**, with **one skip**
because CUDA was unavailable (156.68 s). This includes 28 expert-generator tests
and the existing detection, repair and control checks. Repository-wide Ruff
passed and `git diff --check` reported no whitespace errors. The new tests use
actual CPU optimization, subprocess termination, file locks and artifact replay;
the reproducibility comparison permits `1e-10` absolute numerical differences
across one and two workers in the same environment.

Exact generation commands (use the Windows interpreter substitution in README
if the virtualenv launcher is blocked):

```powershell
uv run python -m backend.experiments.expert_trajectories.pipeline generate --config backend/experiments/expert_trajectories/config.toml --output backend/experiments/expert_trajectories/example_dataset
uv run python -m backend.experiments.expert_trajectories.pipeline validate --output backend/experiments/expert_trajectories/example_dataset
uv run python -m backend.experiments.expert_trajectories.pipeline summary --output backend/experiments/expert_trajectories/example_dataset
uv run python -m backend.experiments.expert_trajectories.pipeline generate --config backend/experiments/expert_trajectories/config.toml --output backend/experiments/expert_trajectories/example_dataset --resume --workers 1
uv run python -m backend.experiments.expert_trajectories.pipeline generate --config backend/experiments/expert_trajectories/bicycle.toml --output backend/experiments/expert_trajectories/bicycle_dataset
uv run python -m backend.experiments.expert_trajectories.pipeline validate --output backend/experiments/expert_trajectories/bicycle_dataset
uv run python -m backend.experiments.expert_trajectories.pipeline generate --config backend/experiments/expert_trajectories/config.toml --scenes backend/experiments/expert_trajectories/user_scenes.json --output .cache/expert-user-scenes
uv run python -m backend.experiments.expert_trajectories.pipeline validate --output .cache/expert-user-scenes
```

The packaged directories already exist: choose fresh output directories to
reproduce from scratch, or use matching `--resume` to verify/reuse them.

Constrained local optimization does not establish global optimality. All targets
were independently checked, but independent checking still shares the declared
dynamics and circular geometry assumptions. No training quality, deployed
diffusion coverage, executed tracking safety or closed-loop guarantee follows
from these expert examples. The held-out groups are reservations for later
proper evaluation, not a substitute for fresh deployed-output calibration.
