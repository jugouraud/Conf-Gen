# Offline expert trajectories

**Current long-route profile:** all newly generated scenes use the same fixed,
Silverstone-inspired closed circuit (a stylized geometry, not surveyed F1 coordinates).
Its roughly 267 m centerline has distinct fast sections and corner sequences;
start and goal positions are sampled on the centerline, with forward goals 7-25 m ahead
(wrapping at the finish line), and one circular
obstacle strictly between them. Obstacle longitudinal fraction is 0.2-0.8,
its lateral offset is within +/-0.1 m and its radius is 0.14-0.18 m. It stays
stationary during that scene. The 50,000-scene training preset uses the same
distribution and one unconstrained-side guidance label; the route search compares
both passing sides. These are forward (increasing arc-length) routes, not shortest
travel in either direction around a two-way track.

The velocity-disc planner tests bounded Gaussian lateral detours (three widths,
45 amplitudes, and both sides) and ranks them by physical polyline length. The
first independently validated candidate becomes `y_ref_star`. This is the
shortest **within the sampled family**, not a certified global continuous
shortest path. Without an obstacle the reference follows the centerline, with
`q` the normalized physical horizon index. With obstacles, bounded Gaussian
lateral detours are added locally to the centerline, and the candidates are validated
independently. The validator still checks speed, acceleration, footprint,
swept obstacle clearance, track containment, and terminal requirements. One
attempt is recorded per scene, including its random obstacle. The reference
horizon is 96 steps at 0.4 s (38.4 s); the flow-training preset also uses
H=96. The circuit's new `layout = "silverstone"` is frozen in each scene;
missing `layout` in older datasets still means ellipse. The enlarged map uses
`radii = [32, 36]`, 0.7 m half-widths and +/-40 m velocity-disc state bounds.
Regenerate datasets and retrain checkpoints to use the new circuit; old frozen
artifacts must not be reinterpreted as Silverstone data. Older H=16 datasets and
checkpoints likewise need regeneration before use with the current H=96 policy.

Procedural sampling runs in-process to avoid one Python worker startup per
reference. Optimization mode and optional optimizer fallback keep killable
subprocesses. All existing dataset hashes, split independence, validation,
and atomic receipts still apply. Existing example datasets belong to their
original source revision and must not be silently reused as new targets.

Only validated `(observation, guidance, y_ref_star)` are used for vector-field
training. Absolute `(x,y)` already identifies progress on this *fixed* circuit;
obstacles remain in the existing context. The xy output retains shape
(H+1, 2), but now H=96. This does **not** give the trained field an arbitrary-track conditioning interface.
Procedural generation still supports velocity-disc only; optimization remains
available for bicycle models. The more general optimizer/legacy-map details
below apply to that opt-in mode rather than the fixed-route preset.

Generate `(observation, guidance, y_ref_star)` with constrained direct-shooting
trajectory optimization. The expert receives the scene and goal directly. It
does not call diffusion or track a diffusion-generated reference. General stages
live in `backend/src/expert`; this directory owns scene distributions and examples.
The default reuses the online MPC's `LinearDynamics`, circular swept geometry and
SciPy SLSQP backend. No checkpoint or new dependency is required.

## Commands

From the repository root, with Python 3.12 and the locked environment:

```powershell
uv sync --locked
uv run python -m backend.experiments.expert_trajectories.pipeline generate --config backend/experiments/expert_trajectories/config.toml --output .cache/expert-demo
uv run python -m backend.experiments.expert_trajectories.pipeline validate --output .cache/expert-demo
uv run python -m backend.experiments.expert_trajectories.pipeline summary --output .cache/expert-demo
uv run python -m backend.experiments.expert_trajectories.pipeline generate --config backend/experiments/expert_trajectories/config.toml --output .cache/expert-demo --resume --workers 1
```

Generation exports its per-attempt validation results without automatically rerunning
the complete dataset audit. Run `validate` explicitly before using a dataset for
training or evaluation.

Use a fresh directory for changed source, inputs or scientific configuration.
Resume verifies completed artifacts before writing, retains committed attempts,
and reruns only attempts whose receipts were never committed. Worker count can
change; seeds and target selection do not depend on completion order. A dataset
OS lock rejects concurrent coordinators. Each attempt has a cooperative solver
deadline and a parent process timeout of `solver_timeout + startup_timeout`;
timeout terminates the actual worker. Retries are bounded and only apply to
failed guesses. Generate returns nonzero if no training targets survived. Such a
completed failure dataset remains inspectable with validate/summary.

Supplied scenes and the optional steering model:

```powershell
uv run python -m backend.experiments.expert_trajectories.pipeline generate --config backend/experiments/expert_trajectories/config.toml --scenes backend/experiments/expert_trajectories/user_scenes.json --output .cache/expert-user-scenes
uv run python -m backend.experiments.expert_trajectories.pipeline generate --config backend/experiments/expert_trajectories/bicycle.toml --output .cache/expert-bicycle
uv run python -m backend.experiments.expert_trajectories.pipeline validate --output backend/experiments/expert_trajectories/example_dataset
uv run python -m backend.experiments.expert_trajectories.pipeline validate --output backend/experiments/expert_trajectories/bicycle_dataset
```

`user_scenes.json` demonstrates an original scene, a related goal variant and a
rejected initial condition in the same episode, plus three independent groups.
No supplied scene is resampled or silently retimed. All obstacle predictions must
cover the entire requested physical horizon. At least four independent groups
are required to reserve all four splits.

On the delivery Windows machine, Application Control blocks the virtualenv
launcher. These commands were run with the same installed packages using:

```powershell
$env:PYTHONPATH = "$PWD/.venv/Lib/site-packages"
& C:/Users/Poseidon/AppData/Roaming/uv/python/cpython-3.12.15-windows-x86_64-none/python.exe -m backend.experiments.expert_trajectories.pipeline validate --output backend/experiments/expert_trajectories/example_dataset
```

## Defaults and numerical meaning

`config.toml` explicitly lists every default. The dataset seed is 20261109;
12 independent episodes sample rectangular maps, start/goal positions and one
static or piecewise-linear moving circular obstacle. The map extends from
`x=-1` by the sampled width, with the sampled height centered on `y=0`.
Absolute start times are sampled from 100 to 200 seconds. The configured
`invalid_initial_fraction=0.15` deliberately includes rejection cases, without
resampling them away. This fraction is a probability, not an exact count.

The default vehicle has state/output `[x,y]`, control `[vx,vy]`, and exact
`s[i+1]=s[i]+dt*u[i]`. Footprint radius is 0.15 m, speed limit 1.5 m/s, discrete
command slew limit 1.5 m/s², state box ±10 m, input box ±2 m/s. The acceleration
limit means `norm(u[i]-u[i-1])/dt`; instantaneous velocity changes in this model
are not a physical acceleration certificate. Steering is inapplicable to the
disc, and its diagnostics are null. Do not interpret it as an Ackermann car.

The optional bicycle uses `[x,y,yaw,v,steering]` and `[acceleration,steering_rate]`:
`xdot=v*cos(yaw)`, `ydot=v*sin(yaw)`, `yawdot=v*tan(steering)/wheelbase`,
`vdot=acceleration`, `steeringdot=steering_rate`. Controls are held constant over
each physical interval; four RK4 substeps define `F`. Speed is nonnegative;
configured bounds also constrain steering, steering rate and acceleration.
State yaw stays unwrapped. The example exports `[x,y,sin(yaw),cos(yaw)]`, not an
interpolation across wrapped angles. Its footprint radius 0.65 m is a configured
enclosing disc, not a fitted vehicle body. The online controller currently
supports linear dynamics; a bicycle dataset requires a matching future planner
adapter and controller before deployment.

Default horizon is `N=16`, `dt=0.4 s` (6.4 s). Every exported target has exactly
`(N+1,n_y)` samples at `start_time+i*dt`; `i` is the planning horizon index, not
diffusion time or the online physical iteration `k`. Retiming is deliberately
unsupported: change the scene prediction coverage/config and optimize again.
Positions use metres, angles radians, and CPU arrays FP64. No angle interpolation
is used for collision geometry; the position projection is checked separately.

The expert minimizes

```text
goal_weight*dt*sum_i ||position[i]-goal||²
+ terminal_weight*||position[N]-goal||²
+ effort_weight*dt*sum_i ||u[i]||²
+ smoothness_weight/dt*sum_i ||u[i]-u[i-1]||².
```

Position cost uses `i=1..N`; control costs use `i=0..N-1`. Prior control is initial
velocity for the disc and zero acceleration/steering-rate for the bicycle. Weights
are 0.2, 30, 0.1 and 0.2. Terminal position error must be ≤0.08 m and terminal
speed ≤0.08 m/s. Optional heading tolerance is 0.1 rad. `terminal_mode="progress"`
instead requires initial goal distance minus final goal distance ≥ configured
`minimum_progress` (default 1 m); terminal speed still applies. Neither mode
guarantees global optimality. Solver failure does not prove scene infeasibility.

SLSQP uses up to 160 iterations and `ftol=1e-8`; independent acceptance tolerance
is `1e-6`. Default guesses have lateral offsets 0.9 and 1.5 m, no retries, a 10 s
solver deadline, two workers and one native thread each. Retry perturbation std
is 0.03 in control units. At most 16 workers, 8 guesses, 4 retries, 128 intervals,
50,000 scene rows and 50,000 attempts are allowed. `workers*threads_per_worker`
must fit the machine's logical CPU count. No multi-node scaling is claimed.

## Alternatives and independent validation

Left/right guidance constrains signed lateral position when the vehicle crosses
the selected obstacle's moving plane along the initial-to-goal axis. It requires
footprint + obstacle radius + required clearance + 0.05 m passage extra, at every
such crossing. The exact same scene/obstacle trajectories are reused. Speed
guidance scales the vehicle speed cap; the bicycle example uses 0.4 and 1.0.
Names alone do not establish distinct solutions. Export chooses the minimum-cost
validated guess for each guidance and removes later choices whose time-aligned
positional RMS differs by less than 0.08 m from an already retained maneuver.
Guidance order resolves such duplicates; all attempts remain in the manifest.

Only independently validated targets are exported. Checks recompute start,
dynamics residuals, state/input/speed/acceleration/steering constraints, timestamps,
output map, terminal conditions, guidance and footprint clearance. Convergence
alone never grants acceptance. Failed, timed-out and duplicate attempts are retained
separately. Raw solver iterates are retained when available, even if invalid.

For the disc, exact synchronized swept-segment checks cover both optimizer knots
and every obstacle time breakpoint. The validator catches collisions between
waypoints and obstacle turns between optimizer knots. Required surface clearance
is 0.12 m. Rectangle boundaries are inset by footprint radius. For the bicycle,
the validator checks both the exported straight reference segments and dense
integrated positions. It subtracts the chord bound
`(max_acceleration + max_speed²*tan(max_steering)/wheelbase)*(dt/substeps)²/8`
plus configured numerical margin (0.002 m). This is conservative under the stated
model and bounded derivatives; the configured RK4 error margin is not proven.
Arbitrary shapes, uncertain obstacle forecasts, tire slip and executed tracking
error are outside this offline model. The bicycle demo exercises straight speed
profiles, not a validated catalogue of turning maneuvers in traffic.

## Data use and reproducibility

See [the schema](../../../docs/expert-dataset-schema.md) and
[measured results](RESULTS.md). Train from **`splits/train.jsonl`**, not the combined
inspection file `dataset.jsonl`. Independent groups are assigned before alternatives
with weights 1/2, 1/6, 1/6, 1/6, largest-remainder allocation, and at least one
group in each split. Related supplied variants must have the same true `group_id`;
the software cannot discover falsely declared independence. Downstream windows
and augmentations must inherit these assignments.

Normalization uses accepted training targets only, population standard deviation,
and a `1e-8` scale floor. Observation vectors are state features (bicycle heading
as sin/cos), goal x/y, initial velocity x/y. Obstacles, maps, timestamps and guidance
remain raw. Per-dimension reference/control moments pool training time samples.
Statistics are optional; persisted targets remain in physical units.

Frozen manifests bind source hashes, lockfile, Python/NumPy/SciPy/platform, full
configuration, sampling distribution, all raw scenes, groups and deterministic
seeds. Validation refuses changed source files. Use the recorded revision to
audit old data. These hashes detect accidental changes, not malicious forgery.
Numerical reproducibility is tested at `1e-10` across worker counts in one
environment; wall-clock measurements and deadline outcomes depend on load.
Atomic replacement and the lock protect process-interruption recovery, not an
fsync-backed power-loss transaction. Keep backups for valuable long runs.

These are expert training labels. The reserved `future_calibration` scenes must
later be evaluated using outputs of the frozen deployed diffusion/repair policy.
Expert clearances are not conformal calibration scores for that policy and do
not imply released or closed-loop safety. This change does not train a model.

## Verification

```powershell
uv run python -m pytest tests/unit/test_expert_planning.py tests/contract/test_expert_artifacts.py tests/e2e/test_expert_dataset.py -q
uv run python -m pytest -q
uv run ruff check backend/src backend/runtime backend/experiments tests docs/five-category-nudity/figures/plot_steering_schedules.py
```

Tests use real SciPy optimization and subprocess/filesystem boundaries. They cover
timed moving obstacles, inter-waypoint collisions, invalid limits/dimensions,
initial rejection, guidance, bicycle angle/dynamics checks, data separation,
normalization, deterministic parallel execution, hard/cooperative timeouts,
bounded solver retries, incomplete recovery and completed-artifact corruption.
