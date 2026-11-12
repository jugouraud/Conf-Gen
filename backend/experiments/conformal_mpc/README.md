# Conformal diffusion–MPC demonstrator

This is a runnable **synthetic** motion-planning experiment integrated with the
repository's fixed-probe geometry, finite-sample conformal quantile and existing
smoothstep repair schedules. It is not a trained diffusion planner. Inspection
found SD3 image adapters and image experiment artifacts, but no trajectory
checkpoint, vehicle dynamics or MPC implementation suitable for motion control.

General control stages are in `backend/src/control`; the standalone linear MPC
solver and collision geometry are in `backend/src/mpc`. Only scene generation,
the analytic model, catalogue rules, configuration and evaluation are here.
Existing image pipelines, distance definitions and repair profiles are unchanged.
See [design decisions](../../../docs/conformal-mpc-design.md) and
[observed results](RESULTS.md).

## Run

From the repository root with Python 3.12+:

```powershell
uv sync --locked
uv run python -m backend.experiments.conformal_mpc.pipeline --output .cache/conformal-mpc-run
uv run pytest tests/unit/test_conformal_control.py tests/integration/test_control_pipeline.py tests/contract/test_trajectory_adapter.py tests/e2e/test_conformal_mpc.py
uv run python -m backend.testing
uv run ruff check backend/src backend/runtime backend/experiments tests docs/five-category-nudity/figures/plot_steering_schedules.py
```

Use a new output directory for every run. The default `config.toml` fixes 48
reference, 48 detector-calibration, 20 output-calibration and 12 test episodes,
with at most 24 physical steps each. Running the four policies takes a few minutes
on the tested CPU. `--config PATH` selects an alternative complete configuration;
unknown/missing fields are rejected. `sampling_method="sde"` enables
Euler–Maruyama; `warm_start="shifted"` enables model-owned initialization from a
shifted previous reference. Such changes define a new policy and require a new
complete calibration run. The CLI limits Torch and native BLAS thread pools.

On the tested Windows machine the `.venv` Python launcher was blocked by
Application Control. Tests ran with the installed Python 3.12.15 interpreter and
`PYTHONPATH` pointing to `.venv/Lib/site-packages`; no security setting was changed.
The equivalent command used in this checkout was:

```powershell
$env:PYTHONPATH = "$PWD/.venv/Lib/site-packages"
& "$env:APPDATA/uv/python/cpython-3.12.15-windows-x86_64-none/python.exe" -m backend.experiments.conformal_mpc.pipeline --output .cache/conformal-mpc-run
```

## Architecture and contracts

```text
observation at physical k
  -> encode immutable context c=(state, obstacle predictions, goal, features)
  -> signature at fixed (x_j,t_j) -> joint detector
  -> nearest compatible safe guidance, finite schedule search
  -> integrate selected mixed drift -> decode y_ref[i]
  -> swept reference validation -> constrained tracking MPC
  -> validated, feasible, on-time control u[k], or explicit abstention
```

`k` is the physical replanning step; `i=0..H` indexes physical predictions at
`k*dt+i*dt`; `t` is noise-to-output diffusion time in `[0,1]`. They never share a
grid. References have shape `(H+1,2)` in metres. The default discrete dynamics are
`z[i+1]=z[i]+dt*u[i]`, with position state in metres and velocity control in m/s.
The general solver accepts arbitrary finite linear `A,B,C`, where `C*z` is 2D
position, positive definite tracking/input costs, state/input bounds, and physical
`dt`. It assumes piecewise-linear position motion between samples. It does not
claim to implement nonlinear bicycle dynamics.
Drift has latent-coordinate units per unit of diffusion time, distinct from
physical velocity in m/s. Only the synthetic model uses metre-valued latent
coordinates; a real adapter must preserve its checkpoint's native latent units.

`solve_tracking(initial, reference, obstacles, config, check_deadline=...)` is a
standalone API without diffusion dependencies. It minimizes the sum of position
tracking costs at `i=0..H` and input costs at `i=0..H-1`. State/input bounds and
swept obstacle separation are constraints. The implementation uses local
[SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html),
then recomputes trajectories and checks all residuals independently of the solver
success flag. A failed solve is not a proof that the problem is infeasible.

Every workflow file has one public `run_<step>()` entry point. The online
`run_control_pipeline(...)` composes encoding, detection, repair, sampling,
reference validation, tracking, optional fallback and release. Stage records
carry explicit failures; they expose no control on abstention. The offline
experiment orchestrator composes preparation, probe construction, detector
calibration, policy freezing, output-calibration simulations, output calibration,
test simulations and reporting.

## Detection and repair mathematics

The model is frozen before reference/probe construction or calibration. Probes
are states of reference-episode ODE trajectories at fixed times `t_j<1`. Every
condition uses the identical probe arrays:

```text
tau(c,a)[j] = drift(x_j,t_j,c,a) - baseline(x_j,t_j)
rho(tau,sigma) = ||tau-sigma||_F / sqrt(J)
k_l = ceil((n_l+1)*(1-alpha_l))
r_l = sorted(class_l_scores)[k_l-1], or +inf if k_l>n_l
flag = (d_safe > r_safe) OR (d_unsafe <= r_unsafe)
```

Class scores measure distance to their own class bank. Nearest aggregation is
default. `mean` uses all class references; `exponential` uses the repository's
normalized Gaussian weights `exp(-d²/(2*bandwidth²))`, with bandwidth fixed in
config before calibration. There is no unsafe-radius cap. An infinite unsafe
radius flags every finite signature; this behavior is preserved, not regularized
away. An absent calibration class has radius `+inf` by the same rank rule;
reference banks must still contain both classes.

The frozen catalogue contains all reference episode/guidance entries in stable
order. Labels come from independently checking sampled physical reference
clearance, never from detector scores or guidance names. The synthetic catalogue
has `direct`, `upper`, and `lower` guidance. Safe means safe in its labelled
reference context; it is not a promise about every future context.

For a flagged query, safe entries with a matching physical/latent compatibility
tag are reevaluated **under the current context**. The nearest such field signature
selects the anchor; ties use catalogue order. Saved paths and stored contextual
signatures are never used as replacement anchor drift fields. If this nearest
anchor has no feasible schedule, the result is failure, not a search for a
different anchor using test outcomes.

```text
lambda(t) = kappa*s(t), 0 <= lambda <= 1
b_repaired = (1-lambda)*b_nominal + lambda*b_anchor
tau_repaired[j] = (1-lambda(t_j))*tau_nominal[j] + lambda(t_j)*tau_anchor[j]
```

Every configured candidate is scored by the same joint detector, without assuming
monotone acceptance. Among accepted candidates the exact integral of lambda is
minimized; ties use configured order. This is a minimum over a finite grid, not a
global continuous optimum. Existing `SteeringSchedule` smoothstep profiles and
their exact integrals are reused; the demo grid contains R1 constant and R2 fixed
early/middle/late windows. Other existing profiles can be supplied directly to
the general API. Unflagged queries use zero steering.

Reference generation integrates the selected mixed **field**, then decodes once.
Context and the model's diffusion coefficient are unchanged. Euler/Maruyama use
the same fixed time grid, initializer seed and Brownian draws across paired
methods. Knots do not alter the sampling grid. The default is cold initialization;
warm initialization is an explicit model contract at `t=0`, with a shifted old
path as input. MPC always starts from the same reference-derived control guess
protocol, without hidden cross-method solver state.

## Release, geometry and failures

The validator subtracts the disc vehicle radius and obstacle radius from centre
distance. It analytically minimizes relative position along each simultaneous
linear segment, catching inter-waypoint crossings with moving obstacles. Release
requires physical surface clearance at least `required_clearance +
tracking_uncertainty`, a feasible MPC solution, and the common physical-step
deadline. Numerical solver constraints allow the configured residual tolerance.
The uncertainty margin is an assumption, not an estimated probabilistic tube.
The simulated process disturbance is Gaussian and therefore not bounded by that
margin with certainty.

Missing anchors, infeasible finite searches, invalid model outputs, failed solves
and timeouts have explicit statuses. Old references must match episode, step,
horizon, dt, goal and compatibility; they are shifted, terminal-padded, checked against
the current state and current obstacle predictions, and solved again. Their old
certificate is never reused. All fallback work shares the original deadline.
If no reference can be released, `control=None`. The simulator applies zero
velocity plus the paired disturbance; this is not an invariant safe controller.

Deadline checks are cooperative. They prevent late control release but cannot
preempt a blocking native model/solver call. Hard real-time deployment requires
supervision and a platform-specific safe control policy outside this research
implementation. There is no closed-loop invariance or recursive-feasibility proof.
The demo's 2-second wall-clock computation budget is separate from its 0.4-second
logical simulation timestep: the simulator advances physical time after each
solve. These settings do not demonstrate a real-time 2.5 Hz controller. A deployed
controller must align its budget and prediction timestamps with the actuator's
actual schedule, and account for sensing/actuation delay.

## Real-model adapter

Implement the `TrajectoryModel` contract, or supply a trusted local Torch module
to `TorchTrajectoryAdapter.from_checkpoint(module, path, expected_sha256)`.
The expected digest must come from a trusted source. Loading uses
`weights_only=True`, strict state-dictionary matching, `eval()`, disabled parameter
gradients, inference mode, and checks for parameter/buffer version changes.
The adapter does not download models or execute checkpoint-provided Python.

The module provides `encode(obs)->Context`, `drift(x,t,c,a)`, `baseline(x,t)`,
`decode(x_final)->(H+1,2)`, `initialize(c,rng,warm)`, and `noise(x,t)`.
Torch latent inputs/outputs are converted at the adapter boundary. Encoding may
add immutable numeric features but must preserve the physical context. Latent
shape, normalization, time transformation and guidance interpretation belong to
the module. Noise is a diagonal elementwise coefficient with the latent shape;
correlated-noise models need an extended sampler contract. Initializers must use
the supplied generator and inference must not use hidden randomness or mutate
buffers, parameters, `.data`, context, or other hidden state. Sampling is exposed
through `run_reference_sampling` / `integrate_field`.

Create independent training/reference, detector-calibration, output-calibration
and test scenes for the real model, label the fixed catalogue with an independent
geometric or rollout validator, and rerun all calibration. Episode IDs must denote
independent scene groups, not individual frames from a shared episode. The real
adapter is contract-tested using an actual local Torch checkpoint; no trained
trajectory model or hardware performance has been validated.

## Output coverage and evidence

After freezing **all** deployed choices, run fresh independent output-calibration
episodes. Each target uses one prespecified score per episode:

```text
R_reference = required_clearance - minimum clearance of final candidate references
R_executed  = required_clearance - minimum clearance of actual executed segments
```

The reference minimum is across physical steps, after choosing any fallback and
before release filtering; rejected final candidates remain included. A step with
no reference makes that episode's reference score `+inf`. Executed scores include
all applied controls, disturbances and abstention-stop segments. Scores are fitted
separately for every policy and target using the upper finite-sample quantile;
ties and infinite failures are retained. The report gives test `R<=cutoff`
frequencies over independent episodes, with `vacuous=true` for infinite cutoffs.
This calibration never changes repair, steering, geometric margins or release.

The usual finite-sample marginal score statement assumes exchangeable episodes
and a frozen policy. It does **not** imply conditional safety after release,
closed-loop invariance, or a joint detector safe FPR bounded by `alpha_safe`.
Even a finite cutoff is a score bound, not a safety claim; a positive cutoff can
permit insufficient clearance. Empirical coverage on a small test set can differ
from its nominal probability.

Four paired modes are reported: nominal (diagnostic detection only), safe-only
abstention, joint abstention, and joint detection plus repair. All modes retain
geometric/MPC/deadline release gates and the same fallback protocol. Detection
rates use only each episode's first independently labelled query; correlated
replans are reported as step counts, not independent conformal observations.
Planned collisions include newly sampled proposals and fallback candidates even
when rejected. Executed collisions use actual physical segments. Tracking RMSE
uses Euclidean one-step position error wherever a final candidate exists,
including rejected candidates; this denominator differs across policies.

Atomic artifacts are:

| File | Contents |
| --- | --- |
| `frozen_policy.json` | Schema, source hashes, model/config identity, scene seeds and inputs, independent labels/splits, complete probes and reference signatures, detector scores/radii, schedules, dynamics/costs/constraints and policy hash |
| `output_calibration.json` | Fresh episode results and separate reference/executed fitted score cutoffs bound to each policy hash |
| `evaluation.json` | Paired test seeds, executed positions, controls, per-step flags/failures/fallbacks, proposal/final/executed clearances, steering and measured latency |
| `report.json` | Final complete status, artifact checksums, runtime versions, detection and control metrics, held-out marginal score coverage and limitations |

Nonfinite scores/radii use JSON strings `"+inf"`/`"-inf"`; no nonstandard JSON NaN
is emitted. The report is published last; a directory without a complete report
is not a successful run. [example_run](example_run/) contains observed artifacts.
Source hashes and runtime versions identify the tested implementation; seeds
alone do not promise identical floating-point results or wall-clock latency.
