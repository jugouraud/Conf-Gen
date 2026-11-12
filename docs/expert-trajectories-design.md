# Offline expert trajectory generation

Decision record, 2026-11-09. Owner: repository maintainers. Status: implemented.
Revisit model assumptions before using targets on a physical vehicle.

Success: a configured CPU command generates scene/goal-conditioned optimized
targets without calling diffusion; only independently validated references enter
dataset exports. Every scene and attempt, including rejection, timeout and solver
failure, has a durable record. Generation resumes without repeating committed
attempts, preserves group splits, fits normalization on training only, and ships
with a verified small dataset, CLI validation/summary, and passing relevant tests.

Inspection found reusable LinearDynamics, circular-footprint swept geometry,
SciPy SLSQP and atomic JSON publication. There is no existing track-map library or
steering vehicle model. Default to the same holonomic velocity-controlled disc;
add a separately configured kinematic bicycle for steering/acceleration constraints.
Maps are configured rectangles, not an implied road-network/map-service integration.
No new package or remote service is required. Existing image and online control
pipelines and their frozen evidence remain unchanged.

Priorities: validated feasible targets; correct physical timing and footprint
geometry; group-level data integrity; bounded/restartable execution; reproducible
scientific values. General stages and contracts belong in backend/src/expert;
procedural map distributions, example configurations and entry points belong in
backend/experiments/expert_trajectories. Each core step owns one public run_<step>
entry point. Orchestrators call stage H1s; no private cross-step calls.

Plan states/controls by direct shooting and a goal/progress, effort and smoothness
objective. Initial guesses are numerical seeds, not targets to track. Search a
finite configured collection with bounded retry seeds. Independently check every
solver result and exported y_ref_star=h(states), even if the solver reports
success. Success is local convergence plus validated constraints, never global
optimality; failure is not proof of scene infeasibility.

Use absolute seconds, metres and radians, FP64 CPU. Targets use exactly N+1
samples at start_time+i*dt. Never silently retime user input or exported targets.
Obstacle samples use piecewise-linear interpolation on explicit timestamps with
no extrapolation; clearance checking includes every obstacle time breakpoint.
Bicycle heading is unwrapped in states; optional sin/cos output avoids wrapped
angle interpolation. Continuous bicycle checking uses substeps with a documented
curvature/acceleration chord bound and a configured numerical error margin; this
is not a physical integration-error certificate.

Assign independent group IDs to train/reference/future_calibration/test before
creating maneuver jobs. Related alternatives inherit one group and unchanged
scene. Reserved expert labels are not conformal calibration observations; future
calibration must sample the frozen deployed generator/repair/controller policy.
Raw observations and states remain available; normalized views are optional and
use training-only statistics. No external transmission or telemetry is added.

One coordinator owns each output under an OS file lock. Worker concurrency is
bounded; each attempt runs in a subprocess with limited native threads and a
hard parent timeout in addition to cooperative solver checks. Receipts publish
atomically and bind source/config/scene/guidance/seed identity. Resume refuses
changed inputs or committed bytes; interrupted, uncommitted attempts may rerun.
Wall-clock diagnostics are not expected to be byte-reproducible across runs.

Smallest convincing checks: analytic straight-line optimization, actual moving
obstacle timestamps and between-knot crossings, dynamics/actuator/footprint/goal
rejections, exported shape/angle semantics, same-scene distinct guidance, split
and normalization isolation, repeatability across worker counts, real subprocess
timeout and checkpoint/resume/tamper rejection, plus a complete demo and existing
repository checks. Review the final diff and uncovered critical branches.

Delivery: [commands and assumptions](../backend/experiments/expert_trajectories/README.md),
[schema and stage contracts](expert-dataset-schema.md), and
[observed results](../backend/experiments/expert_trajectories/RESULTS.md).
The default and bicycle examples ship complete receipt-backed datasets. The
bicycle example validates straight speed alternatives only; physical Ackermann
deployment and a matching nonlinear online tracker require separate work.
