# Expert dataset schema v1

Producer: `backend.src.expert.pipeline.run_expert_pipeline`. Consumers: offline
conditional planner training and research evaluation. These records contain
expert labels, not diffusion outputs or conformal calibration scores. Version
changes that alter shapes, timing, models or field meaning need a new schema;
v1 readers reject unknown input fields and unsupported schema versions.

All files use UTF-8 JSON or JSONL, finite numeric values and no executable
serialization. Duplicate JSON keys, NaN/Infinity and nesting over 32 are rejected.
Input documents/individual records are limited to 16 MiB; combined JSONL exports
to 64 MiB and 10,000 rows; TOML configs to 128 KiB. Use multiple independently
managed datasets for larger workloads. No untrusted pickle or model loading occurs.

## Supplied scene document

The document has exactly `schema="expert-scenes-v1"` and `scenes=[...]`. Each
scene contains the following required fields (including explicit null heading).
See [the supplied example](../backend/experiments/expert_trajectories/user_scenes.json).

| Field | Meaning |
| --- | --- |
| `scene_id` | Unique scene/variant string, 1–200 characters; not a filesystem path |
| `group_id` | True independent episode ID, 1–200 characters; shared by all related variants |
| `map_id` | Map identifier, 1–200 characters |
| `bounds` | `[xmin,xmax,ymin,ymax]` in metres, strictly increasing on each axis |
| `start_time` | Absolute physical seconds, finite |
| `initial_state` | Disc `[x,y]`; bicycle `[x,y,unwrapped_yaw,v,steering]` |
| `initial_velocity` | `[vx,vy]` m/s; must match bicycle heading and speed when applicable |
| `goal` | `[x,y]` in metres |
| `goal_heading` | Finite radians for bicycle terminal heading, otherwise null |
| `obstacles` | Up to 32 timed circular obstacles |

Each obstacle has exactly `obstacle_id`, `times`, `centers`, `radius`. IDs are
nonempty and unique within a scene. `times` is a strictly increasing length-M
array (2≤M≤1024), `centers` is `(M,2)` metres, and radius is finite and nonnegative.
Static obstacles repeat their center at two or more timestamps. Moving obstacle
centers are piecewise linear in **absolute** physical time. Coverage must include
`[start_time,start_time+N*dt]`. Extrapolation is rejected. Knots need not align
with the planner's grid; validation uses the union of obstacle and path knots.

Malformed scene rows are preserved as schema rejections where parsing permits.
An invalid JSON document cannot be safely interpreted and fails before generation.
Geometrically invalid initial scenes remain recorded with their group/split and
reasons. Duplicate scene IDs fail the dataset rather than ambiguously assigning
attempts. At least four structurally valid independent groups must be supplied;
an all-invalid document cannot establish a four-way dataset split.

## Frozen ownership and files

| File | Owner and contents |
| --- | --- |
| `frozen.json` | Preparation: schema `expert-dataset-v1`, SHA-256 `dataset_id`, expanded config, scientific config, raw scenes/status/rejections, immutable group splits, source hashes, runtime and sampling provenance |
| `attempts/<id>.json` | Generation: atomic `{payload_sha256,payload}` receipt for every completed attempt, including failures |
| `targets/<id>.json` | Export: one independently validated selected target, schema `expert-target-v1` |
| `dataset.jsonl` | Export: all targets in deterministic scene/guidance order, for inspection |
| `splits/<split>.jsonl` | Export: identical records restricted to train/reference/future_calibration/test |
| `index.json` | Export: target IDs, every attempt outcome and selection decision, rejected scenes, hashes of receipts/targets/JSONLs |
| `normalization.json` | Normalization: training-only fit or explicit `no_training_targets` |
| `generation_sessions.json` | Generation: workers/threads, running/finished/interrupted sessions, elapsed time and reused/executed counts |
| `summary.json` | Summary: validation receipt, measured aggregate metrics and final artifact hashes |
| `status.json` | Generation/summary: running, complete or no_training_targets, dataset ID and final summary checksum |
| `.writer.lock` | Ephemeral OS lock; not dataset content, not committed |

The science identity excludes only operational worker count; native thread count
is retained. Input scenes, distributions, all source hashes and runtime versions
are included. Each job is keyed by dataset ID, scene index, guidance index,
initial guess index and retry index. A seed derives from this identity plus the
configured root seed. Attempt IDs like `s0000-g0-i0-r0` are generated internally;
user scene/group IDs never form output paths.

Atomic publication owns one file at a time. `status.json` grants completion only
after export, normalization and dataset validation. Failed/interrupted work never
exposes a successful final marker. A process crash before a receipt commit may
repeat that attempt. Committed receipts cannot be overwritten on resume, and
changed source/config/scenes or corrupted completed artifacts are rejected.

## Attempt contract

`payload` contains `input_id` (canonical job hash), `job`, `result`. The job retains
expanded config, unchanged raw observation, guidance, split/group, deterministic
seed and all indices. Result contains:

- `status`, `solver_success`, `feasible`, `eligible`: independent notions of solver
  convergence, physical validation and export eligibility.
- `states`, `controls`, `timestamps`, `y_ref_star`, `cost`, `validation`: final
  numerical candidate and independent checks when available; null otherwise.
- `solver_code`, `iterations`, `runtime_seconds` when the numerical solver ran;
  `worker_seconds` includes subprocess startup and shutdown. Hard-killed workers
  have no invented solver runtime or trajectory.

Statuses include `validated`, `validation_failed`, `solver_not_converged`,
`invalid_scene_or_guidance`, `solver_timeout`, `numerical_or_solver_failure`,
`worker_timeout`, `worker_failure`, `worker_protocol_failure`. A feasible candidate
is eligible only if convergence is required and obtained, or if the explicit
`require_convergence=false` policy permits it. Timeout candidates are never
eligible. All actual retry attempts receive separate receipts. Initial rejection
is a scene outcome and creates no optimizer attempt.

Export independently checks candidates again and selects the minimum computed
expert objective among eligible guesses/retries for each scene-guidance pair.
Attempt ID breaks exact cost ties. Other valid guesses remain in the manifest.
Similar alternative guidance outputs are marked `duplicate_maneuver`; they are
not saved twice as training targets. A failed solver is not a proof that its
scene is infeasible.

## Target contract

Each target records `schema`, `dataset_id`, `target_id`, `scene_id`, `group_id`,
`split`, `observation`, `guidance`, `y_ref_star`, `states`, `controls`, `timestamps`,
`vehicle`, `planner`, `solver`, `validation`, `seed`, `input_id`, `usage`.
`observation` is the complete unchanged raw scene described above, including
map, goal and timed obstacles. Guidance has `name`, `side` (-1,0,+1),
`speed_scale` (0,1] and `obstacle_index` (0–31).

| Numeric array | Shape | Order and units |
| --- | --- | --- |
| `timestamps` | `(N+1,)` | `start_time + i*dt`, seconds, i=0..N |
| `states` | `(N+1,2)` or `(N+1,5)` | Disc xy, or bicycle x/y/yaw/v/steering; m, rad, m/s |
| `controls` | `(N,2)` | Disc vx/vy m/s; bicycle acceleration m/s², steering rate rad/s |
| `y_ref_star` | `(N+1,n_y)` | `h(states[i])`; xy (n_y=2) or x/y/sin(yaw)/cos(yaw) (n_y=4) |

No diffusion time appears in these targets. No retiming, padded endpoints or
inferred timestamps are allowed. Interpolation of positions for clearance is
piecewise linear; bicycle dynamics use the declared RK4 scheme, and angle output
is calculated from unwrapped states at each knot. A downstream resampling step
must independently revalidate and preserve the group assignment.

Validation records `valid`, reason codes, named nonnegative violation magnitudes,
their maximum, goal error/progress, max speed/acceleration/steering/rate,
heading error and achieved lateral passage offsets. `reference_clearance` is
minimum surface distance of the exported polyline at correct obstacle times;
`physical_clearance` uses the model's dense center path. `minimum_clearance` is
the minimum of reference clearance and physical clearance minus interpolation
margin. Obstacle-free clearances are **null**, not Infinity or a fabricated value.
They do not describe a deployed tracking rollout. `maximum_constraint_residual`
mixes constraint units as a diagnostic maximum; inspect named residuals for
physical interpretation. Dynamics residual is in the state component units.

`solver` retains status/code/iterations/cost and measured solver/worker times.
`usage` is `expert_training_target` only for training rows; other splits are
`reserved_expert_label_not_conformal_score`. Normalization stores fitted training
target/group IDs, mean, scale, counts and constant-dimension flags; it never
modifies these raw records. Variable obstacle sequences require a training
encoder that understands their timestamps and masks; this dataset does not
silently flatten or pad them into an unspecified model input.

## Stage API and failure boundaries

| H1 step | Input → output; persistence owner |
| --- | --- |
| `run_scene_sampling` (experiment) | Config or supplied JSON → raw scene rows + sampling provenance |
| `run_expert_preparation` | Config + rows → frozen manifest; rejects conflicting identity or inadequate group count |
| `run_expert_optimization` | One immutable scene/guidance/config/seed → solver result; no filesystem writes |
| `run_expert_validation` | Scene/guidance/config + numeric arrays → independent validation record; no writes |
| `run_expert_generation` | Frozen manifest → bounded worker attempt receipts; coordinator owns commits |
| `run_expert_export` | Frozen manifest + attempts → validated distinct targets and index |
| `run_training_normalization` | Targets + config → training-only moments |
| `run_dataset_validation` | Dataset files → recomputed identity, integrity, physics and split checks |
| `run_dataset_summary` | Verified stage results → aggregate metrics and final status marker |
| `run_expert_pipeline` | Composes stage H1 calls under the dataset lock |

The objective and independent validator share stable dynamics/geometry primitives
but do not share optimizer constraint assembly. Tests use analytic expectations
and adversarial references to detect correlated mistakes. JSON artifact hashes
are integrity/provenance checks, not authentication. Outputs are local research
files; this CLI is not a service for concurrent hostile tenants.
