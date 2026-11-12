# Conformal diffusion and tracking MPC

Implementation decision, 2026-11-09. Owner: repository maintainers. Status:
implemented and CPU-validated. Revisit when a trained trajectory checkpoint or nonlinear vehicle
model is introduced.

The inspected active pipeline supplies Torch/NumPy field distances, finite-sample
quantiles and smoothstep steering schedules. Its checkpoint interface is SD3 image
generation; no motion checkpoint, dynamics or MPC exists. The extension therefore
adds a reusable linear-dynamics tracker and trajectory-model contract, and a
separately labelled synthetic velocity-controlled-disc experiment. Existing image
experiments and their mathematics remain unchanged.

Success means a reproducible CPU command constructs disjoint episode splits,
freezes probes/catalogue/model/policy, calibrates both detector classes, evaluates
four paired policies, separately calibrates reference and executed clearance
scores, and writes replayable artifacts and measured metrics. Tests must establish
quantile/tie inequalities, exact field mixing, context preservation, finite-grid
minimum intervention including disconnected feasible regions, episode isolation,
swept moving-obstacle geometry, and abstention on failures/deadlines.

Priorities are mathematical identity, fail-closed control release, reproducible
episode-level evidence, and model substitution. All calculations use metres,
seconds and FP64 CPU geometry. Physical step `k`, prediction index `i`, and
noise-to-output diffusion time `t` have distinct contracts. Predictions are
piecewise linear over physical time. Disc footprint and a configured uncertainty
margin are explicit approximations, not a general vehicle safety proof.

General stages live under `backend/src/control`, MPC and geometry under
`backend/src/mpc`; experiment generation, catalogue rules and evaluation live
under `backend/experiments/conformal_mpc`. Each workflow step has one public H1;
the online orchestrator composes only those H1 APIs. Existing public quantile,
distance and schedule primitives are reused. No private cross-stage access.

Anchor catalogue entries identify reevaluable guidance, not saved paths. For
repair the nearest compatible safe entry is selected by its freshly evaluated
signature under the query context. Candidate schedules preserve the baseline,
context and noise coefficient, and are ranked only by exact integrated strength
after the unchanged joint detector accepts. There is no monotonic search or
post-calibration tuning. Clearance calibration is observational and never changes
this criterion.

Bounded local SLSQP solves the tracking problem; independent post-solve checks
verify dynamics, bounds and swept clearance. Deadline checks prevent late release
but cannot preempt an arbitrary native model call. Deployment requiring hard
deadlines needs a separately supervised process/controller. Failure returns
explicit abstention, or a freshly validated and re-solved old reference within the
original budget. No fallback is claimed invariant or universally safe.

Calibration artifacts are atomic JSON, with explicit infinity encoding, schema,
source/model/config identities, split membership and seeds. A run is complete
only after its final manifest is published. Tests and a small end-to-end CPU run
are the initial convincing checks; a real-model/hardware claim requires additional
validation on that model and platform.
