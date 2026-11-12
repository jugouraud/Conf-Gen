# Detection and repair contracts

The trajectory control extension is documented in
[conformal MPC design](conformal-mpc-design.md) and the
[runnable experiment](../backend/experiments/conformal_mpc/README.md). It reuses
the public conformal quantile, probe RMS and steering schedule primitives below.
`backend/src/control/pipeline.py` composes one H1 per observation, detection,
repair, sampling, validation, tracking, fallback and release step. General
`backend/src/mpc/solver.py` has no diffusion or detection dependency. Physical
simulation and the synthetic model live only in `backend/experiments/conformal_mpc`.

The repository follows [AGENTS.md](../AGENTS.md) as a single-user research CLI
with one modular Python application. Model inference and the authenticated
Colab CLI are external boundaries. There is no database, web API or distributed
job service. Dataset text is input data and cannot authorize code or commands.

## Core stages

Each core step owns its artifacts and exposes one `run_<step>()` H1. CLI
pipelines select stages and supply configuration; inference adapters are created
at the worker composition boundary. Helpers are optional and stay cohesive.

| H1 | Input | Owned output and failure contract |
| --- | --- | --- |
| `run_nudity_detection_preparation` | Config and 1,000 pairs; five categories, 140/20/40 pairs per category across train/validation/test | Frozen split JSONL, manifest and resource plan; rejects changed or incorrectly stratified inputs before freezing |
| `run_detection_packaging` | Prepared detection directory | Checksummed source/input ZIP; rejects absent preparation and preserves the previous ZIP if packaging fails |
| `run_detection_preparation` | Config, output, `mode=paired/reused/source_slice/controls` | Frozen splits, config and checksums in the selected source format |
| `run_detection_extraction` | Prepared directory, `mode=resident/native/staged/cached`, optional adapter and mode options | Probe banks, FP32 fields with hash sidecars; resident mode fuses FP64 distance products while fields are on device |
| `run_detection_measurement` | Frozen fields, `mode=paired/probes/reused/controls`, space and device | FP64 distance archives and receipts; probe mode supports RMS/max reductions |
| `run_detection_scoring` | Verified distances, `mode=resident/paired/probes/source_slice/reused/controls/comparison` | Training-fitted rules, validation-fitted cutoffs and held-out metrics; no test-label fitting |
| `run_repair_preparation` | Config, output, `mode=paired/controls` | A `PreparedRepair` record; paired mode freezes a scenario-selected worker bundle, controls mode binds saved signatures and benign controls into a `RepairExperiment` |
| `run_repair_execution` | `RepairExperiment`, stage and optional explicit cases | Images, latents and checksummed per-prompt receipts; cached cases are verified, export hooks run after receipt commit, and failures publish a failed stage |
| `run_repair_reporting` | Verified repair receipts | Field statistics, compute summaries and galleries |
| `run_repair_scoring` | Saved images and independent scorer | Per-policy scores and paired image comparisons, separately fingerprinted from field measurements |

`backend/src/detection/pipeline.py` composes extraction, measurement and scoring.
The experiment `pipeline.py` files and worker entry points dispatch public stage
APIs. CPU and GPU repair workers supply their explicit cohorts and completion
hooks to the same repair H1 instead of duplicating the search/sample loop.

## Shared boundaries

The source root contains packages by capability:

- `common/`: paths, atomic storage and verified transfer archives.
- `data/`: prompt identity, paired parsing and partition selection.
- `inference/`: flow-model loading/encoding and independent image scorers.
- `fields/`: signatures, checkpoints, native compatibility, distances and constraints.
- `detection/`: four mode-driven stages, calibration, artifact contracts and report rendering.
- `repair/`: preparation, execution, reporting, scoring, schedules and artifact contracts.

`backend/runtime/` owns CPU/CUDA adapters and Colab transport. Dataset-specific
category names, cohort limits, resource plans and entry points belong under
`backend/experiments/<experiment>/`. Shared code derives categories and cohort
sizes from validated inputs; explicit experiment expectations live in config.
Both preparation modes use one repair API and return the same record type.
No compatibility forwarding modules remain for the old stage filenames.

Steps do not import other steps' private functions. Runtime adapters do not
depend on worker modules. Helpers exist where they clarify substantive work;
H1 functions call libraries and helpers directly when another layer adds nothing.

This layout keeps artifact formats and numerical kernels stable while retiring
duplicate format-specific entry points. Resident extraction deliberately retains
its fused distance calculation to avoid repeated device transfers. Modes preserve
existing source formats rather than rewriting archived runs. New code fingerprints
require fresh prepared outputs; archived scientific evidence keeps its original bytes.

Numerical contracts are unchanged: probe order is time-major then trajectory;
fields have shape `(probes, latent_channels, height, width)` and contain CFG-weighted
prompt-minus-null velocity effects in latent units. Checkpoints are FP32;
reference distances and search geometry use FP64 arithmetic. Higher detection
scores indicate the unsafe prompt label. Safe admission is inclusive and unsafe
exclusion is strict. R1--R5 strengths remain bounded by zero and one.

## Integrity and lifecycle

New detection manifests fingerprint the shared and stage implementation, pinned
model/config/input identities, split order, seeds and numerical settings.
Historical result manifests retain their original bytes; code changes require
fresh inference outputs. Reporting and independent image scoring retain their
own provenance. Seed equality alone does not promise identical GPU results.

JSON documents and transfer ZIPs publish through unique temporary files and an
atomic same-directory replacement. Tensor checkpoints and prompt receipts retain
the existing single-worker-per-run ownership. This does not promise transactions
across an entire directory or permit two inference workers to share an output.
Never mark a failed or interrupted run complete merely because some files exist.

Archive extraction checks the manifest and selected hashes before writes,
rejects path escapes and symlinks, and bounds members (50,000), manifest bytes
(16 MiB), individual expansion (8 GiB) and total expansion (128 GiB). Callers
can supply smaller `ArchiveLimits`. Recovered caches are restricted to the
frozen allowlist; source correction copies the stopped run before changing
additional fields. JSON/ZIP atomicity does not establish concurrent-run safety.

Colab operations use fixed argument lists, finite per-command timeouts and named
sessions with an explicit GPU allowlist. Repair staging validates the prepared
package and multipart recovery hashes before uploads. Credentials are injected
separately and do not belong in bundles, manifests or test fixtures. Local output
and code/config paths are controlled by the researcher; this CLI is not an
untrusted multi-user execution service.

## Verification and current limits

Run the commands in [tests/README.md](../tests/README.md). CI covers the active
source, both experiment entry points, all test levels and the retained schematic
report script. CPU tests use isolated generated inputs and replace costly model
inference only. CUDA/native model and live Colab compatibility need a provisioned
smoke run. No network operation or paid compute allocation is part of CPU tests.

CODE6 is applied to new stage/shared APIs and touched contracts. Some existing
numerical helpers still use dictionary records and inferred local types; replacing
them all or introducing a new static checker would be a separate migration.
Scientific manifests continue to reject changed implementations rather than
silently updating recorded results.

## Offline expert data

`backend/src/expert` owns scene contracts, expert direct-shooting optimization,
independent trajectory validation, bounded process execution, atomic receipts,
export and training-only normalization. It reuses `src/mpc` dynamics/geometry and
the SciPy numerical stack. The expert objective receives the scene and goal;
it does not track a generated reference or change conformal detector/repair math.
`backend/experiments/expert_trajectories` owns procedural distributions, supplied
scene examples and configurations. One H1 per core stage composes through the
final orchestrator, with one dataset coordinator holding an OS lease.

See the [design decision](expert-trajectories-design.md),
[artifact and stage contracts](expert-dataset-schema.md), and
[commands/results](../backend/experiments/expert_trajectories/README.md).
Related variants retain their preassigned episode split. Reserved expert labels
must not substitute for outputs of the frozen deployed policy during future
conformal calibration. The default disc matches the online demonstrator; the
optional offline bicycle is not an implemented nonlinear online controller.
