# Testing

The conformal diffusion–MPC tests use a real CPU sampler and SciPy constrained
solver, exact moving-disc geometry, and real Torch weights-only checkpoint I/O.
They cover quantile/tie boundaries, exact signature mixing, nonmonotone finite
search, context/episode isolation, cold/warm ODE/SDE randomness, current-context
anchor evaluation, independent output targets, solver/deadline abstention and
old-reference revalidation. The E2E test runs all four paired policies and checks
the resulting atomic calibration/evaluation artifacts. See
[the experiment README](../backend/experiments/conformal_mpc/README.md) for commands.

The suite follows the TEST rules in [AGENTS.md](../AGENTS.md). Tests protect
observable behavior through public functions; they do not parse source trees or
enforce internal call order. Existing experiment outputs are not test fixtures.

Run from the repository root with Python 3.12 or later:

```powershell
uv sync --locked
uv run python -m backend.testing
uv run python -m backend.testing --suite unit
uv run python -m backend.testing --suite integration
uv run python -m backend.testing --suite contract
uv run python -m backend.testing --suite e2e
uv run python -m backend.testing --coverage
uv run pytest -n 2
uv run ruff check backend/src backend/runtime backend/experiments tests docs/five-category-nudity/figures/plot_steering_schedules.py
```

The runner retains combined stdout/stderr, timestamps, and the actual pytest
exit code under `tests/logs/`. A failed test or empty suite returns a failing exit
code. Use `--start-directory PATH` and `--pattern 'test_*.py'` to narrow discovery.
Direct pytest invocation also works; `-m unit` and the other markers select a
level. The PR/push workflow runs every level and uploads transcripts even after
failure. Repository branch protection must require the `cpu-tests` job to block
merges; that repository setting is separate from this checkout.

| Level | Behaviors and case design |
| --- | --- |
| Unit | Hand-ranked finite-sample calibration and tied scores; nearest/uniform/adaptive distances; Unicode identity; inclusive safe and exclusive unsafe boundaries; R1–R5 schedules, deviation profiles, intervention budgets and optimization; seeded sampling; category selection; detection metrics |
| Integration | Real ZIP/checksum transfers, all-member validation before writes, frozen conflicts, permitted status refresh, duplicate-safe retry; JSON prompt partitioning with no leakage; three trajectory detection methods in both modes and held-out calibration isolation; scoring failure/retry/cache protection; child process failure and empty-suite propagation |
| Contract | Actual Torch signature serialization and quadrature weights; stale model/bank rejection; JSON schedule replay; policy fields/probabilities; same-prompt/mode/seed pairing and duplicate rejection |
| E2E | One application workflow from saved PNG images and checksum receipts to paired JSON/CSV/HTML reports and verified output provenance |

Boundary matrices exercise both sides of admission thresholds and invalid
inputs. Hypothesis checks normalization idempotence, distance symmetry and
nonnegativity and bounded schedule interpolation.
Examples are deterministic, deadlines disabled, and failure output contains the
minimized counterexample. Preserve useful counterexamples as explicit regression
cases. Numerical comparisons use tolerances appropriate to the input scale and
FP64 Gram cancellation; expected results use analytic examples rather than a
second copy of production algorithms.

Every file test owns its `tmp_path`; there is no shared mutable experiment,
production dataset, credential, network call, or arbitrary sleep. The PNG scorer
double replaces only expensive inference. Filesystem, Torch, image decoding,
pairing, report generation, and checksums remain real. Tests run independently
and with multiple workers. No test claims to establish concurrency guarantees
for production operations merely by running the suite in parallel.

The E2E test covers the CPU application workflow, with inference replaced. It
does not establish deployed GPU/SSH, SD3, or ShieldGemma model compatibility.
Those require a separately provisioned environment, pinned model revisions,
and a nonproduction smoke run before adapter changes are released. This
repository's active application has no database, HTTP API, authentication
service, or browser UI to test; historical archived services are retired.

Coverage is available with branches and missing lines, with no percentage gate.
Review uncovered critical behavior when changing it; these focused examples do
not claim exhaustive repair-method or external-adapter coverage. Add a failing
behavioral regression before fixing a reproducible defect. Keep tests at the
lowest level that detects it and run the applicable integration/contract checks
for boundary changes.

Repair execution tests exercise explicit frozen cohorts, receipt verification,
completion hooks, interrupted exports and retry through the public H1 API.
Artifact tests check atomic JSON/ZIP publication, bounded archive expansion,
symlink rejection and recovery subsets bound to trusted hashes. The obsolete
RPC-server test was retired with that removed server; resident model, cache,
reconstruction and repair regressions remain covered.

Offline expert generation adds real optimizer and worker-process tests:

```powershell
uv run python -m pytest tests/unit/test_expert_planning.py tests/contract/test_expert_artifacts.py tests/e2e/test_expert_dataset.py -q
```

They verify absolute moving-obstacle timing, swept collisions, dynamics/limits,
goal and guidance consistency, bicycle angle outputs, split isolation and
training-only normalization. Real subprocess tests cover hard timeouts, bounded
failed-solver retries, one-owner locking, deterministic worker-count changes,
atomic receipt recovery and rejection of corrupted completed datasets on resume.
Run with stable source files: manifests deliberately reject edits made during
generation or validation. Demo commands and physical-model limitations are in
the [expert generator README](../backend/experiments/expert_trajectories/README.md).
