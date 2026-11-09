# Five-category repair: twenty unsafe test prompts on an optimized GPU

[Repair workflow](../../../docs/repair/README.md) explains the steering methods,
conformal constraints, paired generation, and toxicity-score comparison.

The active configuration uses only the **unsafe members** of the same twenty
selected test pairs: four distinct scenarios per category, seed 20261102.
Safe test variants are excluded from both field extraction and image generation.
Safe training references and safe-validation normalization remain required.
The admission mode is safe attraction + unsafe avoidance. All five repair
methods R1--R5 and baseline R0 remain enabled, with 32 Euler steps and noise
seed 1729. The design has **120 comparison records** and 100 baseline/repair
image comparisons. Source prompts and detector thresholds are unchanged.

This revision writes to `run_unsafe_gpu_native/`. The stopped forty-prompt run and its
CPU migration records remain under `run/`; their manifests are preserved.

The approved compute policy requests a **high-RAM A100**, falling back to a
high-RAM T4 if A100 allocation fails. Hardware and host RAM are checked before
staging. The session is `five-category-unsafe-repair-gpu-20261102`.

The runtime reuses the recovered bank, cached prompt embeddings, and nineteen
verified fields (including one native compatibility-baseline field), plus corrected fields
from the stopped GPU attempt. A CUDA-native negative condition is required; its
velocity matches the original detector bit for bit. Negative velocity caches
include the condition digest. CPU FP16 negative embeddings cannot be substituted.
The previous `run_unsafe_gpu/` attempt is preserved for diagnosis. It loads
only the pinned FP16 transformer and decoder and leaves them on the GPU.
Probe batches 16/32/64 are timed on the allocated card; the fastest candidate
that passes native numerical compatibility is selected. Null velocities are
computed once per bank/batch. Fresh query/anchor probes use batched GPU
inference. Reference fields are cached once in GPU memory where sufficient
VRAM remains, otherwise high host RAM; temporal geometry uses FP64 CUDA
reductions with bounded batches. Distance verification preserves native FP32
weighting and FP64 arithmetic, at the original rtol=0.001/atol=0.001 limits.

During image integration the original, approved, and null conditions are
evaluated at the same current latent in one packed forward pass. A live
native-equivalence check gates this optimization; failure uses separate native
condition calls. The already measured original velocity is reused for the
perturbation diagnostic. No frozen-bank velocity is substituted along the ODE.
R1 still receives fresh direct probe verification. R2--R5 retain the existing
affine verification policy and conservative reference pruning.

Completed field checkpoints are compressed asynchronously and collected into
local WSL storage in parallel with GPU work. Its location is recorded in
`run_unsafe_gpu_native/gpu/reference_cache.json`. Each completed image case is collected
with receipt and artifact checksums. The monitor terminates Colab only after
final image scoring, all required records, and collection are verified.

Useful review artifacts include `run_unsafe_gpu_native/progress.json`,
`gpu_batch_benchmark.json`, `gpu_sampling_compatibility.json`,
`gpu_execution.json`, `gpu_telemetry.jsonl`, and `reconstruction.json`.
The local collector runs under the user service `thesis-five-category-repair-monitor`
with automatic restart on failure. A hidden WSL client keeps the monitor available
until the service finishes. Live progress polling is independent of checkpoint
downloads; `progress.json` includes collected and pending export counts. Service
receipts are under `gpu/monitor.service.json` and `gpu/monitor.launch.json`.

Per-method schedule-search seconds are retained in prompt receipts, separately
from geometry and shared query costs. The benchmark selects the fastest tested
compatible batch; it does not certify a global runtime optimum.

```powershell
wsl -d Ubuntu -- /home/julien/.cache/thesis-inference-venv/bin/python -m backend.experiments.five_category_nudity_repair_20pairs_20261102.pipeline --stage prepare
wsl -d Ubuntu -- /home/julien/.cache/thesis-inference-venv/bin/python -m backend.experiments.five_category_nudity_repair_20pairs_20261102.pipeline --stage launch
```

The earlier measured T4 cadence was about 10.7 seconds per 64-probe field.
Planning ranges before the new benchmark are 2--4 hours on A100 and 5--8 hours
on T4, including repairs and evaluation. Actual batch measurements should
replace these provisional ranges once available.

## Previous matched-pair run (stopped)

Authorized on 2 November 2026. Uses the held-out test partitions of
`five_category_nudity_detection_20261101`, with four matched safe/unsafe pairs
per category. Selection first samples four distinct test scenarios in each
category, then one variant per scenario, using seed 20261102. Neither detector
scores nor image outcomes filter the selection. All forty prompts are retained,
including already accepted controls and unresolved repairs.

The requested admission mode is **safe attraction + unsafe avoidance only**.
R0 is the baseline; R1--R5 are constant interpolation, a fixed temporal window,
a peak-deviation window, a contiguous-deviation window, and an optimized smooth
schedule. The existing repair engine searches all preset variants and generates
the minimum-intervention candidate from each method. There are 240 comparison
records (40 prompts x 6 methods), with shared noise seed 1729 and 32 Euler steps.
Repairs condition on eligible safe training references; the held-out paired
safe prompt is an evaluation control, not a training anchor.

The original detection runtime has expired. The worker therefore rebuilds its
probe bank and the 1,400 training reference fields from the frozen model,
training prompts, seeds, and settings. It also rebuilds the selected forty
query fields and first 64 safe-validation fields. Native/resident bitwise checks
remain enabled. Before image generation it checks every reconstructed query
distance to every safe and unsafe reference against the saved detection
distances, at rtol=0.001 and atol=0.001. A mismatch stops the experiment.
Reconstruction evidence is in `run/reconstruction.json`; bank identities are
in `run/rebuilt_bank.json`. Serialized bank bytes may differ after regeneration;
the original bank is not claimed to have been recovered.

The original safe and unsafe thresholds remain fixed. Window normalization
uses only safe validation prompts. Actual model velocities at the current
repaired latent are evaluated during image integration. Image outcomes do not
select schedules. ShieldGemma 2, pinned to the existing scoring checkpoint,
scores sexual, dangerous, and violence policies after generation. These are
model-rated image-policy scores; semantic preservation and quality remain
unmeasured.

`run/experiment.json` freezes selection, settings, source hashes, and code
hashes. `run/source/` is an isolated copy of the compact detector artifacts.
Parent experiments are read only. `run/selection.json` lists all selected
prompts and pair IDs. `run/validation.json` records bundle verification.

```powershell
wsl -d Ubuntu -- /home/julien/.cache/thesis-inference-venv/bin/python -m backend.experiments.five_category_nudity_repair_20pairs_20261102.pipeline --stage prepare
wsl -d Ubuntu -- /home/julien/.cache/thesis-inference-venv/bin/python -m backend.experiments.five_category_nudity_repair_20pairs_20261102.pipeline --stage launch
```

The high-RAM T4 session is `five-category-repair-20pairs-t4-20261102`.
`run/progress.json` updates every 30 seconds. Each completed prompt's receipt,
images, and output latents are exported atomically and collected with checksum
verification. Reference fields remain remote. After final reports and all 240
comparison records are collected and verified, the monitor terminates the T4
and records shutdown in `run/gpu/shutdown.json`. Failed runtimes are reported
explicitly; already collected images remain local.

Final galleries: `run/pilot/paired_images.html` for field repair and
`run/image_scoring/shieldgemma2/pilot/report.html` for independently scored
image comparisons. Reports keep field admission and image safety separate.

On 2 November, execution was moved to the local CPU at the user's request.
The Colab worker was stopped before recovery. Its 18 completed field
checkpoints, 4,512 prompt-embedding files, original probe banks, compatibility
evidence, and frozen null velocities were recovered with archive and per-file
checksum verification. The verified native compatibility baseline can supply
one further missing training-reference field without inference. The original
experiment manifest, pair selection, thresholds, methods, mode, and seeds are
preserved. Migration and Colab shutdown receipts are under `run/cpu_migration/`.

The CPU entry point is:

```powershell
wsl -d Ubuntu -- /home/julien/.cache/thesis-inference-venv/bin/python -u -m backend.runtime.repair_cpu --output backend/experiments/five_category_nudity_repair_20pairs_20261102/run --threads 4
```

It loads only the pinned FP16 transformer and decoder, using the recovered
prompt and native null-condition embeddings instead of loading text encoders.
At frozen probe states only, it reuses the saved native null velocities and
evaluates the positive condition once per probe. Native velocities of both
conditions are evaluated at the actual evolving latent during image generation.
CPU kernels and single-probe batching do not promise GPU bitwise equality;
the existing detector-distance reconstruction tolerances remain enforced.
`run/cpu_execution.json` freezes the CPU implementation and execution settings.
Large distance products use FP64 coordinate tiles over disk-backed FP32 vectors,
avoiding a full reference cache in RAM. Existing prompt receipts and field
checkpoints are verified and skipped on resume. CPU image scoring uses the
same pinned ShieldGemma checkpoint in BF16, with excess weights offloaded to
local disk, and writes to `run/image_scoring/shieldgemma2_cpu/pilot/`.
CPU execution and errors are reported in the existing `run/progress.json`.
Computational caches and disk-backed matrices use local WSL storage for faster
file access; their absolute locations are recorded in
`run/cpu_migration/recovery.json` and `run/cpu_execution.json`. Reports, generated
images, receipts, logs, and live progress remain in this workspace. This uses
the computer's CPU; no Colab runtime or local GPU is required.
