# Conf-Gen

Collect CLIP embeddings for COCO captions and evaluate prompt coverage against
the saved COCO region using the I2P benchmark prompts.

## Layout

- `backend/storage/`: SQLite schemas, dataset seeding, and local model copies.
- `backend/embeddings/`: CLIP extraction and COCO region jobs.
- `backend/validation/`: CSV prompt region analysis.
- `backend/gpu/`: Colab transport, workers, and guarded result merging.
- `frontend/coco_region_app.py`: validation analysis dashboard, with
  `coco_region_app.py` as its short root launcher.
- `data/coco/`, `data/validation/`, `data/models/`, and
  `data/research/`: dataset inputs, results, models, and research artifacts.
- `research/`: exploratory Python code and reports; its artifacts live under
  `data/research/`.
- `scripts/`: short WSL and PowerShell wrappers.

Default paths are defined in `backend/paths.py`. Data commands also accept
path overrides.

## COCO embeddings on Colab

The Colab CLI runs in Linux or WSL. Set it up and authenticate in the same
shell that launches the job. From the repository root in WSL Ubuntu:

```bash
bash scripts/setup_colab.sh
colab --auth oauth2 sessions
bash -l ./scripts/coco.sh --gpu
```

The PowerShell wrapper is `.\scripts\coco.ps1 --gpu`. It targets Ubuntu;
set `CONF_GEN_WSL_DISTRO` if your WSL distribution has another name. The job reads
`data/coco/annotations/captions_val2017.json` and
`data/coco/images/val2017/`, then fills `data/coco/coco.sqlite3` in batches.
Completed batches are committed; rerunning finds only captions without
embeddings. For a separate ten-caption source, use
`--annotations data/runs/coco_smoke_10/annotations.json --database data/runs/coco_new_check/coco.sqlite3`.
`--limit 10` processes ten missing captions from the full annotations but still
stores metadata for every caption. `--batch-size 100` reduces upload size when needed.

## Checks

```powershell
ruff check backend frontend scripts tests --select F401,F841,F821
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The full COCO database and model weights are local artifacts under `data/`.
See [the migration record](docs/restructuring-plan.md) for the path changes
made after the 25,014-caption run completed.

## COCO conformal region

After all COCO text contexts are stored, run:

```powershell
.\.venv\Scripts\python.exe -m backend.embeddings.define_region
```

This selects each caption's CLIP `[EOS]` vector, splits COCO by image, fits a
probabilistic-PCA Mahalanobis metric on the reference images' captions, and
uses the other 20% of images to calibrate the 95% nearest-anchor boundary.
Each calibration image contributes its highest caption score. The accepted
region is a union of equal-radius
balls in whitened embedding space. Its anchors, whitening parameters, radius,
split metadata, and source database signature are saved in
`data/coco/coco_region.npz`. Re-running the command loads that file when the
database and settings match; `--force` recomputes it. Other code can call
`define_coco_region()` and then `region.score(vectors)` or
`region.contains(vectors)` with 768-dimensional `[EOS]` vectors. The boundary
models the COCO caption distribution, not a harmfulness label. The score is
nearest-anchor distance directly; the research's MST score gives the same
union-of-balls boundary only when its anchor graph is connected at the radius.

Open the validation analysis dashboard:

```powershell
.\.venv\Scripts\python.exe coco_region_app.py --port 8081
```

The root page compares CSV `prompt_toxicity` scores for blocked and allowed
prompts, then shows toxicity density curves, category breakdowns, and a
paginated prompt table. It reads all 4,703 rows from
`data/validation/i2p_benchmark.csv` (or a CSV passed with `--prompts`). Saved
metrics load from `data/validation/runs/coco_region_blocking_summary.json` and
row decisions from `data/validation/runs/coco_region_blocking.csv`; only the
visible nine prompt rows are sent to the browser. Opening the page does not run
prompt encoding or load the 3D region visualization. If the saved results are
missing or stale, select **Analyze prompts** to calculate them; **Recompute
analysis** refreshes current results.

Recomputation checks each distinct prompt against the full COCO region and
calibrated order-1 W1 and adaptive order-infinity budgets. The transportation
cache can be prepared with
`python -m backend.embeddings.transport_budget`. CLIP EOS vectors and token
clouds are cached in `data/validation/validation.sqlite3`, so repeated runs
reuse stored encodings. Pass `--validation-database` to select another cache.
The projection cache can be prepared with
`python -m backend.embeddings.project_region`; use `--projections` if it has
a nondefault location.

To analyze all prompts in one batch, run
`python -m backend.validation.region_report`. It writes per-prompt decisions to
`data/validation/runs/coco_region_blocking.csv` and aggregate statistics to
`data/validation/runs/coco_region_blocking.json`. Repeated prompts share one
text-space decision, but retain separate CSV rows.
