# Conf-Gen

Collect CLIP inner embeddings for prompt-image data and COCO captions. Prompt
and image harmfulness scores are **validation labels** for inspecting or
evaluating representations; they are not predictor inputs.

## Layout

- `backend/storage/`: SQLite schemas, dataset seeding, and local model copies.
- `backend/embeddings/`: CLIP extraction and fairness/COCO embedding jobs.
- `backend/validation/`: ShieldGemma scoring and validation-label import.
- `backend/gpu/`: Colab transport, workers, and guarded result merging.
- `frontend/app.py`: NiceGUI runner, with `app.py` as the short root launcher.
- `data/fairness/`, `data/coco/`, `data/validation/`, `data/models/`, and
  `data/research/`: dataset inputs, results, models, and research artifacts.
- `research/`: exploratory Python code and reports; its artifacts live under
  `data/research/`.
- `scripts/`: short WSL and PowerShell wrappers.

Default paths are defined in `backend/paths.py`. Every data command also accepts
path overrides. Existing harmfulness columns remain readable, including the
legacy `inner_embedding_harmfulness` field, which no pipeline fills.

## Local commands

Run from the repository root with the project environment:

```powershell
.\.venv\Scripts\python.exe app.py --database data\fairness\fairness.sqlite3
.\.venv\Scripts\python.exe -m backend.storage.fairness
.\.venv\Scripts\python.exe -m backend.embeddings.fairness --inner-embeddings --device cpu
.\.venv\Scripts\python.exe -m backend.validation.fill --prompt-harmfulness --image-harmfulness
```

Validation fills only missing (`NULL`) prompt or image scores and preserves
existing values, including `0.0`. The UI overlays these scores on embedding
plots. Local inference accepts `--device`; the Colab path accepts `--gpu`.

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

The prompt-image Colab runner is `bash -l ./scripts/colab.sh --gpu T4` in
WSL, or `.\scripts\colab.ps1 --gpu T4` from PowerShell. It writes a separate
output database by default. Validation labels can be merged into the fairness
database with `bash -l ./scripts/validate.sh --image-harmfulness --gpu` in WSL
or `.\scripts\validate.ps1 --image-harmfulness --gpu` from PowerShell.
ShieldGemma downloads require Hugging Face authorization via `--hf-token`,
`HF_TOKEN`, or cached login. The Colab session is released after the job.

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
