# Repository restructuring plan

Status: **applied on 2026-10-05**, after the full COCO embedding run completed.
This document records the target layout, data mapping, and verification.

## Target layout

```text
app.py                         # tiny launcher: python app.py
backend/
  storage/
    fairness.py                 # entries schema, seeding, default paths
    coco.py                     # images/captions schema and annotation upsert
    models.py                   # local Hugging Face model storage
  embeddings/
    clip.py                     # CLIP extraction
    fairness.py                 # prompt-image embedding CLI
    coco.py                     # COCO local/GPU embedding CLI
  validation/
    shieldgemma.py              # prompt/image score model
    fill.py                     # missing-score CLI and database updates
    import_sample.py            # reproducible toxic-source sample import
  gpu/
    client.py                   # Colab session/upload/download/archive
    merge.py                    # guarded prompt-image result merge
    fairness_worker.py          # remote prompt-image worker
    coco_worker.py              # remote COCO worker
frontend/
  app.py                        # NiceGUI entry point: python -m frontend.app
  page.py                       # controls and layout
  plots.py                      # loading, projection, Plotly figures
scripts/
  coco.sh, coco.ps1             # WSL/PowerShell convenience wrappers
  colab.sh, colab.ps1           # generic Colab runner
  validate.sh, validate.ps1     # validation runner
  setup_colab.sh                # Colab CLI setup
research/                      # renamed exploratory Python/Markdown code
  ...
tests/
  storage/, embeddings/, validation/, gpu/, ui/
docs/
  style.md, restructuring-plan.md
pyproject.toml, README.md
```

`frontend/app.py` owns the frontend runner. Root `app.py` forwards to it
so `python app.py` also works; it contains no UI logic. Each backend package has an `__init__.py` and owns one concern.
CLI names remain short and stable: `python -m backend.embeddings.coco`,
`python -m backend.embeddings.fairness`,
`python -m backend.validation.fill`, and `python -m frontend.app`.

## Data layout

```text
data/
  fairness/
    records.json                # formerly hf_test_fairness_real.json
    images/                     # formerly data/sa_*.jpg
    fairness.sqlite3            # formerly data/fairness_data.sqlite3
  coco/
    annotations/                # formerly coco_annotations/annotations
    images/val2017/             # formerly coco_annotations/val2017
    coco.sqlite3                # formerly data/coco.sqlite3
  validation/
    toxic_source/               # formerly data_toxic/ image source files
    runs/                       # sample manifest, stage/result DBs, reports, backup
    examples/                   # formerly examples/ images and score JSON
  runs/
    coco_smoke_10/              # earlier ten-caption source and DB
    coco_check/                 # post-migration live Colab check
    embedding_cli_smoke_10/
    toxicity_cli_smoke_10/
  models/                       # formerly models/
  cache/huggingface/            # formerly root .cache/huggingface, if project-owned
  research/
    cache/                      # cache_*.npz, person_basis.npy
    outputs/                    # research-generated outputs
```

All dataset inputs, derived databases, model weights, cache artifacts,
examples, and research outputs belong under `data/`. The virtual environment,
Git metadata, and tool caches (`.pytest_cache`, `.ruff_cache`) are development
state, not datasets. Keep `data/` ignored as appropriate, while retaining any
intentional small fixtures in `tests/`.

## Migration and verification

1. The closed full COCO database contained 25,014 captions and 25,014 text
   contexts and passed `PRAGMA quick_check` before and after its same-volume
   move. All 5,000 referenced COCO images exist. A Windows read while the
   original job was active had returned `database disk image is malformed`;
   checks on the closed database returned `ok`.
2. Code moved into role-based backend packages. `backend/paths.py` supplies
   local defaults; COCO schema/upsert now lives in `backend/storage/coco.py`.
   Colab archives contain the nested package layout. Both the COCO and
   fairness archives were extracted and imported in isolated directories.
3. Dataset inputs and artifacts moved under `data/`. All 262 stored fairness
   image paths were rewritten transactionally and resolve to existing files.
   The fairness database passes `quick_check`; a fresh seed from its moved
   JSON source creates 252 records. Research output and cache paths now point
   to `data/research/`.
4. The renamed WSL and PowerShell wrappers were exercised. PowerShell
   wrappers select Ubuntu explicitly, with `CONF_GEN_WSL_DISTRO` as an
   override. A live Colab T4 run extracted ten COCO embeddings; its isolated
   database has 25,014 annotation rows, ten embeddings, and passes
   `quick_check`. The Colab session terminated. A second run against the full
   migrated database reported zero pending captions without allocating a GPU.
5. The active backend/frontend/scripts/tests tree passes full Ruff checks;
   research code passes unused-import/variable/name checks and compiles.
   All 48 unit tests pass. The optional research pytest suite was not run
   because pytest is absent from the current environment.

The versioned fairness inputs, earlier research artifacts, and examples stay
visible to Git at their new paths; large COCO inputs, model weights, caches,
and generated runs stay ignored.

The existing harmfulness columns remain readable. Prompt and image scores stay
in `backend.validation` and are used as evaluation/visualization labels, never
as predictor features. The legacy `inner_embedding_harmfulness` column remains
in the schema for compatibility; no pipeline should populate it.

## Rollback boundary

The layout change used same-volume renames rather than deleting datasets. If
rollback is needed, stop all database writers, reverse the moves shown above,
restore the previous import paths from Git, and rewrite `entries.image_path`
transactionally to the restored image locations. Recheck both SQLite files
before resuming any job.
