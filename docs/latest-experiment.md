# Latest experiment

The active run is `data/flow_t2i_two_mode_1000`, including the held-out
`additional_tests/style_matched_safe_prompts_1000` evaluation. It compares five
field-signature methods in safe-attraction and joint unsafe-avoidance modes.

Read the main and additional-test `comparison.md` files for results and limitations.
All original manifests, probe banks, signatures, splits, calibrated rules,
distance caches, decisions, and provenance receipts remain in place.

## Commands

Install dependencies from `requirements-flow.txt` with Python 3.12 or later.
Run these commands from the repository root:

```powershell
python -m scripts.prepare_t2i_comparison
python -m scripts.extract_t2i_comparison
python -m scripts.measure_t2i_distances
python -m scripts.score_t2i_comparison
```

Extraction and default distance measurement require a suitable GPU and access
to the pinned image model. Completed caches are already present. Preparation
checks frozen inputs and refuses a different manifest in the same output folder.

The additional test uses the supplied intended-benign controls:

```powershell
python -m scripts.prepare_additional_test --input PATH_TO_ORIGINAL_CONTROLS_JSON
python -m scripts.extract_t2i_comparison --output data/flow_t2i_two_mode_1000/additional_tests/style_matched_safe_prompts_1000 --split additional_safe_test
python -m scripts.evaluate_additional_test
```

The original input path is recorded in the additional-test manifest; a copy is
preserved as `input.json`. Use a fresh output folder for different inputs.
The additional test preserves the parent banks, anchors, cutoffs, and unsafe tests.

Colab entry points are `backend.gpu.t2i_comparison_client`,
`backend.gpu.t2i_safe_transfer`, `backend.gpu.t2i_comparison_finish`, and
`backend.gpu.additional_test_client`. They require an authenticated Colab CLI
and an existing GPU session. Use each module's `--help` where supported.

Run the retained CPU checks with:

```powershell
python -m unittest discover -s tests
```

## Retained dependencies and historical files

`data/flow_t2i_two_mode_1000/reused_safe_source` holds the frozen safe source required by
`configs/t2i_two_mode_comparison.toml`: its experiment manifest, safe splits and
features, paired banks, distance caches and receipts, and five baseline reports.
Historical source paths in unchanged configs resolve to this retained cache.
Its obsolete unsafe signatures, image sample, and other old results were removed.
The distance receipts still describe the original run, including its historical
unsafe rows; the current experiment reuses only their safe rows.

`data/cache/T2I-RiskyPrompt.json` remains the unsafe prompt source. All of
`archive/`, README files, configuration files,
dependency manifests, Git history, and the installed runtime/tooling are retained.
Older configurations and historical READMEs may describe removed workflows.
The obsolete snapshot under `data/experiment_archive/` has been cleared.

Shared method definitions and file reuse now live in `backend/flow/comparison.py`.
Extraction checkpoints and transfer bundles live in `backend/flow/extraction_run.py`.
The current workflow no longer imports the obsolete comparison runners.
