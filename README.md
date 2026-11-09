# Five-category nudity detection and repair

This repository contains only the five-category nudity experiments: prompt-label
detection and the paired field-repair study built from its frozen outputs.

## Experiments

- [Detection](backend/experiments/five_category_nudity_detection_20261101/README.md)
- [Repair](backend/experiments/five_category_nudity_repair_20pairs_20261102/README.md)
- [Consolidated results](docs/five-category-nudity/README.md)

## Run

```powershell
uv sync --locked
uv run python -m backend.experiments.five_category_nudity_detection_20261101.pipeline --help
uv run python -m backend.experiments.five_category_nudity_repair_20pairs_20261102.pipeline --help
```

The detection input is `data/controls/five_category_nudity_prompt_pairs_1000.csv`.

Repository conventions are in [AGENTS.md](AGENTS.md). See the
[stage and artifact contracts](docs/architecture.md) and [test commands](tests/README.md).

Reusable code is grouped under `backend/src/{common,data,inference,fields,detection,repair}`.
Detection uses `prepare.py`, `extract.py`, `measure.py` and `score.py`; each stage
selects its supported artifact format or execution strategy through `mode`.
Repair uses `prepare.py`, `execute.py`, `report.py` and `score.py`.
`run_repair_preparation(config, output, mode="paired")` also accepts `mode="controls"` for
repair input formats. Experiment policies and entry points live in
`backend/experiments/`; CPU/CUDA and Colab integration live in `backend/runtime/`.

```powershell
uv run python -m backend.testing
uv run ruff check backend/src backend/runtime backend/experiments tests docs/five-category-nudity/figures/plot_steering_schedules.py
```
