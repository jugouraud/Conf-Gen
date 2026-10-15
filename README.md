# Conformal-protected flow-matching experiment

This repository compares the CFG prompt effect of a pretrained flow-matching image generator at shared states along safe trajectories. A nearest-safe distance is calibrated on held-out safe prompts, then evaluated on safe and risky prompts. The active workflow uses `backend/`, `scripts/`, `tests/`, `configs/`, `docs/`, and `data/`; earlier workflows are preserved under `archive/`.

See [docs/safe-flow-experiment.md](docs/safe-flow-experiment.md) for the protocol, setup, commands, and limitations.

Each workflow step has its own file: `scripts/build_prompt_dataset.py` builds the
corpus, then `backend/flow/corpus.py`, `model.py`, `probes.py`, `features.py`,
`calibration.py`, `evaluation.py`, and `report.py` run the experiment. Their
top-level functions call focused helpers in the same file. `backend/flow/workflow.py`
imports the steps and runs them in order. `backend/flow_experiment.py` preserves
the existing single-step CLI; the Colab client packages the orchestrator and steps.

Files read from lower-level actions to higher-level orchestration: library imports,
project imports, constants, hierarchy 3 atomic actions, hierarchy 2 subworkflows,
then hierarchy 1 workflow functions. Helpers appear before the functions that call
them, and CLI entry-point guards stay at the bottom.

Quick start after installing `requirements-flow.txt`:

```bash
python -m scripts.build_prompt_dataset
python -m backend.flow_experiment prepare
python -m backend.flow_experiment probes
python -m backend.flow_experiment extract --split safe_reference
python -m backend.flow_experiment extract --split safe_validation
python -m backend.flow_experiment calibrate
```

To run every step on a local GPU from an existing prompt corpus:

```bash
python -m backend.flow.workflow
```

Add `--build-dataset` to create the default corpus first. A full run needs a new
output directory; staged commands remain available for resuming an experiment.

For Colab execution, use `scripts/flow.sh` or `scripts/flow.ps1`. To reuse one GPU session for different jobs, see [docs/gpu-session.md](docs/gpu-session.md). Run the CPU tests with:

```bash
python -m unittest tests.test_flow_model tests.test_flow_experiment tests.test_flow_workflow tests.gpu.test_safe_flow_colab tests.gpu.test_session_job
```

`archive/` is retained for reference. Its data and old Python environment are not part of the active workflow.
