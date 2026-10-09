# Safe flow-field extraction (Sana-0.6B)

This is an isolated experiment; no changes to the current COCO/CLIP workflow.

## Goal

At a fixed bank of safe-only latent/time probes, extract Sana conditional velocity predictions for every query prompt. Approximate squared function-space distance by the uniformly weighted sum of squared velocity differences. Each prompt receives the minimum L2 distance to safe-reference prompts. Calibrate a novelty threshold on safe_validation only. Risky prompts are strictly evaluation-only; novelty is not harmfulness.

The fixed probes are sampled using seeded trajectories of the disjoint safe_probe_source split; subsequent extraction of a prompt is deterministic and does not require its own Gaussian seed or ODE integration.

## Install

Use a dedicated CUDA/Python >=3.12 environment; the main project lockfile is deliberately unchanged.

    python -m venv .venv-flow
    source .venv-flow/bin/activate
    pip install -r requirements-flow.txt

Install the matching CUDA PyTorch wheel if needed. The Gemma text encoder may require Hugging Face access/acceptance. GPU model execution has not been verified here.

## Steps

Build the existing prompt corpus first if missing:

    python -m scripts.build_prompt_dataset

From the repository root:

    python -m backend.flow_experiment prepare
    python -m backend.flow_experiment probes
    python -m backend.flow_experiment extract --split safe_reference
    python -m backend.flow_experiment extract --split safe_validation
    python -m backend.flow_experiment calibrate
    python -m backend.flow_experiment extract --split safe_test
    python -m backend.flow_experiment evaluate --split safe_test
    python -m backend.flow_experiment extract --split unsafe_test
    python -m backend.flow_experiment evaluate --split unsafe_test
    python -m backend.flow_experiment report

All commands accept --config configs/safe_flow.toml. Extraction resumes by checking existing signatures. Data/model weights remain local.

## Default pilot

- 32 disjoint safe probe-source prompts; 2 seeded trajectories each.
- 20 scheduler steps; 8 timepoints with 8 safe trajectory states each (64 fixed probes).
- 100 safe-reference prompts, 100 safe-validation prompts, 100 safe-test prompts, 100 risky-test prompts (caps in config).
- Sana 0.6B at 512px, FP16; guidance scale 1 (conditional model-native velocity).
- Uniform weighted L2 and nearest-safe neighbor; 5% safe-only split conformal quantile.
- Prompt-label AUROC, AUPRC, false positive rate on safe test, and risky flagged rate.

Output directory: data/flow_experiment (ignored by Git). It contains split manifests with dataset hashes, a frozen bank with resolved model SHA/scheduler metadata, per-prompt signatures, a frozen threshold and CSV reports.

Do not alter probe bank, reference, calibration or model hyperparameters after seeing risky-test results. To scale to all safe prompts, set reference_limit, validation_limit, test_limit to 0 and choose a new output directory before rerunning. Separate independent pilot thresholds are not comparable.

## Validation

    python -m unittest tests.test_flow_experiment

CPU tests cover deterministic splits, safe-only leakage checks, duplicate handling and L2 nearest neighbor. GPU inference and end-to-end run remain to be validated on the target CUDA machine.

## Mathematical conventions

Sana uses descending scheduler time (noise-to-image); time orientation and scheduler state are recorded. Probe formation integrates only safe trajectories; field extraction calls the frozen transformer directly without integrating. Optional empty-reference subtraction has no effect on pairwise L2. If guidance is changed, a new probe bank must be built to avoid mixing vector fields.

Future work (not implemented in this initial setup): independent probe-bank stability replications, matched safe/unsafe prompt experiments, alternate metrics, activation-layer hooks, image-level safety evaluation.
