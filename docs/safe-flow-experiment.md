# Fixed-probe flow-matching experiment

This implements Section 2 of `Conformal_protected_flow_matching_models (4).pdf`.
The pretrained generator is Stable Diffusion 3 Medium, whose transformer output
is used as a flow-matching direction after changing from the checkpoint's
descending noise level `sigma` to the paper's increasing time `t = 1 - sigma`.
In Diffusers v0.32, the scheduler update is `x_next = x + (sigma_next - sigma)
* model_output`, so the paper-oriented velocity is `V = -model_output`.
The model is frozen; the training equations in Section 1 are background, not
a training procedure performed by this repository.

## Protocol

1. Build a safe DiffusionDB prompt corpus and disjoint safe fit, safe
   calibration, safe test, and risky test splits. Risky rows are evaluation only.
2. Select the safe fit set using `reference_limit` (0 means all). Draw `M`
   independent prompt and Gaussian seed pairs with replacement from that set.
   Run guided Euler trajectories from `t=0` to `t=1`. The integration grid
   includes all `K` midpoint times `t_k=(k-1/2)/K` exactly. The same `M`
   trajectories are observed at every midpoint. Freeze the states, times,
   sampled prompt IDs, seeds, model revision, and integration settings.
3. For every prompt, evaluate the checkpoint's native null and positive
   conditions at the same frozen `(x,t)` probes. Store the signature
   `Phi_Q(p) = w * (V_p - V_null)`. The native null embedding comes from
   `StableDiffusion3Pipeline.encode_prompt` with CFG and `negative_prompt=None`.
   No query trajectory or image is generated.
4. Compute pairwise distance as the square root of the mean squared Euclidean
   difference across all `K*M` probes. Score a query by its nearest safe fit
   signature. Use only held-out safe calibration scores for the split-conformal
   order-statistic threshold. The formal `+infinity` cutoff is represented by
   JSON `null` when the requested quantile rank exceeds the sample count.
   Flag a query only when its score is strictly greater than the cutoff.
5. Measure safe-test false-alarm rate and risky-test flag rate, plus prompt-label
   AUROC and AUPRC. A flag means unusual behavior relative to safe fit prompts;
   it does not establish harmfulness or image-level safety.

The finite-sample false-alarm claim requires exchangeability of calibration
prompts and future safe prompts, conditional on the frozen fit set and probe
bank. The implementation cannot establish this data assumption by itself.

## Default pilot and requirements

The default config is `configs/safe_flow.toml`: SD3 Medium at 512 pixels,
FP16, CFG 7.5, 8 trajectories, 8 midpoint times, and a 32-step base Euler
grid. It caps each scoring split at 100 prompts and sets `alpha=0.05`.
These are pilot resource choices, not values required by the PDF. The gated
checkpoint requires accepting its Hugging Face conditions and authenticating.
Its weights and a CUDA GPU are required for the actual experiment.

Install `requirements-flow.txt` in a Python 3.12 environment with a matching
CUDA PyTorch wheel. The config enables model CPU offload to reduce GPU memory
pressure; the workload is still substantial, and the full GPU run has not
been verified in this workspace. A new output directory is required after
changing the fit set, model, guidance, or probe settings. The default output
is `data/flow_experiment`.

## Commands

```bash
python -m scripts.build_prompt_dataset
python -m backend.flow.workflow
```

The first command is only needed when the prompt corpus does not exist. To
build it and run everything in one command, use `--build-dataset`. Staged
commands are available for resuming:

```bash
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
```

The Colab wrappers `scripts/flow.sh` and `scripts/flow.ps1` support the same
stages (`probes`, `reference`, `calibration`, `safe-test`, `unsafe-test`,
`report`). They package source and state, run one stage on a CUDA session,
and verify each returned artifact. The default GPU is T4; L4, A100, H100,
and G4 are also accepted. The model is gated, so pass `--forward-hf-token`
after authenticating in WSL if using Colab. This explicitly forwards a
temporary token to the runtime; it is not included in returned artifacts.
See [gpu-session.md](gpu-session.md) for session management.

CPU checks:

```bash
python -m unittest tests.test_flow_model tests.test_flow_experiment tests.test_flow_workflow tests.gpu.test_safe_flow_colab tests.gpu.test_session_job
```
