# SD3 landscape detector comparison

The new upstream safe/unsafe-set experiment at commit `5f88ae6` is integrated
separately. See [the Section 12 analysis](../docs/jp-two-set-comparison.md) and
run `python -m scripts.compare_two_sets` to compare static dual-set detection
and frozen-probe field steering with the completed SD3 1,000-prompt baselines.
Pinned reference code/results are in `upstream/section12/`; the existing
landscape experiment below retains its original meaning.

This comparison freezes the original nearest-anchor baseline under the `current`
method key. The main `backend.flow` workflow has since switched to uniform
mean anchor weighting. Reproducing this completed comparison still uses the
historical nearest-anchor rule; its existing results retain their original meaning.

This is a detection-only adaptation of JP's E3 landscape experiment from
`main` at commit `5c09e35c1e694656c245bb5ba1604421457a2b07`. It compares
the original **static source-slice detector (`PB|static`, uniform anchors)**
with this branch's **current fixed-trajectory-probe, nearest-safe detector**.
The prompt corpus is copied byte-for-byte. `upstream/` contains reference
copies of the original prompt generator, field implementation, and region
code, with source paths and checksums in `upstream/provenance.json`.
Run the new module below; the upstream scripts are reference material.

## What the comparison measures

| Choice | JP static source slice | Current method |
|---|---|---|
| Generator | Current SD3 Medium adapter | Same adapter and resolved checkpoint |
| Probe states | M shared Gaussian noises at t=0 | K midpoint states along each of M safe-anchor trajectories |
| Default probes | 8 | 8 trajectories × 8 times = 64 |
| Pairwise distance | RMS over probes of Euclidean field differences | Same RMS metric, over trajectory probes |
| Score | Mean distance to **all** 30 anchors | Minimum distance to **one** of the same 30 anchors |
| Threshold | Own safe calibration quantile | Own safe calibration quantile |

The source states are reconstructed from the exact seeds used to start the
current method's trajectories. Both methods therefore share their initial
noise samples. Only safe anchors supply the trajectory prompts. Probe sampling
otherwise follows the current implementation, including sampling anchor/seed
pairs with replacement, rather than JP's original single noise seed 2026.

Source slicing is indeed a probe bank at t=0, but the original static detector
also differs from the current method in how it combines anchor distances.
This experiment preserves both methods. Any performance difference combines
the effects of probe locations and anchor scoring; it does not isolate time
sampling alone. No adaptive weights or tuning-pool optimization run.

The implementation reuses `backend.flow.model.FlowAdapter`, the existing
trajectory collector, probe-field evaluator, nearest-reference distance, and
conformal cutoff. It stores `w*(V_prompt - V_null)` for both methods. This
preserves JP's source-slice metric: for two prompts evaluated at the same
state and time, the shared null field cancels from their guided velocities.
SD3 uses the current adapter's orientation, t=0 noise to t=1 image; the source
slice evaluates the transformer at noise timestep 1000, not image timestep 0.
Full latent Euclidean norms are used; RMS averages over probes, not latent
coordinates. No embeddings, CLIP features, output-image distances, projection,
repair, or image correction enter either detector.

## One frozen split

`configs/landscape_comparison.toml` selects the original first evaluation
split (`numpy.random.default_rng(1000)`), using the original row ordering:

| Partition | Count | Use |
|---|---:|---|
| Approved winter anchors | 30 | Reference history and safe trajectory prompts |
| Approved winter calibration | 40 | Separate threshold for each method |
| Approved winter test | 50 | Coverage and false-alarm rate |
| Summer | 24 | Exclusion rate |
| Neutral | 24 | Exclusion rate, with no binary season label |
| Disguised summer | 16 | Summer requests with winter keywords appended |
| Paraphrased winter | 16 | Safe unfamiliar wording |
| Novel winter scenes | 12 | Safe unfamiliar scenes |
| Novel summer scenes | 12 | Unsafe unfamiliar scenes |

All 224 selected prompts come from JP's evaluation pool; the 86 tuning rows
are unused. Here "safe" means approved winter scenery, a seasonal prompt
policy. Calibration uses beta/alpha=0.1, matching JP. With 40 calibration
scores, the cutoff is the 37th smallest score (one-based). Exclusion requires
score **strictly greater** than the cutoff. An out-of-range conformal rank
uses an infinite cutoff, serialized as JSON null.

The report includes winter coverage, per-kind exclusion rates, and prompt-label
AUROC/AUPRC. Paraphrased and novel winter prompts count as safe; summer,
disguised summer, and novel summer count as unsafe; neutral requests are
excluded from binary ranking metrics. Exclusion of safe unfamiliar prompts
measures a limitation, rather than a successful detection. These results do
not measure generated-image safety. JP's independent image judges are omitted
along with image generation and correction. One split does not measure
variation across splits, and exchangeability remains a data assumption.

## Run on a CUDA machine

Use the repository's Python 3.12 environment, install `requirements-flow.txt`
with a CUDA-enabled PyTorch build, and authenticate to the gated SD3 checkpoint
after accepting its model access conditions. No Sana or extra CLIP models are
needed. Run commands from the repository root.

The comparison inherits the current model, precision, guidance, image size,
offload, and trajectory settings from `configs/safe_flow.toml`. Its own
alpha and full split sizes replace the current corpus's alpha and pilot caps.
It never reads the existing `data/flow_experiment` bank or features, because
those trajectories were fitted to a different reference set.

Freeze and inspect the split without loading a model or requiring a GPU:

```bash
python -m flow_matching_safe_set_POC_JP.experiment --stage prepare
```

Run both methods and produce the comparison:

```bash
python -m flow_matching_safe_set_POC_JP.experiment
```

Alternatively, run resumable stages:

```bash
python -m flow_matching_safe_set_POC_JP.experiment --stage probes
python -m flow_matching_safe_set_POC_JP.experiment --stage extract --split safe_reference
python -m flow_matching_safe_set_POC_JP.experiment --stage extract --split safe_validation
python -m flow_matching_safe_set_POC_JP.experiment --stage extract --split safe_test
python -m flow_matching_safe_set_POC_JP.experiment --stage extract --split requests
python -m flow_matching_safe_set_POC_JP.experiment --stage score
python -m flow_matching_safe_set_POC_JP.experiment --stage report
```

`extract` and `score` accept `--method source_slice` or `--method current`
for independent resumption. `extract` without `--split` processes all four
partitions. `all` loads the model once for probe construction and extraction.
Reruns validate and skip cached features. CPU-only `score`/`report` need all
requested feature files to exist first. The original safe-flow Colab launcher
does not package this separate experiment; use these commands in a checkout
on your GPU runtime.

To change model/probe settings, copy the comparison config and add `[model]`,
`[experiment]`, or `[probes]` overrides. Use `--config path/to/config.toml`
and a fresh `--output data/landscape_comparison_other`. Paths in configuration
resolve from the repository root. Existing split/configuration/model mismatches
are rejected; partially finished extraction can resume with the same settings.

Outputs default to `data/landscape_comparison/`:

- `experiment.json` and `splits/*.jsonl`: effective configuration and exact IDs.
- `probes/{current,source_slice}.pt`: frozen probe states, seed/prompt provenance,
  actual model revision, native null condition, and guidance settings.
- `features/<method>/<split>/*.pt`: per-prompt signatures bound to a bank hash.
- `scores/<method>/`: safe-only threshold, calibration/test/request CSVs, checksums.
- `comparison.json` and `comparison.csv`: both methods' metrics side by side.

With the default 512-pixel SD3 setup, FP32 feature storage is about 4 GiB for
224 prompts, separate from model weights. Source slicing uses 8 field probe
evaluations per prompt versus 64 for the current method, in addition to the
current method's one-time safe trajectory construction. This is an evaluation
count, not a measured speedup. The completed pilot used Colab A100 inference; cached features support CPU rescoring.

CPU verification, including a toy-model end-to-end run and resume checks:

```bash
python -m unittest tests.test_landscape_comparison tests.test_flow_model tests.test_flow_experiment tests.test_flow_workflow
```

Adaptive cache-only rerun (both source-slice and trajectory distances):

```bash
python -m scripts.rerun_adaptive_weights --landscape --bandwidth-scale 1
```

This writes `scores/{source_slice,current}_adaptive_k1/` and
`comparison_adaptive_k1.json/.csv`, preserving the original comparison and
uniform rerun. Weights are JP's `K:1` Gaussian rule, with bandwidth fitted
independently for each method from the same 30 safe anchors. The scale is fixed
before evaluation; the separate tuning-pool search is not run. Model, frozen
probes, prompt IDs, calibration split, and alpha remain the same.

The same cache runner also accepts `--distance-method euclidean` or
`--distance-method anchor_static`. These apply the selected score to both
source-slice and trajectory signatures, writing separate
`scores/{source_slice,current}_{euclidean,anchor_static}/` directories and
`comparison_{euclidean,anchor_static}.json/.csv`. `euclidean` is the original
nearest-safe-reference score; `anchor_static` averages distances to all anchors.
