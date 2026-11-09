# Five-category nudity prompt-label detection

Authorized on 1 November 2026. Runs on
`data/controls/five_category_nudity_prompt_pairs_1000.csv`, using its own
output directory and high-RAM Colab T4 session,
`five-category-nudity-detection-t4-20261101`.

The input has 1,000 pairs, with 200 pairs in each of five categories:
intimate bedroom scene, figure study nudity, fine art nude photography,
nude body positivity, and nude self portrait. Each category has 140 training,
20 validation, and 40 test pairs. `control_prompt` supplies the safe label;
`target_prompt` supplies the unsafe label. Existing split assignments are kept.

Settings match the reference: pinned SD3 Medium, FP16, 512 x 512, CFG 7.5,
seed 1729, 64 safe-training-conditioned trajectory probes, eight paired source
probes, checkpointed resident extraction, FP64 distances, five scorers, two
decision rules, and RMS and maximum-deviation aggregation (20 comparisons).
Probe banks, thresholds, and adaptive bandwidths are fitted afresh. Native and
optimized bitwise compatibility gates remain enabled. Results measure prompt
labels, not verified generated-image safety.

`source.csv` preserves the input bytes. `validation.json` records the dataset
audit, frozen split checks, and scientific and implementation comparisons.
`run/experiment.json` freezes settings and input, split, and implementation
hashes. RMS results are under `run/`; maximum-deviation results are under
`run/max_deviation/`. Raw fields and embeddings stay on the GPU host.

Run from the repository root in the existing WSL environment:

```powershell
wsl -d Ubuntu -- /home/julien/.cache/thesis-inference-venv/bin/python -m backend.experiments.five_category_nudity_detection_20261101.pipeline --stage prepare
wsl -d Ubuntu -- /home/julien/.cache/thesis-inference-venv/bin/python -m backend.experiments.five_category_nudity_detection_20261101.pipeline --stage package
wsl -d Ubuntu -- /home/julien/.cache/thesis-inference-venv/bin/python -m backend.experiments.five_category_nudity_detection_20261101.pipeline --stage colab-launch
```

Monitor logs and launch receipts live under `run/gpu/`. A lost runtime is
reported as `error/session_lost`; temporary connection failures are retried.
