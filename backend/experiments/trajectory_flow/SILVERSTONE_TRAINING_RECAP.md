# Silverstone trajectory-flow training recap

Snapshot from local `.cache` artifacts on 2026-11-14. This inventory includes
only checkpoints whose frozen architecture records `track.layout = "silverstone"`.
All reported evaluation values are held-out training-group validation values, not
untouched final-test results and not safety certificates.

## Trajectory datasets

Every route uses the stylized fixed Silverstone circuit, a velocity-disc vehicle,
`H=96`, `dt=0.4 s`, one stationary circular obstacle, and independently validated
procedural candidates. `Accepted` means validated expert targets, not complete
matched triplets.

| Dataset directory (ID prefix) | Trajectory regime | Scenes / attempts | Accepted targets (train split) | Guidance targets | Notes |
| --- | --- | ---: | ---: | --- | --- |
| `.cache/silverstone trajs` (`0fabf0ae`) | **d*** baseline: laterally sampled start/goal endpoints; local Gaussian detour around obstacle | 10,000 / 10,000 | 9,486 (4,734) | nominal only | 94.86% attempt acceptance; mean/min clearance 0.1456 / 0.1200 m |
| `.cache/silverstone-trajs-20k` (`e09ac0a3`) | **d*** baseline with laterally sampled endpoints; local obstacle detour | 20,000 / 20,000 | 18,828 (9,399) | nominal only | 94.14% acceptance; mean/min clearance 0.1462 / 0.1200 m |
| `.cache/silverstone-trajs-50k` (`f6fa1bb9`) | **d*** baseline with laterally sampled endpoints; local obstacle detour | 50,000 / 50,000 | 47,195 (23,610) | nominal only | 94.39% acceptance; mean/min clearance 0.1463 / 0.1200 m |
| `.cache/silverstone-matched-10k` (`71c17c6e`) | **d*** baseline with laterally sampled endpoints; nominal/left/right labels share each sampled scene | 10,000 / 30,000 | 20,420 (10,177) | nominal 9,423; left 5,513; right 5,484 | 68.07% acceptance; 9,329 scenes have multiple accepted maneuvers |
| `.cache/silverstone-matched-20k` (`268c1d8f`) | **Centerline** baseline: start/goal lie on centerline; spline residual/local Gaussian detour represents obstacle avoidance | 6,667 / 20,001 | 13,988 (6,955) | nominal 6,432; left 3,812; right 3,744 | 69.94% acceptance; 6,366 scenes have multiple accepted maneuvers; mean/min clearance 0.1657 / 0.1200 m |

`d*` interpolates the endpoint lateral offsets along the forward track arc. The
current centerline regime projects the decoder baseline to the forward centerline
and anchors the physical endpoints; it deliberately removes `d*` as a trajectory
coordinate. A matched dataset contains shared scene alternatives, but a run uses
only complete nominal/left/right triplets when its frozen setting says
`require_matched_triplets = true`.

## Recovered training configuration

The exact TOML text was not copied into a checkpoint, but each completed
`metadata.json` preserves its semantic `[model]` architecture and `[training]`
settings. It also stores the training-source SHA-256; the prefixes below identify
the implementation version used. Thus the tables below recover all operational
settings, while comments, whitespace, TOML key ordering, and any unused TOML
fields cannot be reconstructed.

All listed runs use the same frozen circuit model fields: `horizon=96`,
`dt=0.4`, `context_width=64`, and a Silverstone track with radii `[32, 36]`,
half-widths `0.7`, and resolution `256`. Unless a row says otherwise, their
training fields are `epochs=200`, `batch_size=16`, `learning_rate=0.0005`,
`weight_decay=0.0001`, `grad_clip=1.0`, `unconditional_probability=0.15`,
`validation_fraction=0.2`, `patience=35`, `threads=2`, `device="cpu"`, and
`seed=20261112`.

| Checkpoint(s) | Recovered model fields | Recovered loss / matched fields | `train.py` source SHA prefix |
| --- | --- | --- | --- |
| `trained-silverstone` | width 128, depth 3, knots 24 | Pre-geometry-loss metadata; no auxiliary settings recorded | `112e01867db0` |
| `trained-silverstone-256-width` | width 256, depth 3, knots 48 | Pre-geometry-loss metadata; no auxiliary settings recorded | `112e01867db0` |
| `trained-silverstone-48-knots` | width 128, depth 3, knots 48 | Pre-geometry-loss metadata; no auxiliary settings recorded | `112e01867db0` |
| `trained-silverstone-20k` | width 256, depth 3, knots 48 | Pre-geometry-loss metadata; no auxiliary settings recorded | `112e01867db0` |
| `trained-silverstone-20k-24` | width 256, depth 3, knots 24 | Pre-geometry-loss metadata; no auxiliary settings recorded | `112e01867db0` |
| `trained-silverstone-20k-24-4` | width 256, depth 4, knots 24 | Pre-geometry-loss metadata; no auxiliary settings recorded | `112e01867db0` |
| `trained-silverstone-50k-48-256` | width 256, depth 4, knots 24 | Pre-geometry-loss metadata; no auxiliary settings recorded | `27c7a0edbca8` |
| `trained-silverstone-20k-mixed-loss`, `trained-silverstone-heun128` | width 256, depth 4, knots 48 | `focus_weight=10`, `clearance_weight=2`, `focus_sigma_m=0.5`, `clearance_margin_m=0.12`, `require_matched_triplets=false` | `79d72f1dccdd` |
| `trained-silverstone-matched-10k` | width 256, depth 4, knots 48 | `focus_weight=10`, `clearance_weight=2`, `focus_sigma_m=0.5`, `clearance_margin_m=0.12`, `require_matched_triplets=true` | `79d72f1dccdd` |
| `trained-silverstone-matched-20k` | width 256, depth 4, knots 48 | `focus_weight=0`, `clearance_weight=0`, `focus_sigma_m=0.5`, `clearance_margin_m=0.12`, `require_matched_triplets=false` | `79d72f1dccdd` |
| `trained-silverstone-matched-20k-mixed` | width 256, depth 4, knots 48 | `focus_weight=10`, `clearance_weight=2`, `focus_sigma_m=0.5`, `clearance_margin_m=0.12`, `require_matched_triplets=false` | `79d72f1dccdd` |

The evaluation TOML was not embedded as a separate object, but each
`evaluation.json` records the sampler that actually produced the reported
metrics: all rows use Heun with 32 steps except `trained-silverstone-heun128`,
which uses 128 steps.

## Completed checkpoint results

Architecture is `width / depth / spline knots`; all models use context width 64.
`MSE` is best held-out flow-matching MSE. `Objective` includes geometric terms
where recorded. Evaluation uses the stated Heun sampler; latency is mean sampling
latency in milliseconds. Collision fraction and minimum clearance are diagnostic
metrics only, not release criteria.

| Checkpoint directory | Dataset | Architecture | Training objective | Fit / validation targets | Best epoch / MSE / objective | Eval ADE (m) | Collision fraction | Latency (ms) |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| `trained-silverstone` | `0fabf0ae` | 128 / 3 / 24 | FM only | 3,787 / 947 | 197 / 0.658657 / — | 0.06784 | 0.6705 | 39.04 |
| `trained-silverstone-256-width` | `0fabf0ae` | 256 / 3 / 48 | FM only | 3,787 / 947 | 187 / 0.519655 / — | 0.06555 | 0.6441 | 80.93 |
| `trained-silverstone-48-knots` | `0fabf0ae` | 128 / 3 / 48 | FM only | 3,787 / 947 | 187 / 0.716920 / — | 0.08198 | 0.7054 | 38.38 |
| `trained-silverstone-20k` | `e09ac0a3` | 256 / 3 / 48 | FM only | 7,519 / 1,880 | 200 / 0.402963 / — | 0.05972 | 0.5793 | 39.89 |
| `trained-silverstone-20k-24` | `e09ac0a3` | 256 / 3 / 24 | FM only | 7,519 / 1,880 | 200 / 0.416686 / — | 0.05545 | 0.5388 | 38.25 |
| `trained-silverstone-20k-24-4` | `e09ac0a3` | 256 / 4 / 24 | FM only | 7,519 / 1,880 | 199 / 0.401631 / — | 0.05470 | 0.5293 | 40.88 |
| `trained-silverstone-20k-mixed-loss` | `e09ac0a3` | 256 / 4 / 48 | FM + focus 10 + clearance 2 | 7,519 / 1,880 | 200 / 0.550373 / 0.745015 | 0.06584 | 0.4766 | 39.95 |
| `trained-silverstone-heun128` | `e09ac0a3` | 256 / 4 / 48 | FM + focus 10 + clearance 2 | 7,519 / 1,880 | 200 / 0.550373 / 0.745015 | 0.06588 | 0.4755 | 210.51 |
| `trained-silverstone-50k-48-256` | `f6fa1bb9` | 256 / 4 / 24 | FM only | 18,888 / 4,722 | 199 / 0.316003 / — | 0.05088 | 0.4845 | 40.53 |
| `trained-silverstone-matched-10k` | `71c17c6e` | 256 / 4 / 48 | FM + focus 10 + clearance 2; **complete triplets required** | 1,953 / 489 | 191 / 1.002981 / 1.238990 | 0.09046 | 0.5153 | 64.68 |
| `trained-silverstone-matched-20k` | `268c1d8f` | 256 / 4 / 48 | FM only; matched dataset but **triplet filtering disabled** | 5,574 / 1,381 | 196 / 0.472875 / 0.472875 | 0.06142 | 0.3396 | 43.90 |
| `trained-silverstone-matched-20k-mixed` | `268c1d8f` | 256 / 4 / 48 | FM + focus 10 + clearance 2; matched dataset but **triplet filtering disabled** | 5,574 / 1,381 | 185 / 0.740365 / 0.866381 | 0.05095 | 0.1434 | 50.09 |

## Interpretation boundaries

- Compare checkpoints directly only when their dataset, representation, objective,
  sampler, and validation split match. In particular, the centerline matched-20k
  results are not an apples-to-apples continuation of the older `d*` datasets.
- The current experiment configuration is a centerline, matched-label setup. It
  should be regenerated and retrained under its frozen configuration before using
  a result as the baseline for a new comparison.
- `.cache/trained-silverstone-matched-10k-fm-only` is intentionally excluded from
  the table because it has neither `metadata.json` nor `evaluation.json`; it is
  not a completed, auditable training result.
