# Five-category nudity: detection and repair results

Both experiments completed on **2 November 2026**. Trajectory-nearest RMS
detection achieved **0.99895 AUROC** with the joint rule. On the 20-prompt repair
pilot, constant steering reduced model-rated unsafe images from **14/20 to 0/20**,
using substantially more intervention than the temporal methods.

The frozen SD3 Medium generator used 512 x 512 images, CFG 7.5, and noise seed
1729. The dataset contains 1,000 matched safe/unsafe pairs: 200 each for intimate
bedroom scenes, figure-study nudity, fine-art nude photography, nude body
positivity, and nude self portraits. Each category contributes 140 training,
20 validation, and 40 test pairs; scenarios stay within one split.

## Detection

Evaluation uses **200 safe + 200 unsafe test prompts**. Five scorers, two probe
reductions (RMS/maximum), and two decision rules give 20 configurations.
Thresholds use validation data at alpha = 0.05. Safe-only means safe attraction;
joint adds unsafe avoidance. AUPRC is average precision, with unsafe labels
positive and 50% test prevalence.

| Scorer | Reduction | Safe-only AUROC / AUPRC | Joint AUROC / AUPRC | Joint unsafe flagged | Joint safe false positives |
|---|---|---:|---:|---:|---:|
| Trajectory nearest | RMS | 0.9629 / 0.9627 | 0.9990 / 0.9989 | 100% | 11% |
| Trajectory static | RMS | 0.8389 / 0.7815 | 0.9562 / 0.9570 | 79% | 6.5% |
| Trajectory adaptive | RMS | 0.8396 / 0.7871 | 0.9658 / 0.9655 | 79.5% | 6.5% |
| Source-slice static | RMS | 0.5381 / 0.5266 | 0.5594 / 0.5950 | 21.5% | 14% |
| Source-slice adaptive | RMS | 0.5384 / 0.5340 | 0.5720 / 0.6104 | 24% | 14% |
| Trajectory nearest | Maximum | 0.9101 / 0.8784 | 0.9410 / 0.9302 | 67% | 7% |
| Trajectory static | Maximum | 0.7051 / 0.6304 | 0.7469 / 0.7424 | 56% | 25% |
| Trajectory adaptive | Maximum | 0.7129 / 0.6359 | 0.7657 / 0.7587 | 56.5% | 24% |
| Source-slice static | Maximum | 0.5257 / 0.5268 | 0.5482 / 0.5664 | 13% | 10% |
| Source-slice adaptive | Maximum | 0.5332 / 0.5350 | 0.5506 / 0.5817 | 16.5% | 12% |

Trajectory-nearest RMS was strongest on this test set. Its joint rule flagged
200/200 unsafe and 22/200 safe prompts; safe-only flagged 152/200 unsafe and
13/200 safe prompts. RMS outperformed maximum reduction for every scorer/rule
in AUROC; source-slice discrimination remained close to chance. These measure
prompt-label discrimination. The joint rule has no 95% safe-acceptance guarantee.

Sources: full [RMS metrics](../../backend/experiments/five_category_nudity_detection_20261101/run/comparison.csv),
[maximum metrics](../../backend/experiments/five_category_nudity_detection_20261101/run/max_deviation/comparison.csv),
and [detection workflow](../detection/README.md).

## Repair and independent image scoring

The completed `run_unsafe_gpu_native` selected **20 unsafe-labelled test prompts**
(four distinct scenarios per category, selection seed 20261102), without
filtering by detection or image scores. All 20 were flagged by the retained
trajectory-nearest RMS joint detector. Original thresholds stayed fixed;
anchors came from safe training references. Each prompt used 32 Euler steps
and shared noise across R0--R5: **120 generation records and 100 repair pairs**.
Earlier stopped attempts are excluded.

ShieldGemma 2 scores sexual, dangerous, and violence policies. Toxicity is their
maximum score; an image is counted unsafe at toxicity >= 0.5. Each row includes
all 20 prompts, including failed or unchanged repairs.

| Method | Field repairs | Mean integrated strength | Mean toxicity | Mean reduction [95% CI] | Unsafe images |
|---|---:|---:|---:|---:|---:|
| R0: original | 0/20 | 0.0000 | 0.6833 | -- | 14/20 (70%) |
| R1: constant | 20/20 | 0.8572 | 0.0026 | 0.6807 [0.4881, 0.8612] | 0/20 (0%) |
| R2: fixed window | 20/20 | 0.1788 | 0.3678 | 0.3155 [0.0649, 0.5644] | 7/20 (35%) |
| R3: peak deviation | 13/20 | 0.2134 | 0.4338 | 0.2494 [0.0309, 0.4863] | 8/20 (40%) |
| R4: contiguous deviation | 20/20 | 0.4226 | 0.2459 | 0.4374 [0.1965, 0.6642] | 5/20 (25%) |
| R5: optimized smooth | 20/20 | 0.2171 | 0.3350 | 0.3483 [0.0771, 0.6100] | 7/20 (35%) |

See [five steering-schedule graphs](steering-schedules.md) for schematic R1--R5
definitions explaining how strengths, window positions, and lengths are chosen.

Field repair means passing both frozen detector constraints. Integrated strength
is the mean time-integrated steering weight; reduction is original minus repaired
toxicity, with prompt-bootstrap intervals. R1 had the lowest model-rated toxicity;
R2 used the least mean steering among methods repairing all 20 fields. R3 left
seven unresolved cases with unchanged images. R2/R3/R4/R5 also moved 2/1/1/2
originally below-threshold images above threshold, respectively.

Field acceptance did not ensure a below-threshold image score. This pilot has
one noise seed and four prompts per category; semantic preservation and image
quality were unmeasured. Model-rated toxicity is not a calibrated probability
of harm, and these results do not establish a safety/preservation optimum.

Sources: [field metrics](../../backend/experiments/five_category_nudity_repair_20pairs_20261102/run_unsafe_gpu_native/pilot/comparison.csv),
[image-score summary](../../backend/experiments/five_category_nudity_repair_20pairs_20261102/run_unsafe_gpu_native/image_scoring/shieldgemma2/pilot/summary.csv),
[paired scores](../../backend/experiments/five_category_nudity_repair_20pairs_20261102/run_unsafe_gpu_native/image_scoring/shieldgemma2/pilot/pairs.csv),
[scored gallery](../../backend/experiments/five_category_nudity_repair_20pairs_20261102/run_unsafe_gpu_native/image_scoring/shieldgemma2/pilot/report.html),
and [repair workflow](../repair/README.md).
