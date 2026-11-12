# Observed synthetic run — 2026-11-09

The delivered [report](example_run/report.json),
[frozen policy and detector calibration](example_run/frozen_policy.json),
[output calibration](example_run/output_calibration.json), and
[paired evaluation](example_run/evaluation.json) were generated with the checked-in
`config.toml`. Artifact hashes, every recorded source hash and the policy hash
were verified against the final implementation.

Policy identity:
`de02c822b5f02094d14c30f2fcdcd15b52242a7e9c850b9285369309b0bd9f7f`.

This is an analytic synthetic field, with no trained trajectory checkpoint.
Execution used Windows 11, Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1 and
Torch 2.14.1+cpu on an Intel Core i7-8550U (4 cores / 8 logical processors).
The experiment limits Torch and BLAS to one thread. Latencies are observed
wall-clock times from one run, not a throughput benchmark or real-time guarantee.

## Independent episodes and detection

| Partition | Safe first-query labels | Unsafe first-query labels |
| --- | ---: | ---: |
| Training/reference | 28 | 20 |
| Detector calibration | 33 | 15 |
| Repaired-output calibration | 13 | 7 |
| Test | 7 | 5 |

Both detector alphas were 0.15. The exact upper ranks were 29/33 for the safe
class and 14/15 for the unsafe class, giving `r_safe=0.034162436356947726` and
`r_unsafe=0.054955087761487754`. All modes used identical scene seeds and paired
diffusion/process randomness. At the first query, safe-only and joint detection
both flagged all 5 unsafe scenes and 3 of 7 safe scenes: unsafe detection 100%,
safe false-positive rate 42.86%. The same nominal flags are diagnostic only in
nominal operation. These small-sample frequencies do not establish a population
FPR bound; joint admission has no `alpha_safe` FPR guarantee.

## Control outcomes

Collision rates are episode rates. Planned collisions include proposals rejected
by the geometric release gate; executed collisions use actual swept segments.
All four policies retain geometric, MPC and deadline gates.

| Metric | Nominal | Safe-only | Joint | Repaired |
| --- | ---: | ---: | ---: | ---: |
| Physical steps | 288 | 199 | 199 | 123 |
| Controls released | 44.44% | 24.12% | 24.12% | 71.54% |
| Fresh repair searches without a feasible schedule | — | — | — | 112 |
| Validated old-reference releases | 0 | 42 | 42 | 77 |
| Planned collision episodes | 5/12 | 0/12 | 0/12 | 0/12 |
| Executed collision episodes | 0/12 | 0/12 | 0/12 | 0/12 |
| Minimum planned clearance (m) | -0.36488 | 0.94959 | 0.94959 | 0.94959 |
| Minimum executed clearance (m) | 0.85902 | 0.94795 | 0.94795 | 0.94795 |
| Goal completion | 0/12 | 6/12 | 6/12 | 11/12 |
| One-step tracking RMSE (m) | 0.84780 | 0.00785 | 0.00785 | 0.00908 |
| Mean integral of steering strength | 0 | 0 | 0 | 0.04065 |
| Mean pipeline latency (ms) | 57.04 | 16.65 | 15.37 | 166.11 |
| p95 pipeline latency (ms) | 220.57 | 51.73 | 52.61 | 198.55 |
| Maximum pipeline latency (ms) | 444.11 | 57.03 | 97.28 | 236.25 |

Tracking error includes steps with rejected final candidates, so nominal error
also measures the gap between a proposed path and the simulator's abstention
stop. It is not solely a comparison of feasible MPC tracking performance.
Per-step steering averages include zero steering on failed fresh repairs and old
reference reuse. Fallback can retain the effect of an earlier repaired reference
without a new diffusion intervention.

Of 112 infeasible fresh repair searches, 77 recovered through freshly validated
old references and 35 ended in abstention. The safe-only and joint policies each
abstained on 151 steps. Nominal had 118 final reference-clearance failures, 40
reference-start mismatches, one solver failure and one incompatible fallback.
No deadline misses were observed under the configured **2-second** budget; the
simulation timestep is 0.4 seconds, so this is not evidence of real-time execution.

## Separately calibrated output scores

Each target uses 20 fresh episodes and `alpha_output=0.2`, hence the 17th ordered
calibration score. One score is the episode minimum-clearance deficit; no-output
steps yield `+inf` for the reference target. The policy was fixed before this
calibration, and these scores never changed the repair criterion or release gates.

| Policy | Reference cutoff | Test reference score coverage | Executed cutoff | Test executed score coverage |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 0.333080 | 9/12 = 75% | -0.964735 | 9/12 = 75% |
| Safe-only | +inf | 12/12, **vacuous** | -0.897801 | 10/12 = 83.33% |
| Joint | +inf | 12/12, **vacuous** | -0.897801 | 10/12 = 83.33% |
| Repaired | +inf | 12/12, **vacuous** | -0.880471 | 9/12 = 75% |

Empirical held-out coverage need not equal the nominal 80% in twelve episodes.
Infinite reference cutoffs are explicitly uninformative. A marginal clearance
score statement does not imply conditional safety after release, closed-loop
invariance, or a bound on the joint detector's safe FPR.

## Observed limitations and verification

The reference catalogue was fitted to initial contexts. Later replanning contexts
often lie outside its narrow safe region, causing many infeasible fresh repairs.
Goal completion relies heavily on old-reference fallback. The synthetic model
also repeatedly bends newly generated paths, whereas fallback tracks an existing
path to completion. These results do not establish that repair generally
outperforms nominal planning. A real study needs representative contextual banks,
a trained motion model, more independent episodes and an independently justified
tracking uncertainty model.

The synthetic guidance contribution is constant over diffusion time; the baseline
depends on the latent state. Its results cannot validate richer nonlinear
diffusion behavior. A separate nonlinear-field test verifies that the sampler
integrates mixed drift rather than interpolating decoded paths.

The final repository suite passed **233 tests**, with one CUDA-only test skipped
because CUDA is unavailable. Repository Ruff checks and `git diff --check` passed.
Coverage review of the new reusable code identified and led to regression tests
for numerical cancellation and rejected encoder data reaching fallback. A
pre-existing Windows path incompatibility in the local Colab staging fixture was
fixed without weakening production path validation or assertions. The real-model
adapter was checked with a local Torch checkpoint; no trained motion checkpoint,
GPU inference, physical vehicle, recursive feasibility or hard real-time claim
was validated.
