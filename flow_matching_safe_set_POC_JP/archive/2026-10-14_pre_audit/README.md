# flow_matching_safe_set_POC

Proof of concept: the order-1 Wasserstein conformal region of the manuscript
(`meta_full_paper/`), extended from weight vectors to time-dependent vector fields
(diffusion models and linear flow matching). The goal is to keep a newly trained field close to
historical *safe* fields.

## Contents

| path | what |
|---|---|
| `flow_matching_safe_set_POC.tex` / `.pdf` | the note: definitions, all statements with proofs, toy experiments (Sections 1-7); prompt-conditioned analysis (Section 8); catalogue of design choices (Section 9); experiment plan (Section 10) |
| `safe_prompt_fields.tex` / `.pdf` | the distilled note (23 pages): mathematical preliminaries (Section 2: measures, $L^2$ spaces of fields, projections, conditional expectation, flows, couplings, convexity, exchangeability), linear flow matching and reflow (Section 3, with Figure 1 and Table 1 comparing reflow with standard flow matching), the order-1 region of safe fields and its certificates (Sections 3-7), POC evidence (Section 8) and the E1 test, step by step with one figure per step (Section 9) |
| `refs.bib` | bibliography. The first 6 entries are copied verbatim from `meta_full_paper/ref.bib`; all the others were checked against their sources on 2026-10-09, and Part 4 (references of `safe_prompt_fields.tex`, Section 2, and a corrected `villani2009optimal`) on 2026-10-14 (source URLs are in the comments) |
| `custom_2020.sty`, `informs2014.bst` | copied from `meta_full_paper/`, so the folder compiles on its own |
| `figures/` | 9 figures (PDF for LaTeX, PNG previews), made by `code/make_figures.py` |
| `tables/` | `numbers.tex` (every number quoted in the text, as macros) and 4 table bodies, made by `code/make_tables.py` |
| `results/` | caches, per-split JSON files and logs of all runs |
| `code/` | all scripts (see below) |

## Build the PDF

```
pdflatex flow_matching_safe_set_POC && bibtex flow_matching_safe_set_POC \
  && pdflatex flow_matching_safe_set_POC && pdflatex flow_matching_safe_set_POC
```

The only expected warning is the `stmry` bold-font substitution, the same as in the other notes.

## Reproduce the results (from `code/`)

All scripts write to `../results/`. Seeds are fixed and every script is deterministic.

| order | script | content | time |
|---|---|---|---|
| 1 | `python part_a.py` | pilot A: diffusion with exact Gaussian-mixture scores; pairwise forward/reverse path KL; contaminated fields; P(A) | 6.5 min |
| 2 | `python part_a_analyze.py` | pilot A: coverage, exclusion, certificates over 200 splits | 10 s |
| 3 | `python part_c_pre.py` | pilot C: linear flow-matching fits (random Fourier features), fixed-Gram and on-policy regions, 100 splits | 18 min |
| 4 | `python part_c_ft.py j0 j1` | few-shot fine-tuning on test tasks `j0..j1-1` (we ran 8 workers: 0 4, 4 8, ..., 28 32) | 20 min with 8 workers |
| 5 | `python part_d.py` | pilot D: the same linear fits sampled with the marginal-preserving SDE; KL regions; certificates | 11 min |
| 6 | `python example_fits.py 8` and `python time_profiles.py` | data for the example figures | 2 min |
| 7 | `python make_figures.py` and `python make_tables.py` | figures, tables and all quoted numbers | 1 min |

`python part_c_agg.py` prints a summary of the fine-tuning runs.

The pilots were first run from a session scratchpad and then copied here. The two scripts that
were changed afterwards were checked against the original runs. `part_c_ft.py` was refactored
into importable functions and reproduces the original numbers to 2e-13. `part_d.py` gained
code that saves its arrays and reproduces its original log exactly.

## Sections added after the toy study (not yet tested)

Section 8 covers prompt-conditioned flow matching, where safe prompts induce safe fields:
- Straight flows collapse time onto the t = 0 slice (Lemma 7).
- The linear interpolant alone does not make the learned flow straight (Remark 5).
- Map metrics come with a Wasserstein certificate that has no Grönwall factor (Theorem 3).
- Detection scales as √m·Δ (Proposition 7).
- A prompt can be forced into the safe set by one-step blending (Proposition 8) or by projecting its embedding (Proposition 9).
- Local safe sets handle diverse prompts.

Section 9 is a catalogue of design choices: costs (Table 5), regions, enforcement, calibration and practical certificates.

Section 10 is the experiment plan: hypotheses H1-H6, a common protocol, and experiments E1-E6 (Table 6).

## Main findings (toy, 2-D)

- Coverage of held-out safe fields is at the nominal 0.90 for every cost.
- The direction of the path divergence decides what the region excludes:
  - Forward costs, measured along the historical samplers, are convex and cheap. They admit
    contaminated fields with 3% of their mass in the unsafe disk (58% admitted with exact scores).
  - Reverse costs, measured along the candidate's own paths, exclude all of them.
- Certificates (Theorems 1-2) hold on 200/200 splits.
  - With exact scores: the reverse-KL bound is 3.1% against a worst actual 1.2%; the TV bound is 28%.
  - With linear fits: the reverse-KL bound is 13% (worst actual 1.5%); the TV bound is 55%.
- Linear flow matching reduces to the manuscript's Euclidean order-1 region after whitening by
  the average historical Gram matrix. Constrained training is an SOCP (about 0.3 s per solve).
- Few-shot fine-tuning, 12 samples:
  - The constrained fit is 3.5-5x better than least squares in MMD².
  - With 3 extra unsafe samples, the unsafe mass drops from 13.8% (least squares) to 4.7%
    (fixed Gram) and 2.5% (on-policy).
  - Oracle-tuned shrinkage towards the region centre is better still in this concentrated
    family.

## E1: prompt-conditioned toy (`safe_prompt_fields.tex`, Section 9)

A linear flow-matching model affine in a frozen 43-dimensional prompt embedding, trained on safe and
unsafe prompts (rotation, radius, unsafe mass m), 1-rectified and reflowed. It tests the method as
written, then the improvements (nearest-safe-field weights, truncated L^1 and tail map distances)
against prompt-space regions, including a white-box attack on the embedding. Section 9 follows one running
example through the pipeline step by step (prompt, vector field, outputs, score, region, enforcement, certificate,
attack), one figure per step. Results go to
`results/e1/`; `fit.npz` is about 200 MB (it stores the interpolation Gram matrix of every safe prompt).

| order | script | content | time |
|---|---|---|---|
| 1 | `python e1_fit.py` | fit the 1-rectified model, reflow it, simulate the 700-prompt pool for three model/sampler configurations | 2.5 min |
| 2 | `python e1_eval.py 100` | the method as written: coverage and exclusion over 100 splits, enforcement and certificates on one split | 5 min |
| 3 | `python e1_improve.py screen 100` | screen of weights (uniform, nearest fields, prompt kernel) and distances (L^p, truncated, tail) | 1 min |
| 4 | `python e1_phase2.py 5` | the improvements against the method as written and prompt-space regions: enforcement, certificates, attack | 35 min |
| 5 | `python e1_tables.py` and `python e1_figures.py` | table bodies, number macros (`tables/e1_numbers.tex`, `tables/e1_detection.tex`, `tables/e1_certificates.tex`) and the exclusion figure | 1 min |
| 6 | `python e1_walkthrough.py` | the step-by-step figures of Section 9 (task, fields, time, coupling, scores, slice, certificate, attack) and their numbers (`tables/e1_walk_numbers.tex`); the first run caches a slice of field space and a replay of the attack (checked against `phase2.json`) in `results/e1/walkthrough.npz`; `--recompute` rebuilds it | 4 min, then 15 s |

`python e1_report.py` and `python e1_phase2_report.py` print the full summaries.

`python make_reflow_figure.py` draws Figure 1 of `safe_prompt_fields.tex` (standard flow matching against reflow on the
ring, with the exact marginal velocity from Bayes' formula) and writes `tables/reflow_numbers.tex`, including the
fraction of reflow pairs dropped in E1, read from `results/e1_fit.log` (10 s).

## E2: static and adaptive regions, on prompts and on fields (`safe_prompt_fields.tex`, Section 10)

New prompts on the E1 models; tuning pool and evaluation pool are disjoint. Results go to `results/e2/`.

| order | script | content | time |
|---|---|---|---|
| 1 | `python e2_pool.py` | 1000 safe prompts (300 tuning, 700 evaluation), 200 unsafe bases x 4 masses, admission grid | 3.5 min |
| 2 | `python e2_regions.py 50 200` | distances (prompt, pull-back, field), static and tuned adaptive weights, anchor sweep, maps | 2 min |
| 3 | `python e2_enforce_attack.py 3` | back-off and white-box attack for every region | 9 min |
| 4 | `python e2_nuisance.py` | refit with a nuisance token in the prompt; regions against its strength | 7 min |
| 5 | `python e2_figures.py` | figures `fig_e2_*` (mechanism, detection and tuning, density, attack, nuisance), the body of Table `tab:e2_summary` (`tables/e2_summary.tex`) and the quoted numbers (`tables/e2_numbers.tex`) | 30 s |
