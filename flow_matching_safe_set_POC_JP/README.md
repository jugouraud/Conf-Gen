# flow_matching_safe_set_POC

Proof of concept: the order-1 Wasserstein conformal region of the manuscript
(`meta_full_paper/`), extended from weight vectors to time-dependent vector fields
(diffusion models and linear flow matching). The goal is to keep a newly trained field close to
historical *safe* fields.

## Contents

| path | what |
|---|---|
| `flow_matching_safe_set_POC.tex` / `.pdf` | the note: definitions, all statements with proofs, toy experiments (Sections 1-7); prompt-conditioned analysis (Section 8); catalogue of design choices (Section 9); experiment plan (Section 10) |
| `safe_prompt_fields.tex` / `.pdf` | the distilled note (revised on 2026-10-14 after the audit below; Section 11 added on 2026-10-15, Sections 11.8-11.9 on 2026-10-16): mathematical preliminaries (Section 2: measures, $L^2$ spaces of fields, projections, conditional expectation, flows, couplings, convexity, exchangeability), linear flow matching and reflow (Section 3, with Figure 1 and Table 1 comparing reflow with standard flow matching), the order-1 region of safe fields and its certificates (Sections 3-7), POC evidence (Section 8), the E1 test, step by step with one figure per step (Section 9), and the E2 comparison of static and adaptive regions on prompts and on fields (Section 10), and the E3 test on the pretrained text-to-image model Sana 600M with a winter-only safe set, with a method-by-method analysis of what works and what does not, and new methods that decide with a prompt-side distance and repair with a field-side one, with their conformal guarantees (Proposition 11.1) and results (Section 11.8), and the slice methods stated and proved in mathematical detail (Section 11.9) (Section 11; numbers from `tables/e3_numbers.tex`, `tables/e3c_numbers.tex` and the table bodies `tables/e3_*.tex`, `tables/e3c_*.tex`); Section 12 (added on 2026-10-16) adds a second history of known unsafe prompts and a second conformal set, the unsafe set, with its guarantees (Proposition 12.3) and its tests on the same model (numbers from `tables/e3u_numbers.tex` and `tables/e3u_*.tex`); 70 pages |
| `safe_prompt_fields_audit.tex` / `.pdf` | the audit of the version of 2026-10-14, 10:10 (findings F1-F3, M1-M6, R1-R10, H1-H7, and what was confirmed); its numbers come from `tables/audit_numbers.tex`, `tables/audit_probe.tex` and `tables/audit_attack.tex` (`code/audit_checks.py`) |
| `e3_text_to_image.tex` / `.pdf` | the first, stand-alone write-up of experiment E3 (9 pages; Section 11 of `safe_prompt_fields.tex` is the reference version): every mechanism of `safe_prompt_fields.tex`, static and adaptive, on the pretrained text-to-image model Sana 600M with a winter-only safe set; numbers from `tables/e3_numbers.tex` and `tables/e3_*.tex` (`code/e3_figures.py`), figures `figures/fig_e3_*` |
| `archive/2026-10-14_pre_audit/` | the audited note, its PDF, README, `refs.bib`, number macros and the figure scripts the revision changed |
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
| 5 | `python e1_robustness.py` | probe sensitivity (every L^p, truncated and tail score with and without the probe that some prompt's sampler sends beyond radius 4) and the directed search (attack in the five directions that move the probe outputs least) on the E1 regions | 5 min |
| 6 | `python e1_verify.py` | verified deployment of the improved regions (fresh-noise check with Hoeffding or Clopper-Pearson margins, fallback to the nearest anchor), evaluated on independent noise | 45 min |
| 7 | `python e1_tables.py` and `python e1_figures.py` | table bodies (`tables/e1_detection.tex`, `e1_certificates.tex`, `e1_probe.tex`, `e1_verify.tex`), number macros (`tables/e1_numbers.tex`) and the exclusion figure; need steps 5-6 | 1 min |
| 8 | `python e1_walkthrough.py` | the step-by-step figures of Section 9 (task, fields, time, coupling, scores, slice, certificate, attack) and their numbers (`tables/e1_walk_numbers.tex`); the first run caches a slice of field space and a replay of the attack (checked against `phase2.json`) in `results/e1/walkthrough.npz`; `--recompute` rebuilds it | 4 min, then 15 s |

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
| 5 | `python e2_robustness.py` | probe sensitivity of the field regions, the Mahalanobis prompt regions (tuned on the tuning pool; coverage, exclusion, back-off, both attacks, nuisance token), and the directed search on every region but the Euclidean one | 16 min |
| 6 | `python e2_figures.py` | figures `fig_e2_*` (mechanism, detection and tuning, density, attack, nuisance), the body of Table `tab:e2_summary` (`tables/e2_summary.tex`) and the quoted numbers (`tables/e2_numbers.tex`); needs step 5 | 30 s |

## Audit (`safe_prompt_fields_audit.tex`)

After both pipelines (E1 steps 1-6, E2 steps 1-5), `python audit_checks.py` (30 s) writes every number of the audit. Build with
`pdflatex safe_prompt_fields_audit` twice. The audit quotes the audited note through its archived number macros
(`archive/2026-10-14_pre_audit/tables/`), so it keeps its meaning after the revision.

## E3: the safety mechanisms on a pretrained text-to-image model (`e3_text_to_image.tex`)

Sana 600M (512 px, flow matching, `Efficient-Large-Model/Sana_600M_512px_diffusers`, cached locally) on the Apple GPU. Approved
(safe) prompts are winter sceneries only. The requests are summer scenes, scenes whose season is not mentioned ("a pine forest, at
midday"), summer scenes disguised with winter keywords, and safe but unfamiliar requests (winter prompts reworded, winter and summer
prompts of 6 scenes that no approved prompt shows). Regions: prompt (Euclidean, Mahalanobis), source slice (pull-back), field (L^2 as
written, truncated) in CLIP ViT-L/14 features of the outputs, each static and adaptive; enforcement by mixing fields at sampling time
(back-off); plug-in certificates; outputs judged by an independent model (CLIP ViT-B/32) on noise that no score uses. Results go to
`results/e3/`. Section 11 of `safe_prompt_fields.tex` reports them (the stand-alone note `e3_text_to_image.tex` is the first write-up), every number a macro of `tables/e3_numbers.tex` or a table body `tables/e3_*.tex`. Figures of Section 11: `fig_e3_requests`, `fig_e3_detection_map`, `fig_e3_per_request`, `fig_e3_anchors`, `fig_e3_gallery_neutral`, `fig_e3_gallery_summer`, `fig_e3_enforce_map` (one colour per region in all of them: prompt orange, Mahalanobis violet, source slice aqua, field blue; validated for colour-blind readers); the stand-alone note also uses `fig_e3_detection` and `fig_e3_enforce`.

| order | script | content | time |
|---|---|---|---|
| 1 | `python e3_prompts.py`, then `python e3_prompts.py --extend` | prompt grammar and pools (`results/e3/prompts.json`); `--extend` appends the 40 unfamiliar requests once, leaving the first 270 prompts unchanged | 1 s |
| 2 | `python e3_pool.py` | outputs of the 310 prompts on 8 probe and 8 evaluation noises, with their features; one file per prompt in `results/e3/pool/`, resumable | ~1.7 h |
| 3 | `python e3_regions.py` | distances, static and tuned adaptive weights, coverage and exclusion over 200 splits | minutes |
| 4 | `python e3_repairs.py` | mechanism (season gap against scene gap per space, centrality of the pine forests) and number of anchors (M = 30 to 140, 200 splits) | 10 s |
| 5 | `python e3_enforce.py` | back-off of 30 requests in 9 regions, judged, with certificates; one file per (region, request) in `results/e3/enforce/`, resumable | 75 min |
| 6 | `python e3_figures.py` | galleries, detection and enforcement figures, tables `tables/e3_*.tex`, every number in `tables/e3_numbers.tex` | 30 s |

`./e3_run.sh` (from `code/`) runs the six steps and skips what is already done (`caffeinate -i ./e3_run.sh > ../results/e3_run.log
2>&1 &` keeps the Mac awake). State on 2026-10-15: all steps done.

Section 11.8 (methods that decide with one distance and repair with another; added 2026-10-16) runs after these six steps:

| order | script | content | time |
|---|---|---|---|
| 7 | `python e3_combined.py` | joint and Bonferroni regions of the source slice and the field distance over the 200 splits, the median stop level, and how often a scene has no admissible target (`results/e3/combined.json`) | 2 s |
| 8 | `python e3_enforce_combined.py METHOD ...` | enforcement of the new methods on the 30 requests of step 5 and 24 unfamiliar ones, and of the previous static regions on the unfamiliar ones; one file per (method, request) in `results/e3/enforce_combined/`, resumable; without a method name it only assembles `results/e3/enforce_combined.json` | ~3.5 h in all |
| 9 | `python e3_figures_combined.py` | figures `fig_e3c_*`, table bodies `tables/e3c_*.tex`, numbers `tables/e3c_numbers.tex` | 1 min |

`./e3_run_combined.sh` (from `code/`) runs steps 7-9 in order. Run one process at a time: on the Apple GPU, three parallel
processes made each record about four times slower.

Build the stand-alone note (it cites `safe_prompt_fields.tex` through its `.aux`, so build that note first):
`pdflatex e3_text_to_image && bibtex e3_text_to_image && pdflatex e3_text_to_image && pdflatex e3_text_to_image`.

Main findings:
- A season left unspecified is unsafe: Sana fills it with a snow-free scene (0.99 of the images of neutral requests judged unsafe).
- The space decides, the weights adjust. The source slice and the Mahalanobis region exclude every summer request; the field regions
  as written exclude about half and miss the pine forest (snowy pine forests are the most central approved images in CLIP space); the
  prompt region excludes 0.10 (static) to 0.34 (adaptive).
- A region detects what its distance makes large: the Mahalanobis region excludes every safe paraphrase (it detects the wording), the
  prompt and field regions exclude unseen winter scenes (they detect the scene), and only the static source slice detects the season
  alone (it sees the mean colour of the outputs).
- Nearest-anchor weights with 100+ anchors make the field regions exclude 0.99 of the summer requests, but also 0.92 of the safe winter
  prompts of unseen scenes.
- Enforcement keeps the scene only in the field regions (target always of the request's scene; 0.23 of summer images still unsafe,
  from barely excluded requests that the minimal back-off barely changes); the Mahalanobis region makes every image wintry but
  replaces the scene (a snowy lighthouse for 11 of 16 requests). The plug-in certificates are vacuous (bound 1.73 and 1.94): approved
  outputs lie at most 0.13 from the unsafe half-space in CLIP space but 0.71 from one another.

Findings of Section 11.8 (prompt-side decisions with field-side repairs; enforcement on 8 requests per kind of split 0):
- Coverage depends on the decision alone (Proposition 11.1), so the target and the depth of the repair can use other distances.
- Deciding with the static source slice, backing off towards the approved prompt nearest in the field distance, and stopping when
  the deployed outputs are within the median nearest-anchor field distance of that target ("slice->field, local", designed after
  the other results) leaves 0.06 of the summer and 0.02 of the season-unspecified images unsafe (source slice 0.45/0.44, field
  region 0.23/0.14), every target of the requested scene, and the false alarms of the source slice.
- A field target alone makes the images LESS safe (0.64): the same-scene target is far in the source slice, the static score is
  convex along the segment, so the minimal back-off stops earlier. Stopping at the median level of the source slice fixes summer
  (0.05) but not requests just above the threshold (0.28); stopping at the median level of the field distance removes the
  targets of the least central scenes (a third of the scenes; rank correlation 0.95 with centrality).
- A joint region of the source slice and the field distance adds the field's false alarms on new scenes (0.18 against 0.01)
  without adding detection. Mahalanobis->field repairs familiar requests best (0.00/0.02 unsafe, scene kept) but rewrites every
  unfamiliar safe request. The exact projection onto the source-slice region keeps about 0.7 of the request's field and spreads the
  rest over all 30 anchors: the images stay unsafe.

## Section 12: a safe set and an unsafe set (`safe_prompt_fields.tex`, Section 12)

The method of Sections 5-11 knows only approved (safe) fields. Section 12 adds a history of known **unsafe** prompts (explicit summer
prompts) and a second conformal set built from their fields, the **unsafe set**. A request is admitted unchanged only if its field lies
in the safe set and outside the unsafe set; otherwise it is mixed with the nearest admissible approved field (field distance) until both
hold. Both sets are localised with weights read from the prompt embedding (kernel X:c or nearest prompts XN:k of each class), chosen to
make each set as small as possible, i.e. to contain the least of the other class. Proposition 12.3 proves what two sets guarantee
(coverage of each class, the price of the second set on approved prompts and a cap that bounds it, an exactly minimal repair in the
source slice, and what leaving the unsafe set means). Numbers come from `tables/e3u_numbers.tex` and the table bodies `tables/e3u_*.tex`
(`code/e3u_figures.py`); figures `fig_e3u_*`. Nothing of Section 11 is modified: the new prompts and outputs live in separate files.

| order | script | content | time |
|---|---|---|---|
| 1 | `python e3u_prompts.py` | 174 historical unsafe prompts, the next entries of the permutation that drew the summer prompts (so they are exchangeable with the 24 summer requests): 54 for tuning (with the 16 summer tuning prompts) and 120 for evaluation (`results/e3/prompts_unsafe.json`, ids 310-483) | 1 s |
| 2 | `python e3u_pool.py` | their outputs on the probe noises only, with features; one file per prompt in `results/e3/pool_unsafe/`, resumable; `pool_unsafe.npz` | ~40 min |
| 3 | `python e3u_regions.py` | tuning of both sets for size, detection over 200 splits (safe split as Section 11, unsafe split seeds 2000+s) for the rules safe set / unsafe set / two sets / two sets capped, the four cells of every kind of request, and the anchor sweep (M = 30 to 140 per class, pooled, weightings fixed in advance) (`results/e3/regions_unsafe.json`) | 20 s |
| 4 | `python e3u_enforce.py S2 "S>F2" F2 "S>F" "S>F:local" F1 P2` | repairs on the first pooled split with M = 140 anchors of each class (static source slice, nearest-prompt field and prompt sets); one file per (method, request) in `results/e3/enforce_unsafe/`, resumable; without a method name it only assembles `results/e3/enforce_unsafe.json` | ~5 h |
| 5 | `python e3u_segment.py j1,j2,j3` | safe and unsafe scores along the whole repair segment of a few requests (`results/e3/segment_unsafe.json`) | ~5 min |
| 6 | `python e3u_figures.py` | figures `fig_e3u_*`, tables `tables/e3u_*.tex`, numbers `tables/e3u_numbers.tex` | 1 min |

`./e3u_run.sh` (from `code/`) runs the steps in order. The static safe sets of step 3 reproduce the regions of Section 11 to 1e-16
(same splits), which checks the new code against the old.

Findings of Section 12 (detection over 200 splits with 30 anchors of each class; repairs on one split with 140 anchors of each class,
8 requests per kind):
- Each set keeps its coverage of its own class (0.89-0.91 approved in the safe sets, 0.90 unsafe in the unsafe sets), so a new summer
  request of the history's kind is excluded with probability at least 0.90: the first guarantee on unsafe requests of the note.
- Only the source slice separates the two classes (each set contains 0.00 of the other). In the prompt and field distances each set
  contains about half of the other class with 30 anchors, so the two-set rule admits only 0.43-0.45 of the approved prompts; a cap on the
  unsafe threshold restores them but removes the gain. Prompt-adaptive weights (nearest prompt of each class) shrink the field sets only
  when every scene has anchors of both classes (140 anchors: the field's unsafe set claims 0.09 of the approved prompts), and then refuse
  new scenes (0.86 of the new winter scenes excluded).
- Detection: the source slice's two sets raise the exclusion of season-unspecified requests from 0.90 to 0.98 and of disguised ones from
  0.63 to 0.83, with no new false alarm (paraphrases 0.03, new winter scenes 0.02): the best detector of the note. The unsafe set, built from
  explicit summer prompts, recognises only 0.62 of the season-unspecified requests (their images: 0.02 green pixels against 0.27).
- Repairs (slice decides, field target): stopping at the entry into the slice's safe set leaves 0.25 of the summer and 0.61 of the
  season-unspecified images unsafe; also leaving the slice's unsafe set: 0.00 and 0.38 (exactly minimal, no sampling run; the requests the
  unsafe set does not recognise get no depth); also entering the field's safe set and leaving its unsafe set ("slice->field, two sets"):
  0.00 and 0.09, scene kept, no false alarm, against 0.00 and 0.02 for the local stop of Section 11.8 rerun on the same split (a level chosen
  after the results). One request meets all four conditions on the probes and stays unsafe on new noise.
- The field's sets alone repair less safely (0.23 and 0.19; with the safe set only 0.41 and 0.19): a mixture can meet them on 8 probes
  while its other outputs are summer images, and every such repair is rejected by the slice's sets. The prompt's two sets admit every
  season-unspecified request of the split (images all unsafe) and alter every disguised request.
