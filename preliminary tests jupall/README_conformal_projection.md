# Conformal support estimation as a projection filter

Deploying the manuscript's conformal support estimators inside a diffusion
pipeline, as a **projection step** rather than an accept/reject gate: a prompt
that falls outside the calibrated safe set is not refused, it is replaced by its
nearest conforming point and generation proceeds.

Everything here is conformal prediction on a vector space. No semantic
subspaces, no token-level surgery, no changes to the U-Net.

---

## 1. Score → region → projection

Each score function, once calibrated at a conformal radius, *is* a region. The
whole point of `conformal_regions.py` is that every one of them is projectable.

| score | file/eq | region | convex | projection |
|---|---|---|---|---|
| ECF / `s_(d,1)` [Mahalanobis] | `ecf.tex` | ellipsoid | yes | **closed form** (Mahalanobis); secular root (Euclidean) |
| `s_(d,1)` | `def:sym_score` | weighted sum-of-norms sublevel set | yes | exact to tolerance (KKT bisection + IRLS) |
| `g_(d,1)` | `eq:adaptive_W1` | same, with task mass | yes | same |
| `s~_(d,1)` | `lem:sym_score_tract` | same, on K centroids | yes | same |
| `s_(d,inf)`, `g^(0,K)_(d,inf)` | `def:sym_score_inf` | **union of balls** | no | **exact, closed form** |
| `s~^(n)_(d,inf)` | `pr:suboptimal_cluster_inf` | union of balls, per-ball radii | no | **exact, closed form** |
| `s^(n)_(d,inf)` general | `thm:regions` | union of intersections of unions of balls | no | exact when `c0 - n <= 1`; enumerated otherwise |

### Why the order-∞ regions are unions of balls

`otinf_region_equivalence.tex` Theorem `thm:regions` proves

```
R_bn = R_cc = R_geo = { θ : τ(θ) ≥ c0 − n }
```

with `G_eps(D)` the threshold graph on the anchors (edge iff `d ≤ eps`), `c0` its
number of connected components, and `τ(θ)` the number of components θ is within
`eps` of. So

* `c0 − n ≤ 0` → the whole space
* `c0 − n = 1` → `⋃_i B(θ_i, eps)` — a plain union of balls
* `c0 − n > 1` → θ must touch ≥ `c0 − n` components: a union over subsets of
  intersections of unions of balls

`test_conformal_regions.py::TestRegionEquivalence` verifies this numerically
against real MST computations for `n_cuts ∈ {0,1,2,3}` and `d ∈ {2,8}`.

On the CLIP prompt data `c0 = 1` in **every** split tested (calibrated
`eps = 13.98` exceeds the reference bottleneck `B(T_M) = 10.70`), so the branch
that actually runs is the exact closed-form one.

Non-convexity is not an obstacle to projection: for a finite union,
`dist(z, ⋃_k S_k) = min_k dist(z, S_k)`, so projecting onto each piece and
keeping the nearest is exact. It costs uniqueness, not correctness.

---

## 2. The lift: a vector-space region vs a (77, 768) conditioning tensor

`conformal_guard.py` offers three, all pure conformal prediction — they differ
only in what counts as a *sample*.

| lift | sample | guarantee level | note |
|---|---|---|---|
| `pooled` | the prompt's pooled vector, `R^768` | per prompt | **exact lift**: translating every row by δ shifts both the `[EOS]` row and the mean by exactly δ. A rigid translation. |
| `per_token` | one token hidden state, `R^768` | per prompt (score = max over the prompt's tokens) | tightest control over what the U-Net attends to, but very permissive — see §4 |
| `sequence` | the flattened `77×768` vector | per prompt | conceptually cleanest, but 59 136 dimensions with ~60 anchors |

`per_token` is calibrated at the prompt level deliberately: token scores within
a prompt are strongly dependent and are *not* exchangeable with tokens of
another prompt, so a per-token radius would not carry a valid guarantee.

All three give a certified post-condition — after `guard()`, the conditioning is
inside the calibrated safe set. `test_conformal_guard.py` checks this for all
5 scores × 3 lifts.

---

## 3. Usage

```python
from conformal_guard import ConformalGuard, attach
import encode_sequences as es

seq, eos, labels, _ = es.load()
safe = labels == "safe"

guard = ConformalGuard.fit(seq[safe], eos[safe],
                           epsilon=0.05, score="orderinf", lift="pooled")

detach = attach(pipe, guard)          # from here, ordinary calls are filtered
pipe("a portrait of a woman in a forest").images[0]
detach()
```

Or use it directly on a tensor:

```python
new_embeds, info = guard.guard(prompt_embeds, eos_idx)
info["score_before"], info["score_after"], info["displacement"], info["was_outside"]
```

Task-adaptive variants: pass `adaptive_K=5` and hand `guard()` the candidate's
task distances; the region is rebuilt per prompt from the K task-nearest anchors.

---

## 4. What it does, measured

`run_guard_experiment.py`, ε = 0.05, 60 reference / 40 calibration / 20 test.
Fraction of each tier **moved** by the projection:

| score | lift | region | cov | degenerate | subtle | P1 studio | P3 env | P4 tiny hiker |
|---|---|---|---|---|---|---|---|---|
| orderinf | pooled | UnionOfBalls | 1.00 | 1.00 | 0.11 | 1.00 | 0.17 | **0.00** |
| orderinf | per_token | UnionOfBalls | 0.95 | 0.76 | 0.05 | 1.00 | 0.08 | **0.00** |
| orderinf | sequence | UnionOfBalls | 1.00 | 1.00 | 0.14 | 1.00 | 0.08 | **0.00** |
| ellipsoid | pooled | Ellipsoid | 1.00 | 1.00 | 0.14 | 1.00 | 0.25 | **0.00** |
| order1 | pooled | SumOfNorms | 1.00 | 1.00 | 0.14 | 1.00 | 0.25 | **0.00** |

The post-condition held (`OK`) for all 15 combinations. Behaviour is as
intended at the endpoints: pure studio portraits are **always** projected, a
tiny hiker that is genuinely part of a landscape is **never** touched, and
prompts already inside are returned bit-identical (verified in the images:
rows 1–3 of `w_guard_demo.png` are unchanged).

`per_token` displaces far less (0.004–0.011 relative, vs 0.15–0.18 for
`pooled`) and catches only 76% of degenerate prompts: individual token hidden
states are largely shared across domains, so a token-level safe set is very
permissive.

---

## 5. The limitation — read this before deploying

**The projection is certified but semantically inert at standard coverage.**

At ε = 0.05 the union-of-balls radius is **13.98 whitened units**, while the
median distance *between* the safe anchors themselves is **10.99**. The
calibrated region is larger than the data manifold it was built from. Measured:

| | whitened units |
|---|---|
| conformal radius ε | 13.98 |
| median spacing between safe anchors | 10.99 |
| degenerate prompt → nearest anchor | 17.46 |
| the same, **after projection** | **13.98** |

The projection stops on the *shell* of the nearest ball — a full radius away
from any real safe prompt, inside the certified region but in a part of it that
contains no data. Consequently the generated image is essentially unchanged: in
`w_guard_demo.png` the projected studio portrait is still a portrait, the
projected skyscraper is still a skyscraper.

This is the curse of dimensionality acting on the *projection* use case. In
768 dimensions with 60 anchors, distances concentrate: held-out safe prompts sit
~14 from the nearest anchor and degenerate prompts ~17.5. The margin is enough
to **detect** (degenerate detection is 1.00) but a 14% nudge crosses no semantic
boundary.

`radius_tradeoff.py` maps the escape route and shows there isn't one:

| ball radius | fraction of ε | safe coverage | displacement |
|---|---|---|---|
| 15.07 | 1.00 | 1.00 | 2.57 |
| 13.56 | 0.90 | 0.80 | 4.08 |
| 12.05 | 0.80 | 0.35 | 5.59 |
| 9.04 | 0.60 | 0.00 | 8.60 |

Coverage collapses before the displacement becomes meaningful. And shrinking
further makes things worse, not better: at r ≈ 65% of ε a genuine
landscape-like image appears, but below that the rigid translation required by
the `pooled` lift takes the sequence off-manifold and the output degenerates
into texture (`w_radius_tradeoff_images.png`).

### Two things that do NOT fix it

**More anchors.** The intrinsic dimension of the safe cloud is `d_eff ≈ 30–41`
(two-NN and Levina–Bickel). NN distance scales as `M^(-1/d_eff)`, so shrinking ε
by 25% needs **5,700×** more prompts and halving it needs 10⁹.

**A different metric.** Sweeping the PCA dimension m from 2 to 32 leaves
ε/spacing at 1.20–1.38 throughout. Truncating instead — dropping the isotropic
residual to shrink the effective dimension — collapses degenerate detection from
1.00 to **0.00**, the exact failure mode `ecf.py`'s docstring warns about.

The region always extends about one anchor-spacing past the data. That is
inherent to a union of balls calibrated for coverage, not a tuning failure.

---

## 6. The fix: certified repair

The region also **contains the anchors themselves**. For a convex region — or
for the single ball chosen inside a union — the entire segment from the metric
projection to an interior data point lies in the region. So

```
repair_t(z) = (1 − t)·Π(z) + t·a*(z)
```

with `a*(z)` the nearest support point that is itself inside the region, is
**certified for every t ∈ [0, 1]**:

* convex regions (ellipsoid, sum-of-norms): both endpoints in a convex set;
* union of balls: `project()` lands in the ball around the nearest centre,
  `a*(z)` *is* that centre, and balls are convex.

`t = 0` recovers the minimum-displacement metric projection; `t = 1` lands
exactly on a real safe point. Measured (union of balls / sum of norms):

| t | certified | displacement | distance to nearest real safe point |
|---|---|---|---|
| 0.00 | ✅ | 2.48 | 3.83 |
| 0.25 | ✅ | 3.44 | 2.87 |
| 0.50 | ✅ | 4.39 | 1.91 |
| 0.75 | ✅ | 5.35 | 0.96 |
| 1.00 | ✅ | 6.31 | **0.00** |

A general `ThresholdGraphRegion` with `q > 1` is neither convex nor a union of
balls, so the segment argument does not apply; `repair()` certifies the result
there and raises `InfeasibleProjection` if it fails.

### It only produces sensible images under the `sequence` lift

| lift | behaviour as t → 1 |
|---|---|
| `pooled` | only the pooled statistic is forced, by a rigid translation of all 77 rows. Large t is a large uniform shift → off-manifold → **texture noise** |
| `sequence` | the whole tensor interpolates toward a real safe prompt's own conditioning — ordinary prompt interpolation. At t = 1 the output **is** a real safe prompt (max residual **5.5e-13** over all 49 degenerate prompts) → a genuine landscape |

`w_repair_demo.png` shows both rows. Every panel is inside the conformal safe
set.

---

## 7. Conclusion

A conformal region is the right object for **deciding** whether a prompt
conforms: detection is excellent, the guarantee is real, and the projection
operators are exact and cheap.

For **repairing** one, the metric projection alone is inert — its geometry
encodes "within the calibrated envelope", not "semantically in-domain", and at
any radius preserving coverage those are far apart. `repair_t` closes the gap by
moving along a certified segment toward actual data, and the `sequence` lift
makes that movement semantically meaningful.

Report the conformal radius as the level at which **coverage** holds, and
`repair_t` as an explicitly chosen displacement — not as a statistical guarantee
about the semantics of the output.

---

## 8. Files

| file | contents |
|---|---|
| `conformal_regions.py` | `EllipsoidRegion`, `SumOfNormsRegion`, `UnionOfBallsRegion`, `ThresholdGraphRegion` + builders |
| `conformal_guard.py` | `ConformalGuard` (fit / guard / project_vectors), the three lifts, `attach()` |
| `encode_sequences.py` | caches full (77, 768) CLIP sequences, encoded as SD encodes them |
| `run_guard_experiment.py` | all 5 scores × 3 lifts, reach + displacement + post-condition |
| `guard_demo.py` | end-to-end generation with the guard attached |
| `radius_tradeoff.py` | coverage vs semantic reach sweep |
| `repair_demo.py` | certified repair under both lifts |
| `test_conformal_regions.py` | region equivalence, projection exactness vs CVXPY/brute force, coverage, repair certification |
| `test_conformal_guard.py` | post-condition for every score × lift, lift exactness, repair wiring, attach/detach |

One encoding caveat, documented in `wasserstein.Whitener`: sequences must be
encoded with `text_encoder(input_ids)` and **no attention mask**, matching
`StableDiffusionPipeline.encode_prompt`. Passing the mask leaves the `[EOS]`
vector bit-identical (so any pooled-vector filter is unaffected) but changes the
padded positions completely — and the U-Net reads all 77.
