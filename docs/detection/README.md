# Detection workflow: prompts to AUROC and AUPRC

Completed five-category nudity detection and repair results are consolidated in
the [results README](../five-category-nudity/README.md).

Safe and unsafe prompts -> frozen flow-field signatures -> reference distances
-> validation calibration -> held-out rankings and metrics.

This describes the five detection methods used in the five-category nudity
experiment. The pretrained SD3 Medium model stays fixed. The target is the
dataset's prompt label: safe = 0, unsafe = 1.

1. **Separate training, validation, and test prompts.** Keep paired prompts and
   all variants of a scenario in the same split. Training prompts define the
   reference sets and probe geometry; validation prompts calibrate thresholds;
   test prompts measure performance. For 1,000 pairs, the split contains
   700 training, 100 validation, and 200 test prompts **per class**.

2. **Represent each prompt by its effect on the model's flow.** At a shared,
   frozen set of latent states and times, evaluate the guided prompt effect:

   $$g_p(x,t)=\gamma\,[v(x,t\mid p)-v(x,t\mid\varnothing)].$$

   Here, $v$ is the model velocity, $\varnothing$ is the empty prompt, and
   $\gamma=7.5$ is the guidance scale. Collect these vectors into a signature.
   Two probe geometries were tested: **trajectory probes**, using eight
   safe-training-conditioned paths at eight intermediate times (64 probes), and
   **source-slice probes**, using the same eight initial Gaussian states at
   $t=0$. Every prompt is evaluated at the same probes within each geometry.

3. **Measure prompt-to-reference distances.** For query $p$ and training
   reference $a$, let $\delta_j=\|g_p(x_j,t_j)-g_a(x_j,t_j)\|_2$, with the norm
   spanning all latent coordinates. Compare two reductions across probes:

   $$d_{\mathrm{RMS}}(p,a)=\sqrt{\frac{1}{J}\sum_j\delta_j^2},
   \qquad d_{\max}(p,a)=\max_j\delta_j.$$

   RMS measures overall deviation; maximum deviation captures the largest
   local difference. This reduction happens before aggregation over references.

4. **Aggregate distances separately against safe and unsafe references.**
   Denote these class scores by $D_S(p)$ and $D_U(p)$.

   | Tested method | Aggregation over a class's training references |
   |---|---|
   | Trajectory nearest (`euclidean`) | Minimum distance |
   | Trajectory static | Uniform mean distance |
   | Trajectory adaptive | Gaussian-weighted mean distance |
   | Source-slice static | Uniform mean distance |
   | Source-slice adaptive | Gaussian-weighted mean distance |

   Adaptive weights satisfy $w_a(p)\propto\exp[-d(p,a)^2/(2h^2)]$ and sum to one.
   Each class's bandwidth $h$ is its median within-class training-pair distance,
   excluding self-distances. Weights change with the query; bandwidths stay
   fixed. Both RMS and maximum deviation are tested for all five methods.

5. **Use conformal prediction to calibrate acceptance regions.** For sorted
   validation scores $z_{(1)},\ldots,z_{(n)}$, the split-conformal radius is
   $z_{(k)}$, where $k=\lceil(n+1)(1-\alpha)\rceil$ and $\alpha=0.05$.
   Safe-validation scores $D_S$ give the safe radius $r_S$; unsafe-validation
   scores $D_U$ give an initial unsafe radius $r_U$. Each method and probe
   reduction is calibrated separately.

   With frozen training choices and exchangeable safe validation/test prompts,
   the safe-only rule has marginal safe acceptance of at least 95%. This is a
   class-conditional coverage statement for safe prompts, not an estimate of
   the probability that a particular prompt is unsafe. It does not ensure
   detection of unsafe prompts.

   The joint rule also caps $r_U$ using the lower order statistic of
   safe-validation distances to unsafe references: rank
   $\lfloor(n_S+1)0.05\rfloor$. The final radius $\tilde r_U$ is the smaller of
   this value and $r_U$. This reduces safe-prompt exclusion by the unsafe
   region; ties can enlarge that region. The resulting joint rule does not
   inherit the safe-only 95% acceptance guarantee.

6. **Evaluate two detection rules on held-out test prompts.**

   | Rule | Flag as unsafe when | Continuous ranking score |
   |---|---|---|
   | Safe-only (`safe_attraction`) | $D_S>r_S$ | $D_S/r_S$ |
   | Joint (`safe_attraction_unsafe_avoidance`) | $D_S>r_S$ or $D_U\leq\tilde r_U$ | $\max(D_S/r_S,\;2-D_U/\tilde r_U)$ |

   Higher ranking scores indicate greater evidence for the unsafe label.
   **AUROC** measures how often an unsafe test prompt ranks above a safe one
   (ties receive half credit). **AUPRC** is reported as average precision:
   precision averaged over recall increments as the ranking threshold varies.
   Unsafe prompts are the positive class; AUPRC depends on their prevalence.

   Both metrics use continuous scores across thresholds, rather than the single
   calibrated binary decision. Conformal calibration sets the operating point
   and the relative scaling of the joint score; positive scaling alone leaves
   safe-only rankings unchanged. Test labels never fit thresholds or bandwidths.

The full comparison has **5 methods x 2 probe reductions x 2 rules = 20
configurations**. These metrics assess prompt-label discrimination; generated
image safety is not measured by this detection workflow.
