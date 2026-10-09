# Conf-Gen

Collect CLIP embeddings for COCO captions and evaluate prompt coverage against
the saved COCO region using the I2P benchmark prompts.

## T2I-native safe region

Build the step-1 one-class corpus from DiffusionDB metadata, the local I2P CSV,
and T2I-RiskyPrompt:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_prompt_dataset
.\.venv\Scripts\python.exe -m backend.embeddings.prompt_region --batch-size 64
.\.venv\Scripts\python.exe -m backend.embeddings.prompt_transport
```

The default corpus has 10,000 filtered, unique safe prompts: 7,000 reference,
1,500 calibration, and 1,500 held-out safe test. It also has 3,000 risky
evaluation prompts: 1,500 each from I2P and T2I-RiskyPrompt. The builder writes
`data/prompts/confgen_prompts.parquet`, a CSV copy, and a manifest. Its
`eligible_for_region` flag is true only for the reference and calibration
splits. Matching prompts are excluded from the risky evaluation sample and
replaced by later source rows, without changing the safe sample.

The second command saves the region to `data/prompts/safe_prompt_region.npz`
and safe-coverage/risky-rejection counts to
`data/prompts/safe_prompt_region_report.json`. It fits the PPCA Mahalanobis
metric on final-layer CLIP EOS vectors from the safe reference split and
calibrates the nearest-anchor radius on the safe calibration split. The third
command scores the research README's full-reference order-1 Wasserstein and
adaptive order-infinity methods in the same metric. Order-infinity selects five
task-nearest anchors by exact token-cloud W1 from the full safe-reference set
and uses the zero-cut MST bottleneck. Each method gets
its own 5% conformal budget from safe calibration prompts; the held-out safe
and risky prompts are evaluation only. Scores are cached in
`data/prompts/safe_prompt_transport_full_reference.sqlite3`, and method budgets
and counts are saved to
`data/prompts/safe_prompt_method_report_full_reference.json`. EOS encodings are
cached in `data/prompts/prompt_eos.sqlite3`, so interrupted runs can resume.
Rerun the third command to create these full-reference outputs; the earlier
60-anchor transport report is retained separately and is not loaded by the app.
The older COCO dashboard and files remain a separate baseline.

### Region definition and three prompt selection rules

Let $R=\{z_1,\ldots,z_M\}$ be the CLIP final-layer `[EOS]` vectors of the
safe-reference prompts, and let $z$ be a candidate prompt vector. The reference
vectors alone determine a probabilistic-PCA covariance: up to $m=32$ leading
eigenvectors $v_j$ retain eigenvalues $\lambda_j$, while the remaining
directions share variance $\gamma$. With reference mean $\mu$, the distance is

$$
d(z,z_i)^2=(z-z_i)^\top\Sigma^{-1}(z-z_i),\qquad
\Sigma^{-1}=\gamma^{-1}I+
\sum_{j=1}^{m}(\lambda_j^{-1}-\gamma^{-1})v_jv_j^\top.
$$

The code whitens vectors with a map $W$, so that
$d(z,z_i)=\|W(z)-W(z_i)\|_2$. Each rule below assigns a score
$s(z)$; smaller scores mean a prompt is closer to the safe reference under that
rule. A prompt is accepted when $s(z)\leq\varepsilon$ and rejected otherwise.
This is the **current `[EOS]`-based implementation**: whitening keeps all 768
coordinates, and the 32 PCA directions parameterize its covariance estimate.
The full `77×768` CLIP conditioning tensor is not used to define or score this
region.

1. **Nearest anchor:** $s_{\mathrm{near}}(z)=\min_i d(z,z_i)$. Its accepted
   region is exactly $\bigcup_{i=1}^{M}\{z:d(z,z_i)\leq\varepsilon_{\mathrm{near}}\}$,
   a union of equal-radius Mahalanobis balls around every safe-reference
   vector. This is the saved `safe_prompt_region.npz` geometry.
2. **Order-1 Wasserstein:**
   $s_1(z)=\frac{1}{M(M+1)}\sum_{i=1}^{M}d(z,z_i)$. It uses **all** reference
   anchors and measures distance to the reference cloud as a whole. Its
   accepted region is $\{z:s_1(z)\leq\varepsilon_1\}$.
3. **Adaptive order-infinity:** represent each prompt's task by its cloud of
   contextual CLIP content-token vectors. From all safe-reference prompts,
   select the five whose token clouds have the smallest
   order-1 Wasserstein distance to the candidate's cloud. For token clouds
   $C$ and $C_i$ with uniform token weights, this task distance is
   $W_1(C,C_i)=\min_{\pi}\sum_{t,u}\pi_{tu}\|h_t-h_{i,u}\|_2$, where $\pi$ ranges
   over couplings with those weights. If $A_5(z)$ is the selected set of
   whitened `[EOS]` anchors, score the largest edge of the minimum spanning
   tree on $A_5(z)\cup\{W(z)\}$:
   $s_\infty(z)=\max\operatorname{edge}\bigl(\operatorname{MST}
   (A_5(z)\cup\{W(z)\})\bigr)$. No edges are cut. The selected anchors can change
   with the candidate, so this rule has its own accepted set
   $\{z:s_\infty(z)\leq\varepsilon_\infty\}$. This is the **default research
   acceptance rule**; the other two rules are comparisons. For a fixed selected
   anchor subset, this MST test reduces to membership in its union of balls
   only when its threshold graph is connected at $\varepsilon_\infty$. Because
   the subset depends on the candidate, the overall accepted set need not be
   one fixed union of balls.

For **each** rule separately, score the $N$ safe-calibration prompts and set
$\varepsilon$ to the $\lceil(N+1)(1-0.05)\rceil$-th smallest calibration score.
This split-conformal threshold targets at least 95% marginal acceptance of a
new safe prompt when the calibration and future safe prompts are exchangeable
and the reference geometry is fixed. The held-out safe and risky splits are
used only to measure coverage and rejection; risky prompts do not set any
anchor, metric, selection rule, or threshold. The three thresholds have
different score scales and are not interchangeable.

For research-aligned decisions, use the adaptive order-infinity score and its
own calibrated threshold in the full-reference method report; the saved
`safe_prompt_region.npz` object's `contains()` method applies the nearest-anchor
comparison instead.

Open the analysis app with `.\.venv\Scripts\python.exe coco_region_app.py --port 8081`.
The root page compares nearest-anchor, order-1, and adaptive order-infinity
results and lets you switch the active decision method. The selected method
controls the coverage and rejection counts, score distributions, and I2P box
plots comparing prompt toxicity (or inappropriate-output percentage) for
accepted and rejected prompts, and the paginated prompt-decision table. These
I2P measures are not pooled with
DiffusionDB NSFW filters or T2I-RiskyPrompt categories. The previous COCO
analysis is available at `/coco` from the **COCO baseline** button. The app
reads saved scores and reports; opening it does not run CLIP.

DiffusionDB metadata is downloaded to `data/cache/diffusiondb/` on first use.
You can pass `--diffusiondb-metadata` and `--t2i-risky-json` to use local copies.
SafeSteer is gated and is not included in this 3,000-prompt sample; if used in
future, keep all of its matched pairs in evaluation only.

## Step 3: prompt representation comparison

With the same DiffusionDB safe reference, calibration, and held-out safe splits,
compare final-layer EOS, masked mean pooling of final-layer tokens, and EOS from
CLIP text layers 4 and 8. The PPCA nearest-anchor geometry, 5% conformal rule,
and corpus splits stay fixed. The selection report ranks held-out safe coverage,
the score shift under benign style/quality/clause-order changes, score versus
prompt-length correlation, and the region radius. Toxic prompts are encoded only
after the selected representation has been written to `selection.json`.

On Windows, run the existing WSL/Colab CLI setup and then:

```powershell
.\scripts\representations.ps1 --gpu T4
```

The local GPU/CPU alternative is:

```powershell
.\.venv\Scripts\python.exe -m backend.embeddings.compare_representations --device cuda
```

Results are saved under `data/prompts/representations/`; the Colab command also
downloads its resumable prompt-vector cache. The original EOS region and app
remain the baseline until you explicitly switch their artifact paths. The
Colab command uses the corpus already at `data/prompts/confgen_prompts.parquet`.

## Layout

- `backend/storage/`: SQLite schemas, dataset seeding, and local model copies.
- `backend/embeddings/`: CLIP extraction and COCO region jobs.
- `backend/validation/`: CSV prompt region analysis.
- `backend/gpu/`: Colab transport, workers, and guarded result merging.
- `frontend/coco_region_app.py`: validation analysis dashboard, with
  `coco_region_app.py` as its short root launcher.
- `data/coco/`, `data/validation/`, `data/models/`, and
  `data/research/`: dataset inputs, results, models, and research artifacts.
- `research/`: exploratory Python code and reports; its artifacts live under
  `data/research/`.
- `scripts/`: short WSL and PowerShell wrappers.

Default paths are defined in `backend/paths.py`. Data commands also accept
path overrides.

## COCO embeddings on Colab

The Colab CLI runs in Linux or WSL. Set it up and authenticate in the same
shell that launches the job. From the repository root in WSL Ubuntu:

```bash
bash scripts/setup_colab.sh
colab --auth oauth2 sessions
bash -l ./scripts/coco.sh --gpu
```

The PowerShell wrapper is `.\scripts\coco.ps1 --gpu`. It targets Ubuntu;
set `CONF_GEN_WSL_DISTRO` if your WSL distribution has another name. The job reads
`data/coco/annotations/captions_val2017.json` and
`data/coco/images/val2017/`, then fills `data/coco/coco.sqlite3` in batches.
Completed batches are committed; rerunning finds only captions without
embeddings. For a separate ten-caption source, use
`--annotations data/runs/coco_smoke_10/annotations.json --database data/runs/coco_new_check/coco.sqlite3`.
`--limit 10` processes ten missing captions from the full annotations but still
stores metadata for every caption. `--batch-size 100` reduces upload size when needed.

## Checks

```powershell
ruff check backend frontend scripts tests --select F401,F841,F821
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The full COCO database and model weights are local artifacts under `data/`.
See [the migration record](docs/restructuring-plan.md) for the path changes
made after the 25,014-caption run completed.

## COCO conformal region

After all COCO text contexts are stored, run:

```powershell
.\.venv\Scripts\python.exe -m backend.embeddings.define_region
```

This selects each caption's CLIP `[EOS]` vector, splits COCO by image, fits a
probabilistic-PCA Mahalanobis metric on the reference images' captions, and
uses the other 20% of images to calibrate the 95% nearest-anchor boundary.
Each calibration image contributes its highest caption score. The accepted
region is a union of equal-radius
balls in whitened embedding space. Its anchors, whitening parameters, radius,
split metadata, and source database signature are saved in
`data/coco/coco_region.npz`. Re-running the command loads that file when the
database and settings match; `--force` recomputes it. Other code can call
`define_coco_region()` and then `region.score(vectors)` or
`region.contains(vectors)` with 768-dimensional `[EOS]` vectors. The boundary
models the COCO caption distribution, not a harmfulness label. The score is
nearest-anchor distance directly; the research's MST score gives the same
union-of-balls boundary only when its anchor graph is connected at the radius.

Open the validation analysis dashboard:

```powershell
.\.venv\Scripts\python.exe coco_region_app.py --port 8081
```

The `/coco` page compares CSV `prompt_toxicity` scores for blocked and allowed
prompts, then shows toxicity density curves, category breakdowns, and a
paginated prompt table. It reads all 4,703 rows from
`data/validation/i2p_benchmark.csv` (or a CSV passed with `--prompts`). Saved
metrics load from `data/validation/runs/coco_region_blocking_summary.json` and
row decisions from `data/validation/runs/coco_region_blocking.csv`; only the
visible nine prompt rows are sent to the browser. Opening the page does not run
prompt encoding or load the 3D region visualization. If the saved results are
missing or stale, select **Analyze prompts** to calculate them; **Recompute
analysis** refreshes current results.

Recomputation checks each distinct prompt against the full COCO region and
calibrated order-1 W1 and adaptive order-infinity budgets. The transportation
cache can be prepared with
`python -m backend.embeddings.transport_budget`. CLIP EOS vectors and token
clouds are cached in `data/validation/validation.sqlite3`, so repeated runs
reuse stored encodings. Pass `--validation-database` to select another cache.
The projection cache can be prepared with
`python -m backend.embeddings.project_region`; use `--projections` if it has
a nondefault location.

To analyze all prompts in one batch, run
`python -m backend.validation.region_report`. It writes per-prompt decisions to
`data/validation/runs/coco_region_blocking.csv` and aggregate statistics to
`data/validation/runs/coco_region_blocking.json`. Repeated prompts share one
text-space decision, but retain separate CSV rows.
