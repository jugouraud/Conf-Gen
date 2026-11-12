# Local NiceGUI trajectory explorer

This frontend visualizes the **existing frozen trajectory-flow model**, not a
new or randomly initialized model. It reads only the ready scene observations
from an expert dataset's `frozen.json`, not expert solution labels. The current
"circuit" is a rectangular driving area (metres), not a closed-loop race track.
Moving circular obstacle positions are interpolated at physical times
`start_time + i*dt` for `i = 0..H`.

From the repository root, after training has produced `.cache/trained-flow`, run:

```bash
uv sync --locked
uv run --with-requirements frontend/requirements.txt python -m frontend.app \
  --checkpoint .cache/trained-flow \
  --sha256 YOUR_TRUSTED_WEIGHTS_SHA256 \
  --dataset .cache/expert-many --split train
```

Find the trusted SHA-256 in your training command's receipt (or a separately
verified training log). The UI **requires** it; do not copy the unverified
`metadata.json` value as the trusted source. Open <http://127.0.0.1:8080>.
The NiceGUI dependency is isolated from the existing scientific `uv.lock`;
`uv run --with-requirements` installs it into an ephemeral overlay.
The app binds only to `127.0.0.1` and has no login or external serving mode.

Choose a ready scene, geometric guidance, seed, and Euler/Heun integration steps.
Click **Generate references** to sample the **actual** learned drift. The same
seed gives each guidance the same initial noise, making side-by-side comparisons
meaningful. Generate at most 12 overlaid references (1–3 seeds per guidance).
The time slider moves the obstacle and vehicle footprint markers along physical
prediction indices; dashed circles include the vehicle footprint. A scene can
be selected from the chosen expert split without loading `y_ref_star`.

The panel shows exact swept obstacle-disc surface clearance (including obstacle
turns between prediction knots) and boundary-disc clearance for each raw
reference. Negative clearance indicates overlap, not a conformal decision.
Sampling does **not** run MPC, detection, steering/repair, or a closed-loop
vehicle. A positive geometric clearance is not a deployment safety certificate.
Inspecting reserved splits for development or checkpoint selection would
invalidate an independent final evaluation; `train` is the default.

Check the patch with:

```bash
uv run pytest tests/unit/test_trajectory_visualizer.py -q
uv run ruff check frontend tests/unit/test_trajectory_visualizer.py
```

CPU is the default and the model is loaded only once at startup. The model
checkpoint's source fingerprints remain intact because this patch does not edit
`backend/src/trajectory_flow/{model,data,train,sample}.py`.

## Fixed expert and predicted trajectory browser

After generating a **new** fixed-obstacle dataset with
`backend.experiments.expert_trajectories.pipeline`, launch:

```bash
uv run --with-requirements frontend/requirements.txt python -m frontend.expert_browser \
  --dataset .cache/expert-fixed --split train
```

For **trained vector-field predictions**, use the very same browser with the
verified checkpoint from the training receipt:

```bash
uv run --with-requirements frontend/requirements.txt python -m frontend.expert_browser \
  --dataset .cache/expert-fixed --split train \
  --checkpoint .cache/trained-flow --sha256 YOUR_TRUSTED_WEIGHTS_SHA256
```

`--checkpoint` and `--sha256` must be provided together. The browser verifies
checkpoint weights/source fingerprints with the existing loader and refuses a
checkpoint with a different dataset ID, horizon, timestep or guidance semantics.
Without a checkpoint, ground-truth browsing remains available and the predicted
checkbox is disabled. The frozen model is loaded once; predictions are sampled
on demand (Heun, 32 steps, noise seed 42) and cached per browser page, not
precomputed for all trajectories. It never substitutes labels for samples.

The ellipse stays at a **constant scale**; obstacle positions differ per scene.
Use the scrollable trajectory list to check each expert and, when available,
predicted path independently. The two paths for one index share the same scene,
goal and guidance; expert lines are solid, sampled predictions are dashed, and
both use the same color for that index. Selections persist, allowing up to 24
combined overlays. The most recently toggled scene's obstacle is always shown; each checked trajectory's own obstacle
and clearance ring use that trajectory's color (expert and prediction share
one obstacle). The background track never moves or rescales. The current row
shows its expert path length and, when sampled, positional RMS prediction error
in metres. The expert paths come from validated `expert-target-v1` files,
verified against the completed index. Predicted paths are raw samples from the
frozen learned vector field and are **not** validated safe paths or MPC rollouts.
"Shortest" means shortest among sampled, dynamically feasible Gaussian detours,
not a proof of global optimality over all continuous collision-free paths.
Long 7-25 m routes use H=96 and require a new dataset plus a newly trained
checkpoint; existing H=16 checkpoints are intentionally rejected. Changing the
split requires restarting the browser; keep `train` for development
so reserved evaluation labels remain untouched. The old `frontend.app` remains
available for its separate sampling/debugging controls, but is no longer needed
for side-by-side expert-versus-model trajectory browsing.
