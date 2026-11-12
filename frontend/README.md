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
