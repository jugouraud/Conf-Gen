import json, glob, os, numpy as np
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rows = [r for f in sorted(glob.glob(os.path.join(RES, "part_c_ft_*.json"))) for r in json.load(open(f))]
for n_bad in [0, 3]:
    R = [r["res"] for r in rows if r["n_bad"] == n_bad]
    meths = list(R[0].keys())
    mm = {k: np.array([x[k]["mmd"] for x in R]) * 1e3 for k in meths}
    pA = {k: np.array([x[k]["pA"] for x in R]) for k in meths}
    # oracle-tuned baselines (best hyper-parameter on mean MMD over the same tasks)
    bestU = min([k for k in meths if k.startswith("U_")], key=lambda k: mm[k].mean())
    bestS = min([k for k in meths if k.startswith("S_")], key=lambda k: mm[k].mean())
    print(f"\nn_bad={n_bad}  ({len(R)} tasks; 12 clean samples{' + 3 at the unsafe centre' if n_bad else ''})")
    print(f"{'method':<12}{'MMD x1e3 mean':>14}{'median':>9}{'P(A) mean':>11}{'P(A) max':>10}{'in ref region':>15}{'in on-pol.':>12}")
    for k in [*[m for m in meths if m.startswith('U_')], *[m for m in meths if m.startswith('S_')], "C_ref", "C_on", "centre", "oracle"]:
        sref = np.array([x[k]["s_ref"] for x in R]); son = np.array([x[k]["s_on"] for x in R])
        tag = " *" if k in (bestU, bestS) else ""
        print(f"{k+tag:<12}{mm[k].mean():>14.1f}{np.median(mm[k]):>9.1f}{pA[k].mean():>11.4f}{pA[k].max():>10.4f}"
              f"{(sref <= 1 + 1e-6).mean():>15.2f}{(son <= 1 + 1e-6).mean():>12.2f}")
    for k in ["C_ref", "C_on"]:
        print(f"  {k} beats tuned U ({bestU}) on {np.mean(mm[k] < mm[bestU]):.2f} of tasks, tuned S ({bestS}) on {np.mean(mm[k] < mm[bestS]):.2f};"
              f" geo-mean MMD ratio vs U {np.exp(np.mean(np.log(np.maximum(mm[k],1e-3)/np.maximum(mm[bestU],1e-3)))):.2f}")
    print("  C_on back-off used on", np.mean([x["C_on"]["backoff"] > 0 for x in R]), "of tasks; mean SOCP time C_ref",
          np.mean([x["C_ref"]["time"] for x in R]).round(2), "s")
