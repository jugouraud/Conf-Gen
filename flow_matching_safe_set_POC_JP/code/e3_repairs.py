"""E3, step 2b: why the regions of e3_regions.py detect what they detect, and whether more anchors repair the weak ones.

(a) Mechanism. In each space, the median distance between prompts that differ in season only (summer against approved, same
    scene), in winter modifier and light only (approved against approved, same scene) and in scene (approved against approved,
    other scene). A region can only separate summer from approved prompts if the season moves a prompt further than the approved
    variability does; the ratio (season gap) / (scene gap) measures it, with gaps taken from the same-scene approved distance.
(b) Anchor count. All 190 approved prompts (tuning and evaluation pools) are split into M anchors, K = 40 calibration prompts and
    the rest as test prompts, for M in M_GRID, over N_SPLITS random splits; coverage and exclusion of the evaluation requests for
    four weightings fixed in advance (uniform, 1 and 3 nearest anchors, kernel 0.3), and false alarms on the safe but unfamiliar
    requests (paraphrases, winter scenes absent from the approved prompts) when e3_prompts.py --extend has added them. Every
    weighting is reported, none is selected, so pooling the tuning prompts selects nothing. Also: how often a scene has no anchor.
(c) Centrality. In the field distance, the mean distance from each approved prompt of the evaluation pool to the others, and that
    of the pine-forest requests: the uniform score adds the season to a baseline that depends on the scene.
Writes ../results/e3/repairs.json.     Usage: python e3_repairs.py
"""
import os, json
import numpy as np
import e3_regions as R

M_GRID = [30, 60, 100, 140]
N_SPLITS = 200
VARIANTS = [("U", None), ("N", 1), ("N", 3), ("K", 0.3)]


def main():
    rows, D, idx = R.load()
    reg = json.load(open(os.path.join(R.RES, "regions.json")))
    scene = np.array([r["scene"] for r in rows])
    app = np.r_[idx["approved_tune"], idx["approved_eval"]]
    summ = np.r_[idx["summer_tune"], idx["summer_eval"]]
    E_full = R.eucl(D["pooled"], D["pooled"])
    out = {"gaps": {}, "anchors": {}, "M_grid": M_GRID, "n_splits": N_SPLITS, "missing_scene": {}}

    for name in ["E", "MH", "PB", "L2", "TR"]:
        par = reg["best"][f"{name}|static"]["par"]
        sp = R.Space(name, D, idx, par); F = sp.feats(D); Df = sp.dist(F, F)

        def med(A, B, same_scene):
            m = ((scene[A][:, None] == scene[B][None]) == same_scene) & (A[:, None] != B[None])
            return float(np.median(Df[np.ix_(A, B)][m]))
        g = dict(app_same=med(app, app, True), app_other=med(app, app, False), summer_same=med(summ, app, True),
                 neutral_same=med(idx["neutral_eval"], app, True))
        g["ratio"] = (g["summer_same"] - g["app_same"]) / (g["app_other"] - g["app_same"])
        g["ratio_neutral"] = (g["neutral_same"] - g["app_same"]) / (g["app_other"] - g["app_same"])
        out["gaps"][name] = g
        print(f"{name:3s} season gap / scene gap {g['ratio']:.2f} (neutral {g['ratio_neutral']:.2f})", flush=True)

    # (c) centrality in the field distance: the uniform score adds the season to a baseline that depends on the scene
    sp = R.Space("L2", D, idx); F = sp.feats(D); Df = sp.dist(F, F)
    ae = idx["approved_eval"]
    m_app = np.array([Df[i, ae[ae != i]].mean() for i in ae])
    pine = lambda kind: next(int(j) for j in idx[f"{kind}_eval"] if scene[j] == 0)
    out["centrality"] = dict(median=float(np.median(m_app)), q90=float(np.quantile(m_app, 0.9)),
                             pine_approved=float(m_app[scene[ae] == 0].mean()),
                             pine_neutral=float(Df[pine("neutral"), ae].mean()), pine_summer=float(Df[pine("summer"), ae].mean()))
    print("centrality (field L2):", out["centrality"], flush=True)

    groups = {"summer": idx["summer_eval"], "neutral": idx["neutral_eval"], "disguised": idx["disguised_eval"]}
    groups.update({k: idx[f"{k}_eval"] for k in R.UNFAMILIAR if len(idx.get(f"{k}_eval", []))})   # safe but unfamiliar requests
    for M in M_GRID:
        miss = [np.mean([not np.any(scene[R.split(app, 7000 + s, m=M, k=40)[0]] == sc) for sc in np.unique(scene[app])])
                for s in range(N_SPLITS)]
        out["missing_scene"][M] = float(np.mean(miss))
    for name in ["E", "PB", "L2", "TR"]:
        par = reg["best"][f"{name}|static"]["par"]
        sp = R.Space(name, D, idx, par); F = sp.feats(D); Df = sp.dist(F, F)
        for v in VARIANTS:
            for M in M_GRID:
                res = []
                for s in range(N_SPLITS):
                    a, c, t = R.split(app, 7000 + s, m=M, k=40)
                    res.append(R.evaluate(Df, E_full, v, a, c, dict(groups, test=t)))
                out["anchors"][f"{name}|{R.vname(v)}|{M}"] = {k: float(np.mean([r[k] for r in res]))
                                                              for k in ["test", *groups]}
            print(f"{name:3s} {R.vname(v):6s} " + " | ".join(
                f"M={M}: cov {out['anchors'][f'{name}|{R.vname(v)}|{M}']['test']:.2f} summer "
                f"{out['anchors'][f'{name}|{R.vname(v)}|{M}']['summer']:.2f} neutral {out['anchors'][f'{name}|{R.vname(v)}|{M}']['neutral']:.2f}"
                for M in M_GRID), flush=True)
    json.dump(out, open(os.path.join(R.RES, "repairs.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
