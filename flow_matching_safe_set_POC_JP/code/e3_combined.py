"""E3, step 2c: decision rules that combine two distances of e3_regions.py (Section 11.8 of safe_prompt_fields.tex).

Joint region: the score g(c) = max_j (s_j(c) - m_j) / q_j of the static source-slice (PB) and field (L2) scores, where m_j and q_j are
the median and interquartile range of the anchors' leave-one-out scores in space j. They depend on the anchors only, so the
calibration prompts stay exchangeable with a new approved prompt, and g is calibrated jointly at level beta (Proposition 11.1(i)).
Bonferroni region: each score calibrated separately at level beta/2, a request excluded if either score exceeds its threshold
(Proposition 11.1(ii)). Both are evaluated on the 200 evaluation splits of e3_regions.py with the same requests, as are the static
regions PB, L2 and MH for reference. Also: the deep level of the repair, the ceil((K+1)/2)-th smallest calibration score, and the
fraction of approved test prompts whose score it bounds (Proposition 11.1(iii)); for the joint region, which of its two
components excludes each request; and, at the threshold and at the deep level, how often a scene has no admissible target (no
anchor or calibration prompt of that scene whose score is within the level), so that a repair must change the scene.
Writes ../results/e3/combined.json.     Usage: python e3_combined.py
"""
import os, json
import numpy as np
import e3_regions as R
from e3_common import RES, kth_threshold

BETA_DEEP = 0.5
SINGLE = ["PB", "L2", "MH"]


def standardiser(Daa):
    """Median and interquartile range of the anchors' leave-one-out scores (static weights): a function of the anchors only."""
    loo = Daa.sum(1) / (len(Daa) - 1)
    q1, m, q3 = np.quantile(loo, [0.25, 0.5, 0.75])
    return float(m), float(q3 - q1)


def main():
    rows, D, idx = R.load()
    reg = json.load(open(os.path.join(RES, "regions.json")))
    Dfull = {}
    for name in SINGLE:
        sp = R.Space(name, D, idx, reg["best"][f"{name}|static"]["par"]); F = sp.feats(D)
        Dfull[name] = sp.dist(F, F)
    label = {int(k): v for k, v in reg["labels"].items()}
    req = np.array(sorted(label))
    groups = {"summer": idx["summer_eval"], "neutral": idx["neutral_eval"], "disguised": idx["disguised_eval"],
              "judged_unsafe": np.array([j for j in req if label[j]]), "judged_safe": np.array([j for j in req if not label[j]])}
    groups.update({k: idx[f"{k}_eval"] for k in R.UNFAMILIAR})
    req_all = np.r_[req, *[idx[f"{k}_eval"] for k in R.UNFAMILIAR]]
    methods = SINGLE + ["joint", "bonferroni"]
    acc = {m: {g: [] for g in ["test", "deep_test", *groups]} for m in methods}
    excl = {m: np.zeros(len(req_all)) for m in ["joint", "bonferroni"]}
    trig = {g: [] for g in groups}                       # joint region: share of its exclusions due to the source-slice component
    scene = np.array([r["scene"] for r in rows])
    no_target = {f"{n}|{lv}": [] for n in ["PB", "L2", "MH", "joint"] for lv in ["thr", "deep"]}   # scenes without an admissible target
    scenes = np.unique(scene[idx["approved_eval"]])
    loss = {n: {int(sc): [] for sc in scenes} for n in ["PB", "L2", "joint"]}                  # per scene, at the deep level
    ratio = {"PB": [], "L2": []}                         # joint threshold in each space, relative to that space's own threshold
    for s in range(R.N_EVAL):
        anc, cal, te = R.split(idx["approved_eval"], 1000 + s)
        sc = {n: (lambda rows_, n=n: Dfull[n][np.ix_(rows_, anc)].mean(1)) for n in SINGLE}
        eps = {n: kth_threshold(sc[n](cal), R.BETA) for n in SINGLE}
        deep = {n: kth_threshold(sc[n](cal), BETA_DEEP) for n in SINGLE}
        std = {n: standardiser(Dfull[n][np.ix_(anc, anc)]) for n in ["PB", "L2"]}
        z = lambda rows_, n: (sc[n](rows_) - std[n][0]) / std[n][1]
        g = lambda rows_: np.maximum(z(rows_, "PB"), z(rows_, "L2"))
        eps_g, deep_g = kth_threshold(g(cal), R.BETA), kth_threshold(g(cal), BETA_DEEP)
        for n in ["PB", "L2"]:
            ratio[n].append((std[n][0] + std[n][1] * eps_g) / eps[n])
        eps_b = {n: kth_threshold(sc[n](cal), R.BETA / 2) for n in ["PB", "L2"]}
        out_b = lambda rows_: (sc["PB"](rows_) > eps_b["PB"]) | (sc["L2"](rows_) > eps_b["L2"])
        for n in SINGLE:
            acc[n]["test"].append(np.mean(sc[n](te) <= eps[n])); acc[n]["deep_test"].append(np.mean(sc[n](te) <= deep[n]))
        acc["joint"]["test"].append(np.mean(g(te) <= eps_g)); acc["joint"]["deep_test"].append(np.mean(g(te) <= deep_g))
        acc["bonferroni"]["test"].append(np.mean(~out_b(te))); acc["bonferroni"]["deep_test"].append(np.nan)
        for gname, rr in groups.items():
            for n in SINGLE:
                acc[n][gname].append(np.mean(sc[n](rr) > eps[n]))
            out = g(rr) > eps_g
            acc["joint"][gname].append(np.mean(out)); acc["bonferroni"][gname].append(np.mean(out_b(rr)))
            if out.any():
                trig[gname].append(np.mean(z(rr, "PB")[out] >= z(rr, "L2")[out]))
        excl["joint"] += g(req_all) > eps_g; excl["bonferroni"] += out_b(req_all)
        pool = np.r_[anc, cal]
        for n, sco, lv in [(n, sc[n](pool), (eps[n], deep[n])) for n in SINGLE] + [("joint", g(pool), (eps_g, deep_g))]:
            for name_lv, level in zip(["thr", "deep"], lv):
                adm = set(scene[pool[sco <= level]])
                no_target[f"{n}|{name_lv}"].append(np.mean([sc_ not in adm for sc_ in np.unique(scene[pool])]))
                if name_lv == "deep" and n in loss:
                    for sc_ in np.unique(scene[pool]):
                        loss[n][int(sc_)].append(sc_ not in adm)
    res = {"eval": {m: {k: float(np.nanmean(v)) for k, v in acc[m].items()} for m in methods},
           "trigger_PB": {k: float(np.mean(v)) if v else None for k, v in trig.items()},
           "joint_threshold_ratio": {n: float(np.median(v)) for n, v in ratio.items()},
           "per_request": {m: dict(zip([int(j) for j in req_all], (excl[m] / R.N_EVAL).tolist())) for m in excl},
           "beta_deep": BETA_DEEP, "nominal_deep": int(np.ceil((R.K + 1) * (1 - BETA_DEEP))) / (R.K + 1),
           "no_target": {k: float(np.mean(v)) for k, v in no_target.items()}}
    # which scenes lose their target at the deep level: against their centrality in the field distance (mean distance of their
    # approved prompts to the other approved prompts of the evaluation pool)
    from scipy.stats import spearmanr
    app = idx["approved_eval"]
    cent = {int(sc): float(np.mean([Dfull["L2"][i, app[app != i]].mean() for i in app if scene[i] == sc])) for sc in scenes}
    name = {int(r["scene"]): r["subject_text"][len("a photograph of "):] for r in rows if r["kind"] == "approved"}
    res["scene_loss"] = {name[sc]: dict(centrality=cent[sc], **{n: float(np.mean(loss[n][sc])) for n in loss}) for sc in cent}
    res["scene_loss_spearman"] = {n: float(spearmanr([cent[sc] for sc in cent], [np.mean(loss[n][sc]) for sc in cent])[0]) for n in loss}
    for m in methods:
        e = res["eval"][m]
        print(f"{m:10s} cov {e['test']:.3f} (deep {e['deep_test']:.3f}) | summer {e['summer']:.2f} neutral {e['neutral']:.2f} "
              f"novel_summer {e['novel_summer']:.2f} | paraphrase {e['paraphrase']:.2f} novel_winter {e['novel_winter']:.2f} | "
              f"disguised {e['disguised']:.2f} | judged unsafe {e['judged_unsafe']:.2f} safe {e['judged_safe']:.2f}", flush=True)
    print("joint: share of exclusions due to the source slice", {k: None if v is None else round(v, 2) for k, v in res["trigger_PB"].items()})
    print("joint threshold / own threshold", {k: round(v, 3) for k, v in res["joint_threshold_ratio"].items()})
    print("scenes of the anchors and calibration prompts without an admissible target", {k: round(v, 3) for k, v in res["no_target"].items()})
    for k, v in sorted(res["scene_loss"].items(), key=lambda kv: kv[1]["centrality"]):
        print(f"  {k:38s} centrality {v['centrality']:.3f} | no target at the deep level: slice {v['PB']:.2f} field {v['L2']:.2f} joint {v['joint']:.2f}")
    print("rank correlation with centrality", {k: round(v, 2) for k, v in res["scene_loss_spearman"].items()})
    json.dump(res, open(os.path.join(RES, "combined.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
