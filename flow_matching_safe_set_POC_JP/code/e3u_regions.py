"""E3u, step 2: a SAFE set and an UNSAFE set in each space of e3_regions.py, with static and prompt-adaptive weights, on the Sana pool
(Section 12 of safe_prompt_fields.tex).

Two classes of history. Safe: the approved (winter) prompts of e3_prompts.py (70 tuning, 120 evaluation). Unsafe: the summer prompts,
the 16 of the tuning pool and the historical ones of e3u_prompts.py (54 more for tuning, 120 for evaluation); the 24 summer requests
of the evaluation pool are exchangeable with them and remain requests. Each class gives a set of fields, as in e3_regions.py:
  score of a candidate field v for a request c:  g(c, v) = sum_i w_i(c) d(v, v_i)  over the M anchors of the class,
  threshold eps = C_{1-beta} of the K calibration prompts of the class (each scored with its own weights),
  set of the request c:  {v : g(c, v) <= eps}.
Spaces d: E (prompt), PB (source slice), L2 (field, chi-space map metric); see e3_regions.Space. Weights: U (static, uniform), or
prompt-adaptive, computed from the request's prompt embedding ONLY, so that the two sets of a request are fixed sets of fields while
its field is repaired: X:c, Gaussian kernel in the Euclidean prompt distance with bandwidth c x the median prompt distance between
the class's anchors; XN:k, equal weights on the k anchors whose prompts are nearest. In the E space they coincide with the K:c and
N:k weights of e3_regions.py.
Tuning (the tuning pools only; 50 splits into 30 anchors and 40 calibration prompts of the class): the adaptive weighting of each set
is the one that makes the set smallest, i.e. that admits the fewest tuning prompts of the OTHER class (safe set: unsafe prompts;
unsafe set: approved prompts); ties broken by the median ratio of their score to the threshold (larger is better).
Decision rules for a request c: 'safe' admits iff v_c is in the safe set (Section 11); 'unsafe' admits iff v_c is outside the unsafe
set; 'both' admits iff v_c is in the safe set AND outside the unsafe set (the rule of Section 12); 'both, capped' is 'both' with the
unsafe threshold lowered to at most the K_CAP-th smallest unsafe score of the SAFE calibration prompts, which bounds the fraction of
approved prompts that the unsafe set claims (Proposition 12.3(iii)).
Evaluation: 200 splits; the safe split of the 120 approved evaluation prompts is that of e3_regions.py (seed 1000 + s: 30 anchors, 40
calibration, 50 test), the unsafe split of the 120 historical unsafe evaluation prompts uses seed 2000 + s (30 / 40 / 50). For every
space, weighting and kind of request: the fractions in the safe set, in the unsafe set, and in each of the four cells (safe only,
unsafe only, both = ambiguous, neither = novel). Anchor sweep: all 190 prompts of each class pooled, M anchors of each class for M in
M_GRID (K = 40), fixed weightings (none selected).
Writes ../results/e3/regions_unsafe.json.     Usage: python e3u_regions.py
"""
import os, json
import numpy as np
import e3_regions as R
from e3_common import RES, kth_threshold

M, K, BETA = R.M, R.K, R.BETA
N_TUNE, N_EVAL = 50, 200
SPACES = ["E", "PB", "L2"]
XC, XK = [0.03, 0.1, 0.3, 1.0], [1, 3]
K_CAP = 2                                   # 'both, capped': unsafe threshold <= 2nd smallest of 40 approved calibration prompts
M_GRID = [30, 50, 70, 100, 140]
SWEEP_W = [("U", None), ("XN", 1), ("XN", 3), ("X", 0.1)]
FEATS = ["pooled", "slice", "chi_probe", "pwB_probe", "subj_probe"]


def load_all():
    """Prompts and features of prompts.json followed by prompts_unsafe.json (ids 0..309, then 310..483). Evaluation-noise features
    exist for the first 310 prompts only (main pool D); F holds the probe features of all of them."""
    rows, D, idx = R.load()
    ru = json.load(open(os.path.join(RES, "prompts_unsafe.json")))["prompts"]
    Du = np.load(os.path.join(RES, "pool_unsafe.npz"))
    assert [r["id"] for r in ru] == list(range(len(rows), len(rows) + len(ru)))
    F = {k: np.concatenate([D[k], Du[k].astype(D[k].dtype)]) for k in FEATS}
    hist = lambda p: np.array([r["id"] for r in ru if r["pool"] == p])
    idx = dict(idx, unsafe_tune=np.r_[idx["summer_tune"], hist("tune")], unsafe_eval=hist("eval"))
    return rows + ru, D, F, idx


def dist_matrix(name, F):
    sp = R.Space(name, F, None)
    f = {"E": F["pooled"], "PB": F["slice"].astype(np.float32), "L2": F["chi_probe"]}[name]
    return sp.dist(f, f)


def wname(w):
    return "U" if w[0] == "U" else f"{w[0]}:{w[1]:g}"


def wparse(s):
    if s == "U":
        return ("U", None)
    k, p = s.split(":")
    return (k, int(p) if k == "XN" else float(p))


def prompt_weights(Ec, Eaa, w):
    """Weights (n, M) of n requests over M anchors from their prompt distances Ec (n, M); Eaa: prompt distances between anchors."""
    kind, par = w
    n, m = Ec.shape
    if kind == "U":
        return np.full((n, m), 1.0 / m)
    if kind == "XN":
        W = np.zeros((n, m)); np.put_along_axis(W, np.argsort(Ec, 1)[:, :par], 1.0 / par, 1)
        return W
    h = par * np.median(Eaa[np.triu_indices(m, 1)])
    L = -(Ec.astype(np.float64) ** 2) / (2 * h * h); L -= L.max(1, keepdims=True)
    W = np.exp(L)
    return W / W.sum(1, keepdims=True)


class OneSet:
    """The set of one class in one space: anchors, prompt-adaptive (or static) weights, and its conformal threshold."""
    def __init__(self, Dsp, Ep, w, anc, cal, beta=BETA):
        self.Dsp, self.Ep, self.w, self.anc = Dsp, Ep, w, np.asarray(anc)
        self.Eaa = Ep[np.ix_(self.anc, self.anc)]
        self.cal_scores = self.score_rows(cal)
        self.eps = float(kth_threshold(self.cal_scores, beta))

    def weights(self, rows):
        return prompt_weights(self.Ep[np.ix_(rows, self.anc)], self.Eaa, self.w)

    def score_rows(self, rows):
        """Scores of the prompts 'rows' under their own fields and their own weights."""
        rows = np.asarray(rows)
        return (self.weights(rows) * self.Dsp[np.ix_(rows, self.anc)]).sum(1)

    def score_with(self, d_anc, w):
        """Score of a candidate field from its distances d_anc (n, M) to the anchors, with the weights w (n, M) of the request."""
        return (w * d_anc).sum(1)


def tune(Dsp, Ep, idx):
    """For each class, the prompt-adaptive weighting that makes its set smallest on the tuning pools."""
    out = {}
    for cls, own, other, seed in [("safe", "approved_tune", "unsafe_tune", 5000), ("unsafe", "unsafe_tune", "approved_tune", 6000)]:
        cands = []
        for w in [("U", None)] + [("X", c) for c in XC] + [("XN", k) for k in XK]:
            inside, ratio = [], []
            for s in range(N_TUNE):
                anc, cal, _ = R.split(idx[own], seed + s)
                st = OneSet(Dsp, Ep, w, anc, cal)
                so = st.score_rows(idx[other])
                inside.append(np.mean(so <= st.eps)); ratio.append(np.median(so / st.eps))
            cands.append(dict(w=wname(w), size=float(np.mean(inside)), ratio=float(np.median(ratio))))
        best = min([c for c in cands if c["w"] != "U"], key=lambda c: (round(c["size"], 6), -c["ratio"]))
        out[cls] = dict(candidates=cands, adaptive=best["w"])
    return out


def cells(in_s, in_u):
    return dict(in_safe=float(np.mean(in_s)), in_unsafe=float(np.mean(in_u)), adm_safe=float(np.mean(in_s)),
                adm_unsafe=float(np.mean(~in_u)), adm_both=float(np.mean(in_s & ~in_u)),
                c_safe=float(np.mean(in_s & ~in_u)), c_unsafe=float(np.mean(~in_s & in_u)),
                c_ambig=float(np.mean(in_s & in_u)), c_novel=float(np.mean(~in_s & ~in_u)))


def main():
    rows, D, F, idx = load_all()
    E = dist_matrix("E", F)
    Dm = {n: (E if n == "E" else dist_matrix(n, F)) for n in SPACES}
    reg = json.load(open(os.path.join(RES, "regions.json")))
    label = {int(k): v for k, v in reg["labels"].items()}
    req = np.array(sorted(label))
    groups = {"summer": idx["summer_eval"], "neutral": idx["neutral_eval"], "disguised": idx["disguised_eval"],
              "paraphrase": idx["paraphrase_eval"], "novel_winter": idx["novel_winter_eval"], "novel_summer": idx["novel_summer_eval"],
              "judged_unsafe": np.array([j for j in req if label[j]]), "judged_safe": np.array([j for j in req if not label[j]])}
    req_all = np.r_[req, idx["paraphrase_eval"], idx["novel_winter_eval"], idx["novel_summer_eval"]]
    pw = F["pwB_probe"]
    res = {"tuning": {}, "eval": {}, "per_request": {}, "anchors": {}, "M_grid": M_GRID, "k_cap": K_CAP,
           "history_unsafe_probe": float(np.mean(pw[idx["unsafe_eval"]] < 0.5)),
           "history_unsafe_tune_probe": float(np.mean(pw[idx["unsafe_tune"]] < 0.5)),
           "approved_unsafe_probe": float(np.mean(pw[idx["approved_eval"]] < 0.5))}
    for name in SPACES:
        res["tuning"][name] = tune(Dm[name], E, idx)
        print(f"{name}: tuned safe {res['tuning'][name]['safe']['adaptive']}, unsafe {res['tuning'][name]['unsafe']['adaptive']} | " +
              " ".join(f"{c['w']}:{c['size']:.2f}/{u['size']:.2f}" for c, u in zip(res['tuning'][name]['safe']['candidates'],
                                                                                  res['tuning'][name]['unsafe']['candidates'])), flush=True)
        for tag in ["static", "adaptive"]:
            ws = ("U", None) if tag == "static" else wparse(res["tuning"][name]["safe"]["adaptive"])
            wu = ("U", None) if tag == "static" else wparse(res["tuning"][name]["unsafe"]["adaptive"])
            acc, ins, inu, inu_cap = {}, np.zeros(len(req_all)), np.zeros(len(req_all)), np.zeros(len(req_all))
            caps, eps = [], []
            for s in range(N_EVAL):
                a_s, c_s, t_s = R.split(idx["approved_eval"], 1000 + s)
                a_u, c_u, t_u = R.split(idx["unsafe_eval"], 2000 + s)
                S, U = OneSet(Dm[name], E, ws, a_s, c_s), OneSet(Dm[name], E, wu, a_u, c_u)
                eps_cap = min(U.eps, float(np.sort(U.score_rows(c_s))[K_CAP - 1]))
                caps.append(eps_cap < U.eps); eps.append((S.eps, U.eps, eps_cap))
                for g, rr in dict(groups, test_safe=t_s, test_unsafe=t_u).items():
                    in_s, su = S.score_rows(rr) <= S.eps, U.score_rows(rr)
                    c = cells(in_s, su <= U.eps)
                    c["adm_both_cap"] = float(np.mean(in_s & (su > eps_cap)))
                    c["in_unsafe_cap"] = float(np.mean(su <= eps_cap))
                    for k, v in c.items():
                        acc.setdefault(g, {}).setdefault(k, []).append(v)
                ins += S.score_rows(req_all) <= S.eps
                su = U.score_rows(req_all); inu += su <= U.eps; inu_cap += su <= eps_cap
            key = f"{name}|{tag}"
            res["eval"][key] = {g: {k: float(np.mean(v)) for k, v in d.items()} for g, d in acc.items()}
            res["eval"][key]["weights"] = dict(safe=wname(ws), unsafe=wname(wu))
            res["eval"][key]["cap_binds"] = float(np.mean(caps))
            res["eval"][key]["eps_median"] = dict(zip(["safe", "unsafe", "unsafe_cap"], np.median(np.array(eps), 0).tolist()))
            res["per_request"][key] = {int(j): dict(in_safe=float(a / N_EVAL), in_unsafe=float(b / N_EVAL), in_unsafe_cap=float(c / N_EVAL))
                                       for j, a, b, c in zip(req_all, ins, inu, inu_cap)}
            e = res["eval"][key]
            print(f"  {key:12s} ({wname(ws)}/{wname(wu)}): cov safe {e['test_safe']['in_safe']:.3f} unsafe-set cov "
                  f"{e['test_unsafe']['in_unsafe']:.3f} | approved: admitted (both) {e['test_safe']['adm_both']:.3f} in unsafe set "
                  f"{e['test_safe']['in_unsafe']:.3f} | cap binds {e['cap_binds']:.2f}", flush=True)
            for g in ["summer", "neutral", "disguised", "novel_summer", "paraphrase", "novel_winter"]:
                d = e[g]
                print(f"      {g:13s} excluded: safe {1 - d['adm_safe']:.2f} unsafe {1 - d['adm_unsafe']:.2f} both {1 - d['adm_both']:.2f} "
                      f"(cap {1 - d['adm_both_cap']:.2f}) | cells safe/unsafe/ambig/novel {d['c_safe']:.2f}/{d['c_unsafe']:.2f}/"
                      f"{d['c_ambig']:.2f}/{d['c_novel']:.2f}", flush=True)
    # anchor sweep: both classes pooled (tuning and evaluation prompts), fixed weightings, M anchors of each class
    app = np.r_[idx["approved_tune"], idx["approved_eval"]]; uns = np.r_[idx["unsafe_tune"], idx["unsafe_eval"]]
    for name in SPACES:
        for w in SWEEP_W:
            for Mx in M_GRID:
                acc = {}
                for s in range(N_EVAL):
                    a_s, c_s, t_s = R.split(app, 7000 + s, m=Mx, k=K)
                    a_u, c_u, t_u = R.split(uns, 8000 + s, m=Mx, k=K)
                    S, U = OneSet(Dm[name], E, w, a_s, c_s), OneSet(Dm[name], E, w, a_u, c_u)
                    for g, rr in dict(groups, test_safe=t_s, test_unsafe=t_u).items():
                        c = cells(S.score_rows(rr) <= S.eps, U.score_rows(rr) <= U.eps)
                        for k, v in c.items():
                            acc.setdefault(g, {}).setdefault(k, []).append(v)
                res["anchors"][f"{name}|{wname(w)}|{Mx}"] = {g: {k: float(np.mean(v)) for k, v in d.items()} for g, d in acc.items()}
            print(f"  sweep {name:3s} {wname(w):6s} " + " | ".join(
                f"M={Mx}: cov {res['anchors'][f'{name}|{wname(w)}|{Mx}']['test_safe']['adm_both']:.2f} sum "
                f"{1 - res['anchors'][f'{name}|{wname(w)}|{Mx}']['summer']['adm_both']:.2f} neu "
                f"{1 - res['anchors'][f'{name}|{wname(w)}|{Mx}']['neutral']['adm_both']:.2f} novW "
                f"{1 - res['anchors'][f'{name}|{wname(w)}|{Mx}']['novel_winter']['adm_both']:.2f}" for Mx in M_GRID), flush=True)
    json.dump(res, open(os.path.join(RES, "regions_unsafe.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
