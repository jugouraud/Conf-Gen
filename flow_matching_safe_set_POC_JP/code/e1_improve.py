"""E1, phase 2: improvements motivated by the phase-1 critique, on the same pool (paired with e1_eval.py).

(a) Localisation: the order-1 score with weights that depend on the candidate,
      kNN-k:  uniform weights on the k safe fields nearest to the candidate in the same distance,
      PK-c:   prompt-kernel weights w_i(e) ~ exp(-|e - e_i|^2 / (2 h^2)), h = c * median anchor-embedding distance.
    Theorem 6.1 holds pointwise for any weights, so the region still certifies outputs, relative to the
    k nearest safe fields; the budget is the calibrated threshold of the localised score.
(b) L^p map metrics, d^(p)(theta, theta') = (mean_l |T_theta(x_l) - T_theta'(x_l)|^p)^(1/p), p >= 1: still a
    pseudometric dominating the L^1 output distance (Theorem 6.1 applies), and a fraction m of noise sent at
    distance Delta costs at least m^(1/p) Delta (Proposition 6.3 with p in place of 2).
Usage: python e1_improve.py screen [n_splits]    -> ../results/e1/improve_screen.json
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time
import numpy as np
from e1_eval import D, CFG, Z, E, NS, NP, M, K, BETA, split, d_eucl, kth_threshold, RES

P_LIST = [1, 2, 4, 8]
R_TAIL = [0.25, 0.5, 1.0]


def d_lp(Ta, Tb, p):
    """L^p map metric on the probe measure, (na, L, 2) x (nb, L, 2) -> (na, nb)."""
    out = np.empty((len(Ta), len(Tb)))
    for i0 in range(0, len(Ta), 64):
        nrm = np.sqrt(((Ta[i0:i0 + 64, None].astype(float) - Tb[None]) ** 2).sum(-1))
        out[i0:i0 + 64] = (nrm ** p).mean(-1) ** (1.0 / p)
    return out


def d_tail(Ta, Tb, r):
    """Tail map score on the probe measure: P_l(|T_a(x_l) - T_b(x_l)| >= r), (na, L, 2) x (nb, L, 2) -> (na, nb)."""
    out = np.empty((len(Ta), len(Tb)))
    for i0 in range(0, len(Ta), 64):
        nrm = np.sqrt(((Ta[i0:i0 + 64, None].astype(float) - Tb[None]) ** 2).sum(-1))
        out[i0:i0 + 64] = (nrm >= r).mean(-1)
    return out


def d_trunc(Ta, Tb, r):
    """Truncated L^1 map metric on the probe measure: mean_l min(|T_a - T_b|, r) / r, a pseudometric with values in [0, 1]
    that dominates the tail P(|T_a - T_b| >= r)."""
    out = np.empty((len(Ta), len(Tb)))
    for i0 in range(0, len(Ta), 64):
        nrm = np.sqrt(((Ta[i0:i0 + 64, None].astype(float) - Tb[None]) ** 2).sum(-1))
        out[i0:i0 + 64] = np.minimum(nrm, r).mean(-1) / r
    return out


def score(Dm, kind, k=None, W=None):
    """Order-1 score from a distance matrix to the anchors (n, M): uniform, kNN-k, or given weights W (n, M).
    Normalised so that the certificate budget of Theorem 6.1 equals the threshold (no (M+1) factor)."""
    if kind == "U":
        return Dm.mean(1)
    if kind == "kNN":
        return np.sort(Dm, axis=1)[:, :k].mean(1)
    return (W * Dm).sum(1)


def prompt_weights(Eq, Eanc, c):
    d2 = d_eucl(Eq, Eanc) ** 2
    h = c * np.median(d_eucl(Eanc, Eanc)[np.triu_indices(len(Eanc), 1)])
    L = -d2 / (2 * h * h); L -= L.max(1, keepdims=True); W = np.exp(L)
    return W / W.sum(1, keepdims=True)


VARIANTS = [("U", None)] + [("kNN", k) for k in (1, 3, 5)] + [("PK", c) for c in (0.1, 0.25)]


def screen(n_splits):
    rows = []
    for cfg in ["rect1", "rect2"]:
        T = D[f"{cfg}_Tp"].astype(float)
        for s in range(n_splits):
            anc, cal, te = split(1000 + s)
            Wpk = {c: prompt_weights(E, E[anc], c) for c in (0.1, 0.25)}
            mats = [(p, d_lp(T, T[anc], p)) for p in P_LIST] + [(f"tail{r}", d_tail(T, T[anc], r)) for r in R_TAIL] + [(f"trunc{r}", d_trunc(T, T[anc], r)) for r in R_TAIL]
            for p, Dm in mats:
                for kind, par in VARIANTS:
                    sc = score(Dm, kind, k=par, W=Wpk.get(par) if kind == "PK" else None)
                    eps = kth_threshold(sc[cal], BETA)
                    r = dict(cfg=cfg, split=s, p=p, kind=kind, par=par, eps=float(eps),
                             coverage=float(np.mean(sc[te] <= eps)))
                    for m in CFG["masses"]:
                        r[f"excl_{m}"] = float(np.mean(sc[NS:][Z[NS:, 2] == m] > eps))
                    rows.append(r)
        print(f"{cfg}: {n_splits} splits screened", flush=True)
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "screen":
        t0 = time.time()
        rows = screen(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
        json.dump(rows, open(os.path.join(RES, "improve_screen.json"), "w"))
        print(f"[{time.time()-t0:.0f}s] saved improve_screen.json")
        for cfg in ["rect1", "rect2"]:
            print(f"\n== {cfg}: coverage and exclusion (means over splits); budget = threshold of the score ==")
            for p in P_LIST + [f"tail{r}" for r in R_TAIL] + [f"trunc{r}" for r in R_TAIL]:
                for kind, par in VARIANTS:
                    R = [r for r in rows if r["cfg"] == cfg and r["p"] == p and r["kind"] == kind and r["par"] == par]
                    ex = " ".join(f"{np.mean([r[f'excl_{m}'] for r in R]):.2f}" for m in CFG["masses"])
                    print(f"p={p} {kind}{'' if par is None else par!s:5s} coverage {np.mean([r['coverage'] for r in R]):.3f}  "
                          f"excluded (m=.01 .03 .1 .3): {ex}   budget median {np.median([r['eps'] for r in R]):.3f}")
