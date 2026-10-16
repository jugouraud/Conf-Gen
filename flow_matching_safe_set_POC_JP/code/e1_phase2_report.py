"""Print every number of the E1 phase-2 comparison from ../results/e1/phase2.json (means over the splits)."""
import os, json
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e1")
P = json.load(open(os.path.join(RES, "phase2.json")))
MASSES = [0.01, 0.03, 0.1, 0.3]
ORDER = ["U-L2", "kNN1-L1", "kNN1-tr1", "kNN1-tl.5", "E-U", "E-kNN1"]


def table(cfg):
    print(f"\n== {cfg}: detection, enforcement and certificates (means over {len({b['split'] for b in P['enforce']})} splits) ==")
    print(f"{'method':10s} {'cov':>5s} {'budget':>7s} | {'excluded m=.01 .03 .1 .3':>25s} | {'P(A) after enforcement':>30s} | "
          f"{'MMD2 .1/.3':>11s} | {'safe chg':>8s} {'safe MMD2':>9s}")
    for me in ORDER:
        B = [b for b in P["enforce"] if b["cfg"] == cfg and b["method"] == me]
        rows = [r for b in B for r in b["rows"]]
        ex = " ".join(f"{np.mean([b[f'excl_{m}'] for b in B]):.2f}" for m in MASSES)
        pa = " ".join(f"{np.mean([r['pA'] for r in rows if r['m'] == m]):.4f}" for m in MASSES)
        mm = "/".join(f"{np.mean([r['mmd'] for r in rows if r['m'] == m]):.0f}" for m in [0.1, 0.3])
        saf = [r for r in rows if r["safe"]]
        print(f"{me:10s} {np.mean([b['coverage'] for b in B]):5.3f} {np.median([b['eps'] for b in B]):7.4f} | {ex:>25s} | {pa:>30s} | "
              f"{mm:>11s} | {np.mean([r['kappa'] > 0 for r in saf]):8.2f} {np.mean([r['mmd'] for r in saf]):9.2f}")
    none = [r for b in P["enforce"] if b["cfg"] == cfg and b["method"] == "U-L2" for r in b["rows"]]
    print("unconstrained P(A):  " + " ".join(f"m={m}: {np.mean([r['pA0'] for r in none if r['m'] == m]):.4f}" for m in MASSES)
          + f"  safe: {np.mean([r['pA0'] for r in none if r['safe']]):.4f}")
    print(f"{'method':10s} certificate on P(A) for the final fields (all lie in their region):")
    for me in ORDER[:4]:
        rows = [r for b in P["enforce"] if b["cfg"] == cfg and b["method"] == me for r in b["rows"]]
        bd, pa, fr = np.array([r["bound"] for r in rows]), np.array([r["pA"] for r in rows]), np.array([r["fresh"] / r["budget"] for r in rows])
        te = np.array([r["tail_emp"] for r in rows])
        unsafe = np.array([r["m"] >= 0.1 for r in rows])
        print(f"   {me:10s} bound median {np.median(bd):.3f} (m>=0.1: {np.median(bd[unsafe]):.3f}) | violations {np.sum(pa > bd)} of {len(pa)} | "
              f"fresh-noise value / budget: median {np.median(fr):.2f}, max {fr.max():.2f}, >1 in {np.mean(fr > 1):.3f} | "
              f"empirical-tail bound median {np.median(te):.3f} | worst P(A) admitted {pa.max():.4f}")


def region_bound(cfg, me, eps, anc):
    """Certificate valid for EVERY field of the region (B = A): uniform mixture for U, worst anchor for kNN1."""
    D = np.load(os.path.join(RES, "fit.npz")); R = np.linspace(0.05, 2.0, 40)
    Te = D[f"{cfg}_Te"][anc].astype(float); r_A = 0.6
    q = np.array([[np.mean(np.linalg.norm(Te[i], axis=1) < r_A + r) for r in R] for i in range(len(anc))])
    if me == "U-L2":
        return float(np.min(q.mean(0) + eps / R))
    per = {"kNN1-L1": q + eps / R, "kNN1-tr1": q + eps * np.maximum(1, 1.0 / R),
           "kNN1-tl.5": np.where(R >= 0.5, q + eps, np.inf)}[me]
    return float(per.min(1).max())


def attacks():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from e1_eval import split
    anc = split(1000)[0]
    print("\n== white-box attack: largest P(A) among admitted fields (12 starts x 300 steps, split 0, fresh noise) ==")
    for cfg in ["rect1", "rect2"]:
        for me in ORDER:
            A = [a for a in P["attack"] if a["cfg"] == cfg and a["method"] == me][0]
            pf = np.array([r["pA_fresh"] for r in A["runs"]]); sc = np.array([r["score"] for r in A["runs"]])
            rb = region_bound(cfg, me, A["eps"], anc) if not me.startswith("E-") else float("nan")
            print(f"{cfg:6s} {me:10s} max P(A) {pf.max():.4f}  median {np.median(pf):.4f}  (score/eps max {sc.max():.3f})  region-wide bound {rb:.3f}")


if __name__ == "__main__":
    for cfg in ["rect1", "rect2"]:
        table(cfg)
    attacks()
