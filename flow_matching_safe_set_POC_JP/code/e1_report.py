"""E1, step 3: print every number of the E1 analysis from ../results/e1/*.json and fit.npz.
Usage: python e1_report.py [enforce_file]   (default eval_enforce.json)"""
import os, sys, json
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e1")
D = np.load(os.path.join(RES, "fit.npz"))
CFG = json.load(open(os.path.join(RES, "fit_config.json")))
NS = int(D["n_safe"]); Z = D["Z"]; MASSES = CFG["masses"]
NOMINAL = int(np.ceil(61 * 0.9)) / 61


def model_quality():
    print("== model quality (pool) ==")
    for cfg in ["rect1", "rect2", "rect2_1step"]:
        pA, zeta = D[f"{cfg}_pA"], D[f"{cfg}_zeta"]
        msg = f"{cfg:12s} safe P(A) {pA[:NS].mean():.4f}, straightness defect zeta median {np.median(zeta[:NS]):.3f} |"
        for m in MASSES:
            s = Z[:, 2] == m
            msg += f" m={m}: P(A) {pA[s].mean():.4f} (target {D['true_pA'][s].mean():.4f})"
        print(msg)


def splits():
    rows = json.load(open(os.path.join(RES, "eval_splits.json")))
    print(f"\n== coverage and exclusion over {max(r['split'] for r in rows) + 1} splits (nominal {NOMINAL:.4f}) ==")
    for cfg in ["rect1", "rect2", "rect2_1step"]:
        for met in ["T", "0", "rho", "E"]:
            R = [r for r in rows if r["cfg"] == cfg and r["metric"] == met]
            cov = np.array([r["coverage"] for r in R])
            ex = "  ".join(f"m={m}: {np.mean([r[f'excl_{m}'] for r in R]):.2f}" for m in MASSES)
            print(f"{cfg:12s} d_{met:4s} coverage {cov.mean():.3f} (sd {cov.std():.3f}) | excluded {ex} | budget (M+1)eps median {np.median([31*r['eps'] for r in R]):.3f}")


def enforcement(fname):
    res = json.load(open(os.path.join(RES, fname)))
    for blk in res:
        cfg, rows = blk["cfg"], blk["rows"]
        meths = sorted({k.split(".")[0] for k in rows[0] if "." in k}, key=lambda x: (x != "none", x))
        print(f"\n== enforcement, {cfg} (eps: {', '.join(f'{k} {v:.4f}' for k, v in blk['eps'].items())}; feasible safe fields {blk['n_feasible']}) ==")
        print(f"{'method':14s} " + " ".join(f"{'P(A) m=' + str(m):>12s}" for m in MASSES) + f" {'MMD2 m=0.1':>11s} {'MMD2 m=0.3':>11s} {'safe changed':>12s} {'safe MMD2':>9s} {'in T-region':>11s}")
        for me in meths:
            line = f"{me:14s} "
            for m in MASSES:
                line += f"{np.mean([r[me + '.pA'] for r in rows if r['m'] == m]):12.4f} "
            for m in [0.1, 0.3]:
                line += f"{np.mean([r[me + '.mmd'] for r in rows if r['m'] == m]):11.2f} "
            saf = [r for r in rows if r["safe"]]
            changed = np.mean([abs(r[me + ".mmd"] - r["none.mmd"]) > 1e-9 for r in saf])
            line += f"{changed:12.2f} {np.mean([r[me + '.mmd'] for r in saf]):9.2f} "
            line += f"{np.mean([r[me + '.sT'] <= 1 + 1e-9 for r in rows]):11.2f}"
            print(line)
        mix = {k: max(r[k] for r in rows) for k in rows[0] if k.startswith("mixcheck")}
        print("prompt mixing check (Cor. 5.3), max |W(e#) - sum gamma W|:", {k: f"{v:.1e}" for k, v in mix.items()})
        # certificates on the d_T region: every final field with s_T <= eps_T
        ins = [(r, me) for r in rows for me in meths if r[me + ".sT"] <= 1 + 1e-9]
        ratio = np.array([r[me + ".coupling"] / r[me + ".budgetT"] for r, me in ins])
        pa = np.array([r[me + ".pA"] for r, me in ins])
        bm = np.array([r[me + ".bound_markov_T"] for r, me in ins]); bt = np.array([r[me + ".bound_tail"] for r, me in ins])
        print(f"Theorem 6.1 on fresh noise over {len(ins)} fields in the d_T region: coupling/budget median {np.median(ratio):.2f}, "
              f"max {ratio.max():.2f}, >1 in {np.mean(ratio > 1):.3f};  P(A) max {pa.max():.4f};  "
              f"Markov bound median {np.median(bm):.3f};  tail bound median {np.median(bt):.3f};  violations Markov {np.sum(pa > bm)}, tail {np.sum(pa > bt)}")


if __name__ == "__main__":
    model_quality()
    splits()
    enforcement(sys.argv[1] if len(sys.argv) > 1 else "eval_enforce.json")
