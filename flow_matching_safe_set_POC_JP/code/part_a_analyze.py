import numpy as np, sys, os, json
from scipy.stats import beta as betad

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
d = np.load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "part_a_cache.npz"))
G_LL, G_BL, G_LB = d["G_LL"], d["G_BL"], d["G_LB"]
pA_L, pA_B, B_base, B_mass = d["pA_L"], d["pA_B"], d["B_base"], d["B_mass"]
P = G_LL.shape[0]
n_big = 20000
masses = np.unique(B_mass)

def phi(c):  # concave envelope: TV <= min(Pinsker, Bretagnolle-Huber) with c = sqrt(2 KL)
    return np.minimum(c / 2, np.sqrt(1 - np.exp(-c ** 2 / 2)))

def kl_bin(p, q):
    p = np.clip(p, 1e-300, 1 - 1e-16); q = np.clip(q, 1e-300, 1 - 1e-16)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def kl_inv(q, kappa):
    lo, hi = q, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if kl_bin(mid, q) <= kappa: lo = mid
        else: hi = mid
    return lo

off = ~np.eye(P, dtype=bool)
print("legit pairwise path KL(P^a||P^b): median %.3f  90%% %.3f  max %.3f" %
      tuple(np.quantile(G_LL[off] / 2, [0.5, 0.9, 1.0])))
print("legit P(A) per field: mean %.2e  max %.2e" % (pA_L.mean(), pA_L.max()))
for m in masses:
    s = B_mass == m
    base = B_base[s]
    rev = G_BL[s][np.arange(s.sum()), base] / 2        # KL(P^bad || P^base)
    fwd = G_LB[base, np.where(s)[0]] / 2               # KL(P^base || P^bad)
    print(f"m={m:<6} P(A) bad: {pA_B[s].mean():.4f} | KL to own clean field: reverse {np.median(rev):.3f}  forward {np.median(fwd):.4f}  (-log(1-m)={-np.log(1-m):.4f})")

M, Kc, beta = 20, 40, 0.1
reps = 200
rng = np.random.default_rng(123)
q_idx = int(np.ceil((Kc + 1) * (1 - beta)))  # 1-indexed rank
res = {k: [] for k in ["F", "R", "K"]}
excl = {k: {m: [] for m in masses} for k in ["F", "R", "K"]}
bounds = {k: [] for k in ["F_tv", "R_tv", "K_kl", "R_kl_min"]}
viol = {k: 0 for k in bounds}; inside_max = {k: [] for k in bounds}
for r in range(reps):
    perm = rng.permutation(P)
    tr, ca, te = perm[:M], perm[M:M + Kc], perm[M + Kc:]
    norm = 1.0 / (M * (M + 1))
    S = {"F": lambda A: np.sqrt(G_LL[np.ix_(tr, A)]).sum(0) * norm,
         "R": lambda A: np.sqrt(G_LL[np.ix_(A, tr)]).sum(1) * norm,
         "K": lambda A: (G_LL[np.ix_(A, tr)] / 2).sum(1) * norm}
    badsel = np.isin(B_base, te); bidx = np.where(badsel)[0]
    Sb = {"F": np.sqrt(G_LB[np.ix_(tr, bidx)]).sum(0) * norm,
          "R": np.sqrt(G_BL[np.ix_(bidx, tr)]).sum(1) * norm,
          "K": (G_BL[np.ix_(bidx, tr)] / 2).sum(1) * norm}
    eps = {}
    for k in ["F", "R", "K"]:
        cal = np.sort(S[k](ca)); eps[k] = cal[q_idx - 1] if q_idx <= Kc else np.inf
        res[k].append((S[k](te) <= eps[k]).mean())
        for m in masses:
            sel = B_mass[bidx] == m
            excl[k][m].append((Sb[k][sel] > eps[k]).mean())
    # P_mix(A): Clopper-Pearson 95% upper bound on the pooled count of the training chains
    cnt = int(round(pA_L[tr].sum() * n_big)); N = M * n_big
    q_ucb = betad.ppf(0.95, cnt + 1, N - cnt)
    bd = {"F_tv": q_ucb + phi((M + 1) * eps["F"]),
          "R_tv": q_ucb + phi((M + 1) * eps["R"]),
          "K_kl": kl_inv(q_ucb, (M + 1) * eps["K"]),
          "R_kl_min": kl_inv(q_ucb, ((M + 1) * eps["R"]) ** 2 / 2 + np.log(M))}
    reg = {"F_tv": "F", "R_tv": "R", "K_kl": "K", "R_kl_min": "R"}
    for k, v in bd.items():
        bounds[k].append(v)
        R = reg[k]
        inside = np.r_[pA_L[te][S[R](te) <= eps[R]], pA_B[bidx][Sb[R] <= eps[R]]]
        mx = inside.max() if inside.size else 0.0
        inside_max[k].append(mx)
        viol[k] += int(mx > v)

print(f"\nM={M}, K_cal={Kc}, beta={beta}, {reps} random splits")
for k, name in [("F", "forward sqrt-KL (historical probes)"), ("R", "reverse sqrt-KL (own probes)"), ("K", "reverse KL (own probes)")]:
    print(f"[{name}] mean coverage of legit test fields: {np.mean(res[k]):.3f}  (5%-95%: {np.quantile(res[k],0.05):.3f}-{np.quantile(res[k],0.95):.3f})")
    print("   exclusion rate of contaminated fields by mass m: " +
          "  ".join(f"m={m}: {np.mean(excl[k][m]):.2f}" for m in masses))
print()
for k, name in [("F_tv", "TV bound, forward region"), ("R_tv", "TV bound, reverse region"),
                ("K_kl", "kl-inversion, reverse-KL region"), ("R_kl_min", "kl-inversion via min + log M, reverse region")]:
    print(f"[{name}] guaranteed P(A) bound: median {np.median(bounds[k]):.3f};  max P(A) of any field inside: median {np.median(inside_max[k]):.4f}, max {np.max(inside_max[k]):.4f};  violations {viol[k]}/{reps}")

# ---- save per-split arrays for the figures ----
json.dump(dict(M=M, Kc=Kc, beta=beta, reps=reps, masses=[float(m) for m in masses],
               coverage={k: [float(x) for x in v] for k, v in res.items()},
               exclusion={k: {str(m): [float(x) for x in v] for m, v in d_.items()} for k, d_ in excl.items()},
               bounds={k: [float(x) for x in v] for k, v in bounds.items()},
               inside_max={k: [float(x) for x in v] for k, v in inside_max.items()},
               violations=viol),
          open(os.path.join(RES, "part_a_splits.json"), "w"), indent=1)
print("saved part_a_splits.json")
