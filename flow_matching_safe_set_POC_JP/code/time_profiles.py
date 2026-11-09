"""Figure data (pilot A setting): per-time-step forward and reverse KL contributions
    forward_k = 1/2 h_k E_{P^b_k} ||s_c - s_b||^2,   reverse_k = 1/2 h_k E_{P^c_k} ||s_c - s_b||^2,
averaged over the training fields b, for a clean held-out field c and contaminated versions of it.
Also samples of the exact-score chains for the task-family figure.  Writes results/time_profiles.npz."""
import os, numpy as np
from diffusion_common import *
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
cache = np.load(os.path.join(RES, "part_a_cache.npz"))
means_all = ring(cache["phi"], cache["rho"])
P = means_all.shape[0]
perm = np.random.default_rng(7).permutation(P); tr, te = perm[:20], perm[60:]
c = te[0]
rng = np.random.default_rng(1)
Z0p, Zp = rng.standard_normal((256, 2)), rng.standard_normal((K, 256, 2))     # same probes as part_a.py

hist = fields_legit(means_all[tr])
_, traj_h = simulate(*hist, Z0p, Zp, store=True)                            # (K, M, n, 2)
cands = {"clean": fields_legit(means_all[[c]]), **{f"m={m}": fields_bad(means_all[[c]], m) for m in [0.03, 0.1]}}
out = dict(ts=ts, hs=hs, task=c, train=tr)
for name, fld in cands.items():
    _, traj_c = simulate(*fld, Z0p, Zp, store=True)                          # (K, 1, n, 2)
    fwd, rev = np.zeros(K), np.zeros(K)
    for k in range(K):
        # forward: probes of every historical chain b; candidate score vs s_b
        Xh = traj_h[k]                                                       # (M, n, 2)
        sc = score_all(Xh.reshape(-1, 2), k, *fld).reshape(Xh.shape)         # candidate at historical probes
        sb = score_diag(Xh, k, *hist)
        fwd[k] = 0.5 * hs[k] * ((sc - sb) ** 2).sum(-1).mean()
        # reverse: probes of the candidate's own chain; every historical score
        Xc = traj_c[k, 0]
        sc2 = score_diag(traj_c[k], k, *fld)[0]
        sb2 = score_all(Xc, k, *hist)                                        # (M, n, 2)
        rev[k] = 0.5 * hs[k] * ((sb2 - sc2[None]) ** 2).sum(-1).mean()
    out[f"fwd_{name}"], out[f"rev_{name}"] = fwd, rev
    print(f"{name}: average forward KL {fwd.sum():.3f}, average reverse KL {rev.sum():.3f}")
# samples for the task-family figure
Z0s, Zs = np.random.default_rng(5).standard_normal((3000, 2)), np.random.default_rng(6).standard_normal((K, 3000, 2))
show = te[:4]
out["show_tasks"], out["show_means"] = show, means_all[show]
fin, _ = simulate(*fields_legit(means_all[show]), Z0s, Zs)
out["chain_samples"] = fin
fin_b, _ = simulate(*fields_bad(means_all[[c]], 0.1), Z0s, Zs)
out["chain_samples_bad"] = fin_b[0]
np.savez(os.path.join(RES, "time_profiles.npz"), **out)
print("saved time_profiles.npz")
