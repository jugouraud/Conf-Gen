"""Figure data for one held-out task (pilot C): few-shot fits, generated samples, velocity fields on a grid
at several times, and ODE trajectories.  Writes results/example_fits.npz."""
import numpy as np, sys
import part_c_ft as P
from lfm_common import simulate, sample_task, tgrid

j = int(sys.argv[1]) if len(sys.argv) > 1 else 0
a = P.te[j]
methods = ["U_lam0.001", "C_ref", "C_on", "S_gam10", "centre"]
out = dict(j=j, task=a, true_means=P.L_means[a], eps_ref=P.eps_ref, eps_on=P.eps_on)
rng = np.random.default_rng(0)
out["true_samples"] = sample_task(P.L_means[a], 2500, rng)
X0 = np.random.default_rng(3).standard_normal((2500, 2))
gx = np.linspace(-3, 3, 21); GX, GY = np.meshgrid(gx, gx); grid = np.c_[GX.ravel(), GY.ravel()]
times = np.array([0.2, 0.5, 0.8, 0.95]); out["grid"] = grid; out["times"] = times
X0t = np.random.default_rng(4).standard_normal((60, 2))
for n_bad, data, A_n, B_n in P.few_shot_data(j):
    W, info = P.fit_methods(A_n, B_n)
    W["oracle"] = P.W_all[a]
    out[f"data_{n_bad}"] = data
    for k in methods + ["oracle"]:
        out[f"W_{k}_{n_bad}"] = W[k]
        out[f"samples_{k}_{n_bad}"] = simulate(P.feat, W[k], X0)
        out[f"pA_{k}_{n_bad}"] = P.p_unsafe(W[k])
        out[f"sref_{k}_{n_bad}"] = P.s_ref(W[k]) / P.eps_ref
        out[f"son_{k}_{n_bad}"] = P.s_on(W[k]) / P.eps_on
        out[f"field_{k}_{n_bad}"] = np.stack([P.feat(grid, t) @ W[k].T for t in times])
        Y = X0t.copy(); traj = [Y.copy()]
        for kk in range(len(tgrid) - 1):
            Y = Y + (tgrid[kk + 1] - tgrid[kk]) * (P.feat(Y, tgrid[kk]) @ W[k].T); traj.append(Y.copy())
        out[f"traj_{k}_{n_bad}"] = np.stack(traj)
    out[f"backoff_{n_bad}"] = info["C_on"]["backoff"]
    print(f"n_bad={n_bad}: " + "  ".join(f"{k}: pA {out[f'pA_{k}_{n_bad}']:.3f} sref {out[f'sref_{k}_{n_bad}']:.2f} son {out[f'son_{k}_{n_bad}']:.2f}"
                                       for k in methods + ["oracle"]), flush=True)
np.savez(f"{P.OUT}/example_fits.npz", **out)
print("saved example_fits.npz")
