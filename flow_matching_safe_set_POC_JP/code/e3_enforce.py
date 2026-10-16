"""E3, step 3: enforcement on evaluation split 0 for every region of e3_regions.py, by mixing fields at sampling time (back-off,
Proposition 6.4), and the judge's verdict on the deployed outputs.

A request whose score exceeds the threshold is mixed with the nearest feasible approved prompt c (anchor or calibration prompt with
score at most the threshold, in the region's own distance): the deployed field is (1 - kappa) v(.; request) + kappa v(.; c), with the
smallest kappa found by bisection whose score is within the threshold. The score of a mixed field is computed in the region's own
space: for E and MH, on the mixed pooled embedding (the prompt mixing of Corollary 6.3, exact for a model affine in the prompt, a
proxy for this one); for PB, on the mixed source slice (exact: the slice of a mixture is the mixture of the slices); for L2 and TR, by
sampling the mixed field on the probes (exact, 4 bisection steps). The deployed field is then sampled on the evaluation noise, which
no score uses, and judged: P(winter) of the judge (CLIP ViT-B/32), snow and green fractions, and similarity to the request's scene.
For the field regions, the plug-in certificate on the set B_L = {chi : ViT-L/14 zero-shot says not winter} (Theorem 7.1 for L2,
Proposition 7.5 for TR) is computed from the anchors' probe outputs and compared with the deployed field's actual P(B_L).
Each (region, request) is saved in ../results/e3/enforce/ as soon as it is done, so a rerun resumes; the results are
assembled into ../results/e3/enforce.json and ../results/e3/enforce_latents.npz.     Usage: python e3_enforce.py
"""
import os, json, time
import numpy as np
import torch
import e3_common as C
import e3_regions as R

N_BIS_FIELD, N_BIS_CHEAP = 4, 18
REGIONS = [("TR", "adaptive"), ("E", "static"), ("E", "adaptive"), ("MH", "static"), ("MH", "adaptive"), ("PB", "static"),
           ("PB", "adaptive"), ("L2", "static"), ("TR", "static")]           # a field region first: its code path is the longest
R_PRIME = np.linspace(0.02, 1.0, 50)


class Region:
    def __init__(self, name, tag, D, idx, best, anc, cal):
        self.name, self.tag = name, tag
        b = best[f"{name}|{tag}"]
        self.v = R.parse(b["variant"]); self.par = b["par"]
        self.sp = R.Space(name, D, idx, self.par)
        self.F = self.sp.feats(D); self.Ep = D["pooled"]
        self.anc = anc
        self.Daa = self.sp.dist(self.F[anc], self.F[anc]); self.Eaa = R.eucl(self.Ep[anc], self.Ep[anc])
        self.eps = C.kth_threshold(self.scores(self.F[cal], self.Ep[cal]), R.BETA)
        pool = np.r_[anc, cal]
        self.feas = pool[self.scores(self.F[pool], self.Ep[pool]) <= self.eps]

    def weights_and_dists(self, f, e):
        Dc = self.sp.dist(f, self.F[self.anc]); Ec = R.eucl(e, self.Ep[self.anc])
        return Dc, Ec

    def scores(self, f, e):
        Dc, Ec = self.weights_and_dists(f, e)
        return R.score(Dc, self.v, self.Daa, Ec, self.Eaa)

    def weights(self, f, e):
        """The transport weights of the score at one candidate (for the certificate)."""
        Dc, Ec = self.weights_and_dists(f, e)
        kind, par = self.v
        if kind == "U":
            return np.full(len(self.anc), 1 / len(self.anc))
        if kind == "N":
            w = np.zeros(len(self.anc)); w[np.argsort(Dc[0])[:par]] = 1 / par; return w
        iu = np.triu_indices(len(self.anc), 1)
        h = par * np.median((self.Daa if kind == "K" else self.Eaa)[iu])
        L = -((Dc if kind == "K" else Ec)[0] ** 2) / (2 * h * h); L -= L.max(); w = np.exp(L)
        return w / w.sum()


def main():
    t0 = time.time()
    rows, D, idx = R.load()
    reg_info = json.load(open(os.path.join(C.RES, "regions.json")))
    R.TR_RS = reg_info["tr_levels"]
    anc, cal, te = R.split(idx["approved_eval"], 1000)
    requests = np.r_[idx["summer_eval"][:8], idx["neutral_eval"][:8], idx["disguised_eval"][:8], np.sort(te)[:6]]
    pipe = C.load_pipe()
    chi, judge = C.Clip("openai/clip-vit-large-patch14"), C.Clip("openai/clip-vit-base-patch32")
    nL = chi.tw - chi.to; nL /= np.linalg.norm(nL)                              # B_L = {u : <u, nL> < 0}
    xp, xe = C.noises(C.PROBE_SEED, C.P_PROBE), C.noises(C.EVAL_SEED, C.P_EVAL)
    cond_cache = {}

    def cond(j):
        if j not in cond_cache:
            c, _, neg = C.encode(pipe, [rows[j]["prompt"]]); cond_cache[j] = c[0]; cond_cache["neg"] = neg
        return cond_cache[j]

    cond(int(requests[0]))
    neg = cond_cache["neg"]
    gen_cache = {}

    def mixed(j, c, kappa, x, tag):
        """Latents of the mixture (1 - kappa) v(.; j) + kappa v(.; c) on the noises x (cached)."""
        key = (j, c, round(kappa, 6), tag)
        if key not in gen_cache:
            if kappa <= 0:
                gen_cache[key] = torch.tensor(D[f"lat_{tag}"][j].astype(np.float32))
            elif kappa >= 1:
                gen_cache[key] = torch.tensor(D[f"lat_{tag}"][c].astype(np.float32))
            else:
                gen_cache[key] = C.sample(pipe, neg, [cond(j), cond(c)], [1 - kappa, kappa], x)
        return gen_cache[key]

    def judge_feats(lat, j):
        imgs = C.decode(pipe, lat)
        fl, fb = chi.image(imgs), judge.image(imgs)
        subj = judge.text([rows[j]["subject_text"]])[0]
        return dict(pwB=judge.p_winter(fb).tolist(), pwL=chi.p_winter(fl).tolist(), snow=C.snow_fraction(imgs).tolist(),
                    green=C.green_fraction(imgs).tolist(), subj=(fb @ subj).tolist())

    out, latents = [], {}
    rec_dir = os.path.join(C.RES, "enforce"); os.makedirs(rec_dir, exist_ok=True)
    for name, tag in REGIONS:
        reg = Region(name, tag, D, idx, reg_info["best"], anc, cal)
        print(f"[{time.time()-t0:.0f}s] region {name}|{tag} ({R.vname(reg.v)}, par={reg.par}): eps {reg.eps:.4f}, "
              f"{len(reg.feas)} feasible targets", flush=True)
        for j in [int(x) for x in requests]:
            rec_file = os.path.join(rec_dir, f"{name}_{tag}_{j:04d}.npz")
            if os.path.exists(rec_file):                                       # done in an earlier run: resume
                d = np.load(rec_file)
                out.append(json.loads(str(d["rec"]))); latents[f"{name}|{tag}|{j}"] = d["lat"]
                continue
            fj, ej = reg.F[j][None], reg.Ep[j][None]
            s0 = float(reg.scores(fj, ej)[0])
            rec = dict(region=f"{name}|{tag}", j=j, kind=rows[j]["kind"], prompt=rows[j]["prompt"], s0=float(s0 / reg.eps), eps=float(reg.eps))
            if s0 <= reg.eps:
                kappa, c, f_dep = 0.0, None, fj
            else:
                c = int(reg.feas[np.argmin(reg.sp.dist(fj, reg.F[reg.feas])[0])])
                fc, ec = reg.F[c][None], reg.Ep[c][None]
                lo, hi, f_dep = 0.0, 1.0, fc
                for _ in range(N_BIS_FIELD if name in ("L2", "TR") else N_BIS_CHEAP):
                    mid = (lo + hi) / 2
                    if name in ("L2", "TR"):
                        fm = chi.image(C.decode(pipe, mixed(j, c, mid, xp, "probe")))[None]
                    elif name == "PB":
                        fm = (1 - mid) * fj + mid * fc
                    else:
                        fm = (1 - mid) * fj + mid * fc
                    em = (1 - mid) * ej + mid * ec
                    if reg.scores(fm, em)[0] <= reg.eps:
                        hi, f_dep = mid, fm
                    else:
                        lo = mid
                kappa = hi
            rec.update(kappa=float(kappa), target=c, target_prompt=None if c is None else rows[c]["prompt"])
            lat = mixed(j, c if c is not None else j, kappa, xe, "eval")
            latents[f"{name}|{tag}|{j}"] = lat.numpy().astype(np.float16)
            rec.update(judge_feats(lat, j))
            if name in ("L2", "TR"):                                            # plug-in certificate on B_L, relative to the anchors
                e_dep = reg.Ep[j][None] if c is None else (1 - kappa) * reg.Ep[j][None] + kappa * reg.Ep[c][None]
                w = reg.weights(f_dep, e_dep)
                u = D["chi_probe"][anc]                                         # (M, P, 768)
                dist_B = (u @ nL)                                               # distance to B_L (positive outside B_L)
                q = np.array([(w[:, None] * (dist_B < rp)).sum(0).mean() for rp in R_PRIME])
                if name == "TR":
                    bound = q + reg.eps * np.maximum(1, reg.par / R_PRIME)
                else:
                    bound = q + reg.eps / R_PRIME
                rec.update(cert=float(bound.min()), cert_r=float(R_PRIME[bound.argmin()]),
                           actual_BL=float(np.mean(np.array(rec["pwL"]) < 0.5)))
            out.append(rec)
            tmp = rec_file[:-4] + ".tmp.npz"                                   # atomic write of this (region, request)
            np.savez(tmp, rec=np.array(json.dumps(rec)), lat=latents[f"{name}|{tag}|{j}"]); os.replace(tmp, rec_file)
            print(f"   {rows[j]['kind']:9s} {j:3d}: s0/eps {rec['s0']:.2f} kappa {kappa:.3f} | P(winter) judge "
                  f"{np.mean(rec['pwB']):.2f} unsafe {np.mean(np.array(rec['pwB']) < 0.5):.2f} subj {np.mean(rec['subj']):.3f}"
                  + (f" | cert {rec['cert']:.2f} actual {rec['actual_BL']:.2f}" if 'cert' in rec else ""), flush=True)
        json.dump(dict(requests=[int(x) for x in requests], anchors=[int(x) for x in anc], rows=out),
                  open(os.path.join(C.RES, "enforce.json"), "w"), indent=0)
        np.savez_compressed(os.path.join(C.RES, "enforce_latents.npz"), **latents)
    print(f"[{time.time()-t0:.0f}s] done")


if __name__ == "__main__":
    main()
