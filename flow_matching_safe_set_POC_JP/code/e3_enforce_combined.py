"""E3, step 3b: enforcement for the methods of Section 11.8 of safe_prompt_fields.tex, which decide with one distance and repair
with another, and for the reference regions of e3_enforce.py on the safe-but-unfamiliar requests.

A method is a decision score s with its threshold eps = C_{1-beta} (split 0 of e3_regions.py, static weights), a stop level and a
target rule:
  decision : PB, L2, MH or E (e3_regions.py), or J, the joint score of e3_combined.py (source slice and field, standardised by
             the anchors' leave-one-out scores, calibrated jointly);
  stop     : 'thr', the threshold eps itself (the minimal back-off of Proposition 6.4), or 'deep', eps' = the ceil((K+1)/2)-th
             smallest calibration score (a request is moved to the median approved score), or 'local', the threshold and also a field
             distance to the target at most the median nearest-anchor field score of the calibration prompts (the deployed field is
             as close to its target as half of the approved prompts are to their nearest anchor; added after the results of the
             other methods);
  target   : 'own', the approved prompt nearest in the decision distance (as in e3_enforce.py), or 'field', the one nearest in the
             field distance L2, among the anchors and calibration prompts whose own score is at most the stop level;
  repair   : (1 - kappa) v(.; request) + kappa v(.; target) inside the guidance, with the smallest kappa found by bisection whose
             score is at most the stop level; or 'proj', the projection of Proposition 6.2(iii) onto the static source-slice region,
             a second-order cone program over the weights of the request and the M anchors (exact: the slice of a mixture of fields
             is the mixture of their slices), deployed as that mixture of up to M + 1 fields.
The score of a mixture is exact for PB (slices) and L2 (the mixture is sampled on the probes), a proxy for E and MH (mixed
embeddings), and both for J. Requests: those of e3_enforce.py (8 summer, 8 season not mentioned, 8 disguised, 6 approved test
prompts) and 8 each of the paraphrases, novel winter and novel summer requests; the reference regions are run on the latter only.
Every deployed field is sampled on the evaluation noise and judged as in e3_enforce.py; it is also sampled on the probes, which
places it with respect to the static source-slice and field regions and gives the plug-in certificate of Theorem 7.1 on B_L.
Each (method, request) is saved in ../results/e3/enforce_combined/ when done (a rerun resumes); assembled into
../results/e3/enforce_combined.json and enforce_combined_latents.npz.
Usage: python e3_enforce_combined.py method [method ...] [--only=j1,j2,...] [--dir=folder]   (no method: assemble only)
"""
import os, sys, json, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"                                   # cvxpy segfaults here with multithreaded BLAS
import numpy as np
import torch
import e3_common as C
import e3_regions as R
from e3_combined import standardiser, BETA_DEEP

N_BIS_FIELD, N_BIS_CHEAP = 4, 18
PRUNE = 1e-3                                               # projection weights below this are dropped (and the rest renormalised)
METHODS = {                                                # name: (decision, stop, target); repair by bisection unless 'proj'
    "PB>F:deep": ("PB", "deep", "field"),
    "PB>F": ("PB", "thr", "field"),
    "L2:deep": ("L2", "deep", "own"),
    "J>F:deep": ("J", "deep", "field"),
    "MH>F": ("MH", "thr", "field"),
    "PB:deep": ("PB", "deep", "own"),
    "PB:proj": ("PB", "thr", "proj"),
    "PB>F:local": ("PB", "local", "field"),                # designed after the results of the methods above (Section 11.8)
    "PB": ("PB", "thr", "own"), "L2": ("L2", "thr", "own"), "MH": ("MH", "thr", "own"), "E": ("E", "thr", "own"),
}
REFERENCE = ["PB", "L2", "MH", "E"]                        # the static regions of e3_enforce.py: run on the unfamiliar requests only
R_PRIME = np.linspace(0.02, 1.0, 50)


class Decision:
    """The decision score of a method on features (dict space -> (n, ...) array), with its threshold and deep level."""
    def __init__(self, name, sp, F, anc, cal):
        self.name, self.anc = name, anc
        self.parts = ["PB", "L2"] if name == "J" else [name]
        self.sp, self.F = sp, F
        self.std = {n: standardiser(sp[n].dist(F[n][anc], F[n][anc])) for n in self.parts} if name == "J" else None
        cs = self.scores({n: F[n][cal] for n in self.parts})
        self.eps, self.deep = float(C.kth_threshold(cs, R.BETA)), float(C.kth_threshold(cs, BETA_DEEP))   # Python floats (NEP 50)

    def part(self, n, f):
        return self.sp[n].dist(f, self.F[n][self.anc]).mean(1)

    def scores(self, f):
        if self.name != "J":
            return self.part(self.name, f[self.name])
        return np.max([(self.part(n, f[n]) - self.std[n][0]) / self.std[n][1] for n in self.parts], 0)


def project_slices(S, j, anc, eps):
    """Proposition 6.2(iii) for the static source slice: gamma in the simplex over (request, anchors) minimising the distance to the
    request's slice subject to the mean distance to the anchors' slices being at most eps. Distances from the Gram matrix of the
    slices (RMS over the probes, as in e3_regions.Space('PB'))."""
    import cvxpy as cp
    A = np.concatenate([S[j][None], S[anc]]).reshape(len(anc) + 1, -1).astype(np.float64)
    P = S.shape[1]
    G = A @ A.T / P
    lam, V = np.linalg.eigh(G); keep = lam > 1e-12 * lam.max()
    L = V[:, keep] * np.sqrt(lam[keep]); sc = np.sqrt(np.trace(G) / len(G)); L = L / sc
    I = np.eye(len(anc) + 1)
    g = cp.Variable(len(anc) + 1)
    cons = [g >= 0, cp.sum(g) == 1, sum(cp.norm(L.T @ (g - I[i + 1])) for i in range(len(anc))) <= len(anc) * eps / sc]
    cp.Problem(cp.Minimize(cp.sum_squares(L.T @ (g - I[0]))), cons).solve(solver=cp.CLARABEL)
    gam = np.maximum(g.value, 0); gam /= gam.sum()
    score = lambda gm: np.mean([np.sqrt(max((gm - I[i + 1]) @ G @ (gm - I[i + 1]), 0)) for i in range(len(anc))])
    pr = np.where(gam >= PRUNE, gam, 0); pr /= pr.sum()
    return (pr if score(pr) <= eps else gam), score(gam), float(np.sqrt(max((gam - I[0]) @ G @ (gam - I[0]), 0)))


def main():
    t0 = time.time()
    names = [m for m in sys.argv[1:] if m in METHODS]                       # no method named: only assemble the records
    only = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--only=")), None)     # smoke tests: these requests only,
    out_dir = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--dir=")), None)  # written to another folder
    rows, D, idx = R.load()
    reg = json.load(open(os.path.join(C.RES, "regions.json")))
    anc, cal, te = R.split(idx["approved_eval"], 1000)
    orig = np.r_[idx["summer_eval"][:8], idx["neutral_eval"][:8], idx["disguised_eval"][:8], np.sort(te)[:6]]
    ext = np.r_[idx["paraphrase_eval"][:8], idx["novel_winter_eval"][:8], idx["novel_summer_eval"][:8]]
    rec_dir = out_dir or os.path.join(C.RES, "enforce_combined"); os.makedirs(rec_dir, exist_ok=True)
    if not names:                                                             # assemble the records, no model needed
        return assemble(rec_dir, [int(x) for x in orig], [int(x) for x in ext], [int(x) for x in anc])
    sp = {n: R.Space(n, D, idx, reg["best"][f"{n}|static"]["par"]) for n in ["PB", "L2", "MH", "E"]}
    F = {n: sp[n].feats(D) for n in sp}
    dec = {n: Decision(n, sp, F, anc, cal) for n in ["PB", "L2", "MH", "E", "J"]}
    n1 = lambda f: sp["L2"].dist(f, F["L2"][anc]).min(1)                      # nearest-anchor field score (weights N:1)
    tau_local = float(C.kth_threshold(n1(F["L2"][cal]), BETA_DEEP))           # its median level on the calibration prompts
    pool = np.r_[anc, cal]
    pipe = C.load_pipe()
    chi, judge = C.Clip("openai/clip-vit-large-patch14"), C.Clip("openai/clip-vit-base-patch32")
    nL = chi.tw - chi.to; nL /= np.linalg.norm(nL)                              # B_L = {u : <u, nL> < 0}
    dist_B = D["chi_probe"][anc] @ nL                                          # (M, P): the anchors' probe outputs, distance to B_L
    q_cert = np.array([(dist_B < rp).mean() for rp in R_PRIME])                # static weights: (1/M) sum_i P_i(B^r)
    xp, xe = C.noises(C.PROBE_SEED, C.P_PROBE), C.noises(C.EVAL_SEED, C.P_EVAL)
    cond_cache = {}

    def cond(j):
        if j not in cond_cache:
            c, _, neg = C.encode(pipe, [rows[j]["prompt"]]); cond_cache[j] = c[0]; cond_cache["neg"] = neg
        return cond_cache[j]

    cond(int(orig[0])); neg = cond_cache["neg"]
    gen_cache = {}

    def mixed(js, gams, tag):
        """Latents of the mixture sum_a gams[a] v(.; js[a]) on the probe or evaluation noises (cached; stored latents for one field)."""
        js, gams = [int(a) for a in js], [float(g) for g in gams]
        if len(js) == 1 or max(gams) >= 1:
            return torch.tensor(D[f"lat_{tag}"][js[int(np.argmax(gams))]].astype(np.float32))
        key = (tuple(js), tuple(round(g, 6) for g in gams), tag)
        if key not in gen_cache:
            gen_cache[key] = C.sample(pipe, neg, [cond(a) for a in js], gams, xp if tag == "probe" else xe)
        return gen_cache[key]

    def mix_feats(js, gams, parts):
        """Features of a mixture of fields in the spaces 'parts': exact slices (PB), sampled outputs (L2), mixed embeddings (E, MH)."""
        if len(js) == 1:
            return {n: F[n][js[0]][None] for n in parts}
        out = {}
        for n in parts:
            if n == "L2":
                out[n] = chi.image(C.decode(pipe, mixed(js, gams, "probe")))[None]
            else:
                out[n] = sum(g * F[n][a].astype(np.float32) for a, g in zip(js, gams))[None]
        return out

    def judge_feats(lat, j):
        imgs = C.decode(pipe, lat)
        fl, fb = chi.image(imgs), judge.image(imgs)
        subj = judge.text([rows[j]["subject_text"]])[0]
        return dict(pwB=judge.p_winter(fb).tolist(), pwL=chi.p_winter(fl).tolist(), snow=C.snow_fraction(imgs).tolist(),
                    green=C.green_fraction(imgs).tolist(), subj=(fb @ subj).tolist()), fl

    for name in names:
        dname, stop, rule = METHODS[name]
        d = dec[dname]
        level = d.deep if stop == "deep" else d.eps
        # 'local': the decision's threshold, and also a field distance to the target at most the median nearest-anchor field
        # score of the calibration prompts: the deployed field is as close to an approved field (its target) as half of the
        # approved prompts are to their nearest anchor
        parts = d.parts + (["L2"] if stop == "local" and "L2" not in d.parts else [])
        ok = lambda f, c: bool(d.scores(f)[0] <= level and
                               (stop != "local" or sp["L2"].dist(f["L2"], F["L2"][c][None])[0, 0] <= tau_local))
        feas = pool[d.scores({n: F[n][pool] for n in d.parts}) <= level]
        reqs = ext if name in REFERENCE else np.r_[orig, ext]
        if only:
            reqs = [int(a) for a in only.split(",")]
        print(f"[{time.time()-t0:.0f}s] {name}: decision {dname} eps {d.eps:.4f}, stop {stop} ({level:.4f}), target {rule}, "
              f"{len(feas)} feasible targets", flush=True)
        for j in [int(x) for x in reqs]:
            rec_file = os.path.join(rec_dir, f"{name.replace('>', '-').replace(':', '_')}_{j:04d}.npz")
            if os.path.exists(rec_file):
                continue
            t1 = time.time()
            s0 = float(d.scores({n: F[n][j][None] for n in d.parts})[0])
            rec = dict(method=name, decision=dname, stop=stop, rule=rule, j=j, kind=rows[j]["kind"], prompt=rows[j]["prompt"],
                       s0=s0 / d.eps, eps=float(d.eps), level=float(level))
            if s0 <= d.eps:
                js, gams, c = [j], [1.0], None
            elif rule == "proj":
                gam, s_proj, moved = project_slices(F["PB"], j, anc, d.eps)
                keep = np.where(gam > 0)[0]
                js = [j if a == 0 else int(anc[a - 1]) for a in keep]; gams = gam[keep].tolist()
                c = int(anc[np.argmax(gam[1:])])
                rec.update(gamma=gam.tolist(), proj_score=float(s_proj / d.eps), proj_moved=moved, n_fields=len(js))
            else:
                ts = "L2" if rule == "field" else dname                         # the distance that picks the target
                c = int(feas[np.argmin(sp[ts].dist(F[ts][j][None], F[ts][feas])[0])])
                lo, hi = 0.0, 1.0
                for _ in range(N_BIS_FIELD if "L2" in parts else N_BIS_CHEAP):
                    mid = (lo + hi) / 2
                    if ok(mix_feats([j, c], [1 - mid, mid], parts), c):
                        hi = mid
                    else:
                        lo = mid
                js, gams = ([c], [1.0]) if hi >= 1 else ([j, c], [1 - hi, hi])
            kappa = 1.0 - (gams[js.index(j)] if j in js else 0.0)
            rec.update(kappa=float(kappa), target=c, target_prompt=None if c is None else rows[c]["prompt"],
                       same_scene=None if c is None else bool(rows[c]["scene"] == rows[j]["scene"]))
            lat = mixed(js, gams, "eval")
            jf, fl = judge_feats(lat, j); rec.update(jf)
            # where the deployed field lies: static source-slice and field scores (relative to their thresholds) and the plug-in
            # certificate of Theorem 7.1 on B_L with the deployed field's own field score
            f_dep = mix_feats(js, gams, ["PB", "L2"])
            sPB, sL2 = float(dec["PB"].scores(f_dep)[0]), float(dec["L2"].scores(f_dep)[0])
            bound = q_cert + sL2 / R_PRIME
            rec.update(sPB_dep=sPB / dec["PB"].eps, sL2_dep=sL2 / dec["L2"].eps, sJ_dep=float(dec["J"].scores(f_dep)[0] / dec["J"].eps),
                       n1_dep=float(n1(f_dep["L2"])[0] / tau_local),
                       tgt_dep=None if c is None else float(sp["L2"].dist(f_dep["L2"], F["L2"][c][None])[0, 0] / tau_local),
                       cert=float(bound.min()), cert_r=float(R_PRIME[bound.argmin()]), actual_BL=float(np.mean(np.array(rec["pwL"]) < 0.5)),
                       chi_change=float(np.mean(np.linalg.norm(fl - D["chi_eval"][j], axis=1))), secs=time.time() - t1)
            tmp = rec_file[:-4] + ".tmp.npz"
            np.savez(tmp, rec=np.array(json.dumps(rec)), lat=lat.numpy().astype(np.float16)); os.replace(tmp, rec_file)
            print(f"   {rec['kind']:12s} {j:3d}: s0/eps {rec['s0']:.2f} kappa {kappa:.3f} | P(winter) {np.mean(rec['pwB']):.2f} "
                  f"unsafe {np.mean(np.array(rec['pwB']) < 0.5):.2f} subj {np.mean(rec['subj']):.3f} same scene {rec['same_scene']} | "
                  f"PB {rec['sPB_dep']:.2f} L2 {rec['sL2_dep']:.2f} | {rec['secs']:.0f}s", flush=True)
    if not out_dir:
        assemble(rec_dir, [int(x) for x in orig], [int(x) for x in ext], [int(x) for x in anc])
    print(f"[{time.time()-t0:.0f}s] done")


def assemble(rec_dir, orig, ext, anc):
    out, lats = [], {}
    for f in sorted(os.listdir(rec_dir)):
        if f.endswith(".npz") and ".tmp" not in f:
            d = np.load(os.path.join(rec_dir, f)); rec = json.loads(str(d["rec"]))
            out.append(rec); lats[f"{rec['method']}|{rec['j']}"] = d["lat"]
    json.dump(dict(requests=orig, unfamiliar=ext, anchors=anc, methods=METHODS, rows=out),
              open(os.path.join(C.RES, "enforce_combined.json"), "w"), indent=0)
    np.savez_compressed(os.path.join(C.RES, "enforce_combined_latents.npz"), **lats)


if __name__ == "__main__":
    main()
