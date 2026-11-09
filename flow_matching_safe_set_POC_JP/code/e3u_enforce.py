"""E3u, step 3: enforcement with a safe and an unsafe set (Section 12 of safe_prompt_fields.tex).

Split. The field and prompt sets are small only when every scene has anchors of both classes (anchor sweep of e3u_regions.py), so
enforcement pools each class (190 approved prompts; the 190 summer prompts of the unsafe history) and uses the first split of that sweep:
M = 140 anchors, K = 40 calibration prompts and 10 test prompts per class (e3_regions.split, seeds 7000 and 8000).
A method is a list of DECISION conditions and a list of STOP conditions. A condition is (space, class, weighting):
  (space, 'safe', w)    the candidate field lies in the safe set of the request in that space,
  (space, 'unsafe', w)  the candidate field lies OUTSIDE the unsafe set of the request in that space,
  ('L2', 'local', None) the field distance between the candidate and the target is at most tau, the median nearest-anchor field
                        distance of the safe calibration prompts (the local stop of Section 11.8),
with the sets of e3u_regions.py and weightings 'U' (static) or 'XN:1' (all the weight on the anchor of the class whose prompt is nearest
to the request; the smallest sets of the sweep at M = 140 in the prompt and field spaces). The weights are those of the REQUEST's
prompt and stay fixed while its field is repaired, so the sets of a request are fixed sets of fields.
A request is admitted unchanged if it meets every decision condition. Otherwise its TARGET is the approved anchor or calibration prompt
nearest to it in the field distance L2 among those whose own field meets every stop condition for this request, and the deployed field
is (1 - kappa) v(.; request) + kappa v(.; target) with the smallest kappa found by bisection whose mixture meets every stop condition
(Proposition 6.4(i): the upper end of the bisection always meets them, since the target does). If no approved prompt meets the stop
conditions, the method falls back to the safe conditions alone (fallback 1), and if none meets these, to the nearest approved prompt,
deployed alone (fallback 2). A condition of a mixture is computed on its mixed source slices (exact), on its outputs sampled on the
probes (exact, N_BIS_FIELD bisection steps) or on its mixed prompt embedding (a proxy, as in e3_enforce.py).
Requests: 8 summer, 8 season not mentioned, 8 disguised requests, the approved test prompts of the split that have evaluation outputs,
and 8 each of the paraphrases, novel winter and novel summer requests (those of e3_enforce_combined.py). Every deployed field is sampled
on the evaluation noise and judged as in e3_enforce.py, and on the probes, which places it with respect to every set; the bisection
trace (kappa, condition scores) is kept.
One file per (method, request) in ../results/e3/enforce_unsafe/ (a rerun resumes); assembled into ../results/e3/enforce_unsafe.json and
enforce_unsafe_latents.npz.     Usage: python e3u_enforce.py method [method ...] [--only=j1,j2]   (no method: assemble only)
"""
import os, sys, json, time
import numpy as np
import torch
import e3_common as C
import e3_regions as R
import e3u_regions as U

M_ENF, SEED_S, SEED_U = 140, 7000, 8000
N_BIS_FIELD, N_BIS_CHEAP = 4, 18
NEAR = "XN:1"
PB_S, PB_U = ("PB", "safe", "U"), ("PB", "unsafe", "U")
F_S, F_U = ("L2", "safe", NEAR), ("L2", "unsafe", NEAR)
P_S, P_U = ("E", "safe", NEAR), ("E", "unsafe", NEAR)
LOCAL = ("L2", "local", None)
METHODS = {
    # Section 11.8 on this split: the static source slice decides, the target is chosen in the field distance, and the repair stops at
    # the threshold of the source slice, or also within the median nearest-anchor field distance of the target (local)
    "S>F": dict(decide=[PB_S], stop=[PB_S]),
    "S>F:local": dict(decide=[PB_S], stop=[PB_S, LOCAL]),
    # two sets in one space: the space decides and stops, with its safe and its unsafe set
    "S2": dict(decide=[PB_S, PB_U], stop=[PB_S, PB_U]),
    "F2": dict(decide=[F_S, F_U], stop=[F_S, F_U]),
    "P2": dict(decide=[P_S, P_U], stop=[P_S, P_U]),
    # field, safe set only, with the same nearest-prompt weights: what the unsafe set adds to F2
    "F1": dict(decide=[F_S], stop=[F_S]),
    # the source slice's two sets decide; the repair stops inside them and inside the field's two sets
    "S>F2": dict(decide=[PB_S, PB_U], stop=[PB_S, PB_U, F_S, F_U]),
}
PLACE = [(sp, cls, w) for sp in ["PB", "L2", "E"] for cls in ["safe", "unsafe"] for w in ["U", NEAR]]


def kname(key):
    return "|".join("-" if k is None else k for k in key)


def main():
    t0 = time.time()
    names = [m for m in sys.argv[1:] if m in METHODS]
    only = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--only=")), None)
    rows, D, Fe, idx = U.load_all()
    app = np.r_[idx["approved_tune"], idx["approved_eval"]]; uns = np.r_[idx["unsafe_tune"], idx["unsafe_eval"]]
    a_s, c_s, t_s = R.split(app, SEED_S, m=M_ENF, k=R.K)
    a_u, c_u, t_u = R.split(uns, SEED_U, m=M_ENF, k=R.K)
    app_test = np.array(sorted(j for j in t_s if rows[j]["pool"] == "eval"))      # those with evaluation outputs
    orig = np.r_[idx["summer_eval"][:8], idx["neutral_eval"][:8], idx["disguised_eval"][:8], app_test]
    ext = np.r_[idx["paraphrase_eval"][:8], idx["novel_winter_eval"][:8], idx["novel_summer_eval"][:8]]
    rec_dir = os.path.join(C.RES, "enforce_unsafe"); os.makedirs(rec_dir, exist_ok=True)
    if not names:
        return assemble(rec_dir, orig, ext, a_s, a_u)
    E = U.dist_matrix("E", Fe)
    Dm = {"E": E, "PB": U.dist_matrix("PB", Fe), "L2": U.dist_matrix("L2", Fe)}
    feat = {"E": Fe["pooled"], "PB": Fe["slice"].astype(np.float32), "L2": Fe["chi_probe"]}
    space = {n: R.Space(n, Fe, None) for n in Dm}
    SETS = {}
    for sp in Dm:
        for w in ["U", NEAR]:
            SETS[(sp, "safe", w)] = U.OneSet(Dm[sp], E, U.wparse(w), a_s, c_s)
            SETS[(sp, "unsafe", w)] = U.OneSet(Dm[sp], E, U.wparse(w), a_u, c_u)
    tau = float(C.kth_threshold(Dm["L2"][np.ix_(c_s, a_s)].min(1), 0.5))      # local stop: median nearest-anchor field distance
    pool = np.r_[a_s, c_s]                                                     # candidate targets: approved anchors and calibration
    print(f"[{time.time()-t0:.0f}s] split: {len(a_s)}/{len(c_s)}/{len(t_s)} approved, {len(a_u)}/{len(c_u)}/{len(t_u)} unsafe; "
          f"thresholds " + ", ".join(f"{kname(k)} {s.eps:.4g}" for k, s in SETS.items()) + f"; tau {tau:.4f}", flush=True)
    pipe = C.load_pipe()
    chi, judge = C.Clip("openai/clip-vit-large-patch14"), C.Clip("openai/clip-vit-base-patch32")
    xp, xe = C.noises(C.PROBE_SEED, C.P_PROBE), C.noises(C.EVAL_SEED, C.P_EVAL)
    cond_cache, gen_cache = {}, {}

    def cond(j):
        if j not in cond_cache:
            c, _, neg = C.encode(pipe, [rows[j]["prompt"]]); cond_cache[j] = c[0]; cond_cache["neg"] = neg
        return cond_cache[j]

    cond(int(orig[0])); neg = cond_cache["neg"]

    def mixed(js, gams, tag):
        """Latents of the mixture sum_a gams[a] v(.; js[a]) on the probe or evaluation noises; the stored latents of a single prompt when
        it has them (tuning prompts have no evaluation outputs and are sampled)."""
        js, gams = [int(a) for a in js], [float(g) for g in gams]
        if len(js) == 1 or max(gams) >= 1:
            a = js[int(np.argmax(gams))]
            if a < len(D[f"lat_{tag}"]) and not np.isnan(D[f"lat_{tag}"][a][0, 0, 0, 0]):
                return torch.tensor(D[f"lat_{tag}"][a].astype(np.float32))
            js, gams = [a], [1.0]
        key = (tuple(js), tuple(round(g, 6) for g in gams), tag)
        if key not in gen_cache:
            gen_cache[key] = C.sample(pipe, neg, [cond(a) for a in js], gams, xp if tag == "probe" else xe)
        return gen_cache[key]

    def mix_feats(js, gams, spaces):
        if len(js) == 1:
            return {n: feat[n][js[0]][None] for n in spaces}
        out = {}
        for n in spaces:
            out[n] = (chi.image(C.decode(pipe, mixed(js, gams, "probe")))[None] if n == "L2"
                      else sum(g * feat[n][a] for a, g in zip(js, gams))[None])
        return out

    def scores(f, wq, conds, tgt=None):
        """Score / threshold of each condition for features f (dict space -> (1, ...)), with the request's weights wq; for the local
        condition, field distance to the target / tau."""
        out = {}
        for key in conds:
            if key[1] == "local":
                out[key] = float(space["L2"].dist(f["L2"], feat["L2"][tgt][None])[0, 0] / tau)
                continue
            st = SETS[key]
            d = space[key[0]].dist(f[key[0]], feat[key[0]][st.anc])
            out[key] = float(st.score_with(d, wq[key])[0] / st.eps)
        return out

    def meets(sc):
        return all((v > 1.0) if key[1] == "unsafe" else (v <= 1.0) for key, v in sc.items())

    def judge_feats(lat, j):
        imgs = C.decode(pipe, lat)
        fl, fb = chi.image(imgs), judge.image(imgs)
        subj = judge.text([rows[j]["subject_text"]])[0]
        return dict(pwB=judge.p_winter(fb).tolist(), pwL=chi.p_winter(fl).tolist(), snow=C.snow_fraction(imgs).tolist(),
                    green=C.green_fraction(imgs).tolist(), subj=(fb @ subj).tolist()), fl

    for name in names:
        spec = METHODS[name]
        dec, stp = spec["decide"], spec["stop"]
        spaces = sorted({k[0] for k in dec + stp})
        reqs = [int(a) for a in only.split(",")] if only else [int(x) for x in np.r_[orig, ext]]
        print(f"[{time.time()-t0:.0f}s] {name}: decide {[kname(k) for k in dec]} | stop {[kname(k) for k in stp]}", flush=True)
        for j in reqs:
            rec_file = os.path.join(rec_dir, f"{name.replace('>', '-').replace(':', '_')}_{j:04d}.npz")
            if os.path.exists(rec_file):
                continue
            t1 = time.time()
            wq = {key: SETS[key].weights([j]) for key in PLACE}               # the request's weights, fixed during the repair
            f0 = {n: feat[n][j][None] for n in spaces}
            s0 = scores(f0, wq, [k for k in dec + stp if k[1] != "local"])
            rec = dict(method=name, j=j, kind=rows[j]["kind"], prompt=rows[j]["prompt"], s0={kname(k): v for k, v in s0.items()},
                       admitted=meets({k: s0[k] for k in dec}))
            trace, fallback = [], 0
            if rec["admitted"]:
                js, gams, c = [j], [1.0], None
            else:
                safe_stp = [k for k in stp if k[1] != "unsafe"]
                ok_t = np.array([meets(scores({n: feat[n][t][None] for n in spaces}, wq, stp, t)) for t in pool])
                if not ok_t.any():
                    fallback = 1
                    ok_t = np.array([meets(scores({n: feat[n][t][None] for n in spaces}, wq, safe_stp, t)) for t in pool])
                if not ok_t.any():
                    fallback, ok_t = 2, np.ones(len(pool), bool)
                cand = pool[ok_t]
                c = int(cand[np.argmin(space["L2"].dist(feat["L2"][j][None], feat["L2"][cand])[0])])
                use = stp if fallback == 0 else safe_stp
                lo, hi = 0.0, 1.0
                if fallback < 2:
                    for _ in range(N_BIS_FIELD if any(k[0] == "L2" for k in use) else N_BIS_CHEAP):
                        mid = (lo + hi) / 2
                        sc = scores(mix_feats([j, c], [1 - mid, mid], spaces), wq, use, c)
                        trace.append(dict(kappa=mid, **{kname(k): v for k, v in sc.items()}))
                        if meets(sc):
                            hi = mid
                        else:
                            lo = mid
                js, gams = ([c], [1.0]) if hi >= 1 else ([j, c], [1 - hi, hi])
            kappa = 1.0 - (gams[js.index(j)] if j in js else 0.0)
            rec.update(kappa=float(kappa), target=c, fallback=fallback, trace=trace,
                       target_prompt=None if c is None else rows[c]["prompt"],
                       same_scene=None if c is None else bool(rows[c]["scene"] == rows[j]["scene"]))
            lat = mixed(js, gams, "eval")
            jf, fl = judge_feats(lat, j); rec.update(jf)
            f_dep = mix_feats(js, gams, ["PB", "L2", "E"])
            rec["place"] = {kname(k): v for k, v in scores(f_dep, wq, PLACE).items()}
            if c is not None:
                rec["place"]["L2|local|-"] = scores(f_dep, wq, [LOCAL], c)[LOCAL]
            rec["meets_stop"] = meets({k: rec["place"][kname(k)] for k in stp}) if c is not None else True
            rec.update(chi_change=float(np.mean(np.linalg.norm(fl - D["chi_eval"][j], axis=1))), secs=time.time() - t1)
            tmp = rec_file[:-4] + ".tmp.npz"
            np.savez(tmp, rec=np.array(json.dumps(rec)), lat=lat.numpy().astype(np.float16)); os.replace(tmp, rec_file)
            print(f"   {rec['kind']:12s} {j:3d}: adm {rec['admitted']!s:5s} kappa {kappa:.3f} fb {fallback} | P(winter) "
                  f"{np.mean(rec['pwB']):.2f} unsafe {np.mean(np.array(rec['pwB']) < 0.5):.2f} subj {np.mean(rec['subj']):.3f} "
                  f"same {rec['same_scene']} meets {rec['meets_stop']} | {rec['secs']:.0f}s", flush=True)
            if len(gen_cache) > 64:
                gen_cache.clear(); torch.mps.empty_cache()
    assemble(rec_dir, orig, ext, a_s, a_u)
    print(f"[{time.time()-t0:.0f}s] done", flush=True)


def assemble(rec_dir, orig, ext, a_s, a_u):
    out, lats = [], {}
    for f in sorted(os.listdir(rec_dir)):
        if f.endswith(".npz") and ".tmp" not in f:
            d = np.load(os.path.join(rec_dir, f)); rec = json.loads(str(d["rec"]))
            out.append(rec); lats[f"{rec['method']}|{rec['j']}"] = d["lat"]
    meth = {k: dict(decide=[kname(c) for c in v["decide"]], stop=[kname(c) for c in v["stop"]]) for k, v in METHODS.items()}
    json.dump(dict(requests=[int(x) for x in orig], unfamiliar=[int(x) for x in ext], anchors_safe=[int(x) for x in a_s],
                   anchors_unsafe=[int(x) for x in a_u], m_enf=M_ENF, methods=meth, rows=out),
              open(os.path.join(C.RES, "enforce_unsafe.json"), "w"), indent=0)
    np.savez_compressed(os.path.join(C.RES, "enforce_unsafe_latents.npz"), **lats)


if __name__ == "__main__":
    main()
