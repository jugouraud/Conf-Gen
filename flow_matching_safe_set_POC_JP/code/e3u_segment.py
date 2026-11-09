"""E3u, step 4: the safe and unsafe scores of a repair along its whole segment, for a few requests (Figure fig_e3u_segment of
Section 12 of safe_prompt_fields.tex).

For each chosen request c and the target t of method 'S>F2' in ../results/e3/enforce_unsafe.json (the split and the sets of
e3u_enforce.py: M = 140 anchors of each class; source slice with static weights, field with the nearest prompt of each class), the
mixtures v_kappa = (1 - kappa) v_c + kappa v_t are scored with the request's weights, relative to the thresholds: in the source slice
exactly, on a fine grid, from the stored slices (the slice of a mixture is the mixture of the slices), and in the field distance on a
coarse grid, by sampling each mixture on the probes. Writes ../results/e3/segment_unsafe.json.
Usage: python e3u_segment.py j1,j2,...
"""
import os, sys, json, time
import numpy as np
import e3_common as C
import e3_regions as R
import e3u_regions as U
from e3u_enforce import M_ENF, SEED_S, SEED_U, NEAR

FINE, COARSE = np.linspace(0, 1, 101), np.linspace(0, 1, 9)


def main():
    t0 = time.time()
    rows, D, Fe, idx = U.load_all()
    enf = json.load(open(os.path.join(C.RES, "enforce_unsafe.json")))
    recs = {r["j"]: r for r in enf["rows"] if r["method"] == "S>F2" and r["target"] is not None}
    picks = [int(a) for a in sys.argv[1].split(",") if int(a) in recs]
    app = np.r_[idx["approved_tune"], idx["approved_eval"]]; uns = np.r_[idx["unsafe_tune"], idx["unsafe_eval"]]
    a_s, c_s, _ = R.split(app, SEED_S, m=M_ENF, k=R.K)
    a_u, c_u, _ = R.split(uns, SEED_U, m=M_ENF, k=R.K)
    E = U.dist_matrix("E", Fe)
    feat = {"PB": Fe["slice"].astype(np.float32), "L2": Fe["chi_probe"]}
    space = {n: R.Space(n, Fe, None) for n in feat}
    SETS = {}
    for sp, w in [("PB", "U"), ("L2", NEAR)]:
        Dsp = U.dist_matrix(sp, Fe)
        SETS[(sp, "safe", w)] = U.OneSet(Dsp, E, U.wparse(w), a_s, c_s)
        SETS[(sp, "unsafe", w)] = U.OneSet(Dsp, E, U.wparse(w), a_u, c_u)
    pipe = C.load_pipe()
    chi = C.Clip("openai/clip-vit-large-patch14")
    xp = C.noises(C.PROBE_SEED, C.P_PROBE)
    out = {}
    for j in picks:
        t = recs[j]["target"]
        conds, _, neg = C.encode(pipe, [rows[j]["prompt"], rows[t]["prompt"]])
        res = {"target": t, "fine": FINE.tolist(), "coarse": COARSE.tolist()}
        f_l2 = {}
        for k in COARSE:
            f_l2[float(k)] = (feat["L2"][j if k == 0 else t][None] if k in (0, 1)
                              else chi.image(C.decode(pipe, C.sample(pipe, neg, conds, [1 - k, k], xp)))[None])
        for key, st in SETS.items():
            w = st.weights([j])
            if key[0] == "PB":
                vals = [st.score_with(space["PB"].dist(((1 - k) * feat["PB"][j] + k * feat["PB"][t])[None], feat["PB"][st.anc]), w)[0]
                        for k in FINE]
            else:
                vals = [st.score_with(space["L2"].dist(f_l2[float(k)], feat["L2"][st.anc]), w)[0] for k in COARSE]
            res["|".join(key)] = (np.array(vals) / st.eps).tolist()
        out[j] = res
        print(f"[{time.time()-t0:.0f}s] {rows[j]['kind']:9s} {j}: target {t} | field safe "
              f"{np.round(res['L2|safe|' + NEAR], 2).tolist()} unsafe {np.round(res['L2|unsafe|' + NEAR], 2).tolist()}", flush=True)
    json.dump(out, open(os.path.join(C.RES, "segment_unsafe.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
