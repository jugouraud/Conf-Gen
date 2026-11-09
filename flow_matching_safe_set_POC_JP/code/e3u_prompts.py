"""E3u (Section 12 of safe_prompt_fields.tex): the HISTORY of unsafe prompts, from which the unsafe sets are built.

The summer prompts of e3_prompts.py are the first 40 entries of one random permutation of the 400 summer combinations (scene x summer
modifier x light), whose last 16 entries are the disguised requests. The historical unsafe prompts are the next entries of the SAME
permutation, so that they are exchangeable with the 24 summer requests of the evaluation pool (Assumption 5.1 for the unsafe class):
54 for tuning, which join the 16 summer prompts of the tuning pool, and 120 for evaluation (unsafe anchors, unsafe calibration prompts
and unsafe test prompts). Their ids continue those of prompts.json, which is not modified.
Writes ../results/e3/prompts_unsafe.json.     Usage: python e3u_prompts.py
"""
import os, json
import numpy as np
from e3_prompts import SCENES, LIGHTS, WINTER, SUMMER, SIZES, RES

N_TUNE, N_EVAL = 54, 120


def main():
    rng = np.random.default_rng(31)                                  # the draws of e3_prompts.main(), in the same order
    combos = lambda mods: [(s, m, l) for s in range(len(SCENES)) for m in range(len(mods)) for l in range(len(LIGHTS))]
    rng.permutation(len(combos(WINTER)))
    summ_all = [combos(SUMMER)[i] for i in rng.permutation(len(combos(SUMMER)))]
    rows = json.load(open(os.path.join(RES, "prompts.json")))["prompts"]
    n_sum = SIZES["summer_tune"] + SIZES["summer_eval"]
    old = [r["prompt"] for r in rows if r["kind"] == "summer"]
    mine = [f"a photograph of {SCENES[s]}, {SUMMER[m]}, {LIGHTS[l]}" for s, m, l in summ_all[:n_sum]]
    assert old == mine, "the permutation of e3_prompts.py was not reproduced"
    disg = {r["prompt"][:-len(", snowy winter cold")] for r in rows if r["kind"] == "disguised"}
    new = summ_all[n_sum:n_sum + N_TUNE + N_EVAL]
    assert len(new) == N_TUNE + N_EVAL and n_sum + N_TUNE + N_EVAL <= len(summ_all) - SIZES["disguised_eval"]
    out = []
    for i, (s, m, l) in enumerate(new):
        p = f"a photograph of {SCENES[s]}, {SUMMER[m]}, {LIGHTS[l]}"
        assert p not in old and p not in disg
        out.append(dict(id=len(rows) + i, kind="summer_hist", pool="tune" if i < N_TUNE else "eval", scene=s, prompt=p,
                        subject_text=f"a photograph of {SCENES[s]}"))
    json.dump(dict(n_tune=N_TUNE, n_eval=N_EVAL, first_id=len(rows), prompts=out),
              open(os.path.join(RES, "prompts_unsafe.json"), "w"), indent=1)
    from collections import Counter
    print(f"{len(out)} historical unsafe prompts, ids {out[0]['id']}-{out[-1]['id']}:", Counter(r["pool"] for r in out),
          "| scenes per pool:", {p: len({r['scene'] for r in out if r['pool'] == p}) for p in ["tune", "eval"]})


if __name__ == "__main__":
    main()
