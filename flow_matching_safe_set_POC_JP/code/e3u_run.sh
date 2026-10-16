#!/bin/bash
# Section 12 of safe_prompt_fields.tex (a safe and an unsafe set), after e3_run.sh and e3_run_combined.sh.
# Every step resumes: e3u_pool.py skips the prompts already in results/e3/pool_unsafe/, e3u_enforce.py the (method, request) pairs
# already in results/e3/enforce_unsafe/. Run one process at a time on the Apple GPU.
# Usage, from this folder:  caffeinate -i ./e3u_run.sh > ../results/e3/logs/e3u_run.log 2>&1 &
set -e
cd "$(dirname "$0")"
python e3u_prompts.py                                   # 174 historical unsafe prompts (prompts_unsafe.json), seconds
python e3u_pool.py                                      # their probe outputs, ~13 s per prompt (~40 min)
python e3u_regions.py                                   # tuning, detection over 200 splits, anchor sweep, ~2 min
python e3u_enforce.py S2 "S>F2" F2 "S>F" "S>F:local" F1 P2   # seven methods on the M = 140 split, ~5 h in all
python e3u_enforce.py                                   # no method: assembles enforce_unsafe.json (needed by the next step)
python e3u_segment.py 209,237,236                       # scores along three repair segments (Figure fig_e3u_segment), ~5 min
python e3u_figures.py                                   # figures, tables and numbers of Section 12
echo "E3u done"
