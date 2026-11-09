#!/bin/bash
# E3 pipeline (safe_prompt_fields.tex, text-to-image test on Sana 600M). Every step resumes where it stopped:
#   e3_pool.py skips the prompts already in results/e3/pool/, e3_enforce.py the (region, request) pairs in results/e3/enforce/.
# Usage, from this folder:  ./e3_run.sh            (or, to keep the Mac awake and log to a file:
#                           caffeinate -i ./e3_run.sh > ../results/e3_run.log 2>&1 &)
set -e
cd "$(dirname "$0")"
[ -f ../results/e3/prompts.json ] || python e3_prompts.py
python e3_prompts.py --extend   # appends 40 safe-but-unfamiliar requests once (earlier prompts unchanged)
python e3_pool.py          # ~25 s per evaluation prompt, ~14 s per tuning prompt (310 prompts in all)
python e3_regions.py       # a few minutes
python e3_repairs.py       # seconds: mechanism and anchor count
python e3_enforce.py       # ~75 min
python e3_figures.py       # galleries, detection and enforcement figures, tables
echo "E3 done"
