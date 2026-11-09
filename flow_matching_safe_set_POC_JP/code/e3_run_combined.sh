#!/bin/bash
# Section 11.8 of safe_prompt_fields.tex (methods that decide with one distance and repair with another), after e3_run.sh.
# Every step resumes: e3_enforce_combined.py skips the (method, request) pairs already in results/e3/enforce_combined/.
# Run the steps one after another: on the Apple GPU, parallel processes are slower than one (measured: about 4x per record).
# Usage, from this folder:  caffeinate -i ./e3_run_combined.sh > ../results/e3/logs/combined.log 2>&1 &
cd "$(dirname "$0")"
python e3_combined.py                                                    # seconds: joint and Bonferroni regions, 200 splits
python e3_enforce_combined.py "PB>F:deep" "PB>F" "L2:deep" "MH>F" "J>F:deep"   # the planned new methods, ~2 h
python e3_enforce_combined.py "PB>F:local"                               # added after their results, ~45 min
python e3_enforce_combined.py PB L2 MH E                                 # previous regions on the unfamiliar requests, ~20 min
python e3_enforce_combined.py "PB:proj" --only=206,207,208,209,210,211,212,213   # projection control, 8 summer requests, ~35 min
python e3_enforce_combined.py "PB:deep"                                  # deep back-off with the source slice's own target, ~20 min
python e3_enforce_combined.py                                            # no method: assembles enforce_combined.json
python e3_figures_combined.py                                            # figures, tables and numbers of Section 11.8
echo "E3 combined done"
