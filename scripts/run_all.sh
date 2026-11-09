#!/usr/bin/env bash
# One command from raw data to every stage 7-8 table and figure (sequential).
# Quick trial on a new machine: SEEDS_A="0" SEEDS_B="0" bash scripts/run_all.sh
# Full run (about 25-35 CPU hours): bash scripts/run_all.sh. For the side-by-side version open three
# terminals and run scripts/stage7_lane.sh 1, 2 and 3, then scripts/stage7_8_report.sh.
# Finished runs are skipped, so the script can be restarted after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
export PY=${PY:-.venv/bin/python}
"$PY" -m pytest -q
"$PY" -m src.dataset                 # stage 1 tables and figures
"$PY" -m src.models                  # parameter table
"$PY" -m scripts.leakage_checks      # stage 4, about 10 min
"$PY" -m src.freeze check            # configs as frozen before the test split was read
for lane in 1 3 2; do bash scripts/stage7_lane.sh "$lane"; done   # lane 2 waits for main_A, which lane 1 writes
bash scripts/stage7_8_report.sh
