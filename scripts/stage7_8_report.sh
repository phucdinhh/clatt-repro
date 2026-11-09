#!/usr/bin/env bash
# Stage 7.4-7.6 and 8.2 tables and figures from the finished runs, then the acceptance checks.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
"$PY" -m src.evaluate main_A
"$PY" -m src.evaluate main_B --baselines main_A
"$PY" -m src.evaluate --compare main_A main_B
"$PY" -m src.stats main_A
"$PY" -m src.stats main_B --baselines main_A
"$PY" -m src.stats ablation
"$PY" -m src.plots --final main_A main_B
"$PY" -m src.plots --ablation
"$PY" -m src.freeze check
"$PY" -c "from scripts.leakage_checks import chon_sieu_tham_so as c; res = c(); print(res); assert res['status'] == 'PASS'"
