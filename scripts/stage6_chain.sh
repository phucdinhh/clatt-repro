#!/usr/bin/env bash
# Stage 6 only (restart of the tail of stage5_6_chain.sh after the pruner warmup fix): 10 studies -> report + check 5.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/logs/stage5_6_chain.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

for m in CLATT CNN-BiLSTM CNN-LSTM LSTM CNN; do
  for g in 1 2; do
    say "tune $m group $g"
    python -m src.tune --model "$m" --group "$g" > "results/logs/tune_${m}_g${g}.log" 2>&1
  done
done
python -m src.tune --report >> "$LOG" 2>&1
python -c "from scripts.leakage_checks import chon_sieu_tham_so as c; print(c())" >> "$LOG" 2>&1
say "stage 6 done"
