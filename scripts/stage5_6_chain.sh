#!/usr/bin/env bash
# Unattended chain (user authorised decisions on 2026-11-02; prefer the paper's values when the rule allows):
# wait for sweep_L -> L* -> sweep_loss -> loss* -> write T*, L*, loss* to base.yaml -> stage 6 tuning -> report + check 5.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/logs/stage5_6_chain.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
proposed() { python -c "import yaml; print(yaml.safe_load(open('configs/proposed_stage5.yaml'))['$1'])"; }

while pgrep -f "scripts/stage5.sh L" >/dev/null; do sleep 60; done
say "sweep_L finished"
python -m scripts.stage5_report >> "$LOG" 2>&1
L=$(proposed layers)
say "L*=$L, starting sweep_loss (mae)"
scripts/stage5.sh loss 90 "$L" mae > results/logs/stage5_sweep_loss.out 2>&1
python -m scripts.stage5_report >> "$LOG" 2>&1
LOSS=$(proposed loss)
say "loss*=$LOSS; writing T=90 layers=$L loss=$LOSS to configs/base.yaml"
python - "$L" "$LOSS" <<'EOF'
import re, sys
from pathlib import Path
p = Path("configs/base.yaml"); s = p.read_text()
for key, value, note in [("T", "90", "stage 5 T* on val (paper: 68), user-confirmed 2026-11-02"),
                         ("layers", sys.argv[1], "stage 5 L* on val (paper: 2)"),
                         ("loss", sys.argv[2], "stage 5 loss* on val (paper: mse); early stopping stays on val MSE")]:
    s, n = re.subn(rf"^{key}:.*$", f"{key}: {value}  # {note}", s, flags=re.M)
    assert n == 1, key
p.write_text(s)
EOF

for m in CLATT CNN-BiLSTM CNN-LSTM LSTM CNN; do
  for g in 1 2; do
    say "tune $m group $g"
    python -m src.tune --model "$m" --group "$g" > "results/logs/tune_${m}_g${g}.log" 2>&1
  done
done
python -m src.tune --report >> "$LOG" 2>&1
python -c "from scripts.leakage_checks import chon_sieu_tham_so as c; print(c())" >> "$LOG" 2>&1
say "stage 6 done"
