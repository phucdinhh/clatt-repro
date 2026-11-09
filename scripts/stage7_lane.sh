#!/usr/bin/env bash
# Stage 7.2-7.3 and 8.2 test runs, split into 3 lanes that can run side by side (one terminal each):
#   lane 1: main_A (4 baselines seed 0, 5 deep models x 12 pairs x SEEDS_A), then ablations A1, A2
#   lane 2: main_B group 1 (--tuned, SEEDS_B), then, once main_A is done, ablations A3, A4
#   lane 3: main_B group 2 (--tuned, SEEDS_B)
# Every run uses --final, which refuses to start unless configs match configs/FROZEN.sha256.
# Resumable: finished runs (metrics.json with the same config) are skipped. Each process keeps num_threads = 4.
# Usage: scripts/stage7_lane.sh 1|2|3     env: PY, SEEDS_A (default "0 1 2 3 4"), SEEDS_B (default "0 1 2")
set -euo pipefail
cd "$(dirname "$0")/.."
LANE=${1:?usage: scripts/stage7_lane.sh 1|2|3}
PY=${PY:-.venv/bin/python}
read -r -a SA <<< "${SEEDS_A:-0 1 2 3 4}"
read -r -a SB <<< "${SEEDS_B:-0 1 2}"
DEEP=(CLATT CNN-BiLSTM CNN-LSTM LSTM CNN)
LOG=results/logs/stage7_lane$LANE.log
DONE_A=results/logs/main_A.done
mkdir -p results/logs
say() { echo "$(date '+%F %T') lane $LANE: $*" | tee -a "$LOG"; }
run() { say "start src.run $*"; "$PY" -m src.run "$@"; say "done  src.run $*"; }
command -v caffeinate > /dev/null && { caffeinate -i -w $$ & }  # keep the Mac awake while this lane runs

case $LANE in
  1)
    run --exp main_A --models Persistence MA5 "AR(1)" Ridge --group all --seeds 0 --final
    run --exp main_A --models "${DEEP[@]}" --group all --seeds "${SA[@]}" --final
    date '+%F %T' > "$DONE_A"
    for v in A1 A2; do run --exp "abl_$v" --models CLATT --group all --seeds "${SA[@]}" --final --ablation "$v"; done
    ;;
  2)
    run --exp main_B --models "${DEEP[@]}" --group 1 --seeds "${SB[@]}" --final --tuned
    until [ -f "$DONE_A" ]; do say "waiting for main_A before ablations"; sleep 300; done
    for v in A3 A4; do run --exp "abl_$v" --models CLATT --group all --seeds "${SA[@]}" --final --ablation "$v"; done
    ;;
  3)
    run --exp main_B --models "${DEEP[@]}" --group 2 --seeds "${SB[@]}" --final --tuned
    ;;
  *) echo "lane must be 1, 2 or 3" >&2; exit 2 ;;
esac
say "lane finished"
