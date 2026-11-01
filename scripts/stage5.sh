#!/usr/bin/env bash
# Stage 5 sweeps (val only). Usage:
#   scripts/stage5.sh T 20 40 60 68 72 90     -> exp sweep_T/T<T>
#   scripts/stage5.sh L <T*> 1 3 4            -> exp sweep_L/L<L>   (L=2 is reused from sweep_T/T<T*>)
#   scripts/stage5.sh loss <T*> <L*> mae      -> exp sweep_loss/<loss> (mse is reused)
set -euo pipefail
cd "$(dirname "$0")/.."
run() { python -m src.run --models CLATT --group all --seeds 0 1 2 --config-override fixed_split_T=90 "$@"; }
case "$1" in
  T)    shift; for t in "$@"; do run --exp "sweep_T/T$t" --config-override "T=$t"; done ;;
  L)    T=$2; shift 2; for l in "$@"; do run --exp "sweep_L/L$l" --config-override "T=$T" --config-override "layers=$l"; done ;;
  loss) T=$2; L=$3; shift 3; for f in "$@"; do run --exp "sweep_loss/$f" --config-override "T=$T" \
          --config-override "layers=$L" --config-override "loss=$f"; done ;;
esac
