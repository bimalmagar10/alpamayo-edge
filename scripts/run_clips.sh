#!/usr/bin/env bash
# Run infer.py over the first N eval clips for each mode, one log per run.
#
#   bash scripts/run_clips.sh                 # 10 clips, modes eager and tuned
#   NUM_CLIPS=100 MODES="eager compile tuned" bash scripts/run_clips.sh
#   bash scripts/run_clips.sh --num-steps 20  # extra args go to infer.py
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
NUM_CLIPS="${NUM_CLIPS:-10}"
MODES="${MODES:-eager tuned}"
mkdir -p "$EDGE_ROOT/logs"

mapfile -t CLIPS < <(python - <<PY
import pandas as pd
print("\n".join(pd.read_parquet("$EDGE_METADB")["key"].astype(str).tolist()[:$NUM_CLIPS]))
PY
)
echo "${#CLIPS[@]} clips, modes: $MODES"
for mode in $MODES; do
  for clip in "${CLIPS[@]}"; do
    log="$EDGE_ROOT/logs/${mode}_${clip:0:8}.log"
    if grep -q "Saved results" "$log" 2>/dev/null; then echo "have $log"; continue; fi
    echo "== $mode $clip -> $log"
    python "$HERE/scripts/infer.py" --mode "$mode" --clip-id "$clip" "$@" 2>&1 | tee "$log"
  done
done
python "$HERE/scripts/summarize.py"
