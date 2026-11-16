#!/usr/bin/env bash
# Run infer.py over the eval set, one process per mode: the model is loaded and compiled once,
# clips whose results exist are skipped, failed clips are logged in results/<model>/sweep.log.
#
#   bash scripts/run_clips.sh                                  # all 644 clips, --mode tuned, CoT test full+empty
#   CONDITIONS="" bash scripts/run_clips.sh                     # the plain benchmark (1 and 6 candidates) instead
#   SHARD=0 NSHARDS=4 bash scripts/run_clips.sh                # node 1 of 4: clips 1, 5, 9, ...
#   SHARD=1 NSHARDS=4 bash scripts/run_clips.sh                # node 2 of 4
#   NUM_CLIPS=10 MODES="eager tuned" bash scripts/run_clips.sh # a small comparison
#   CLIP_LIST=missing.txt bash scripts/run_clips.sh            # only these ids (scripts/missing_clips.py)
#   bash scripts/run_clips.sh --num-steps 20                   # extra args go to infer.py
#
# Logs: $EDGE_ROOT/logs/<mode>_shard<k>.log. Then: python scripts/summarize.py
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
NUM_CLIPS="${NUM_CLIPS:-644}"
MODES="${MODES:-tuned}"
CONDITIONS="${CONDITIONS-full empty}"   # the CoT test; CONDITIONS="" runs the plain benchmark
SHARD="${SHARD:-0}"
NSHARDS="${NSHARDS:-1}"
mkdir -p "$EDGE_ROOT/logs"

if [[ -n "${CLIP_LIST:-}" ]]; then
  select=(--clip-list "$CLIP_LIST")
else
  select=(--metadb "$EDGE_METADB" --num-clips "$NUM_CLIPS")
fi

cot=()
[[ -n "$CONDITIONS" ]] && cot=(--cot-condition $CONDITIONS)
for mode in $MODES; do
  log="$EDGE_ROOT/logs/${mode}${CONDITIONS:+_cot}_shard${SHARD}.log"
  echo "== mode $mode, shard $SHARD/$NSHARDS, ${CLIP_LIST:-$NUM_CLIPS clips}, ${CONDITIONS:-plain benchmark} -> $log"
  python "$HERE/scripts/infer.py" --mode "$mode" "${select[@]}" "${cot[@]}" \
    --shard "$SHARD" --nshards "$NSHARDS" "$@" 2>&1 | tee -a "$log"
done
if [[ "$NSHARDS" == 1 ]]; then python "$HERE/scripts/summarize.py"; fi
