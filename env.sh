#   source env.sh
# Cluster paths for the H100 side. Override any of them before sourcing.

EDGE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export EDGE_REPO
export EDGE_ROOT="${EDGE_ROOT:-/mnt/SHARED-SCRATCH/bthapama/alpamayo-edge-work}"   # outputs, caches
export EDGE_MODEL="${EDGE_MODEL:-$EDGE_ROOT/Alpamayo2-Edge}"                        # hf download of nvidia/Alpamayo2-Edge
export EDGE_CLIPS="${EDGE_CLIPS:-$EDGE_ROOT/clips}"                                 # cached windows (scripts/download_clips.py)
export EDGE_METADB="${EDGE_METADB:-/mnt/SHARED-SCRATCH/bthapama/alpamayo-work/evaluation/1005_7cam_gold_eval_metadb_public.parquet}"
export EDGE_REVISION="${EDGE_REVISION:-f8951d095de3589b3face1dec3f4394c8ecedb38}"    # immutable HF revision NVIDIA's README pins

# Hugging Face: the dataset streams camera video, so stay online.
export HF_HOME="${HF_HOME:-/mnt/SHARED-SCRATCH/bthapama/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# Keep torch.compile / Triton kernels across processes (--mode compile / tuned).
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$EDGE_ROOT/cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$EDGE_ROOT/cache/triton}"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$EDGE_REPO/.venv}"
export PYTHONUNBUFFERED=1

edge_check_paths() {
    local bad=0 v val
    for v in EDGE_ROOT EDGE_CLIPS HF_HOME TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR; do
        eval "val=\${$v:-}"
        if mkdir -p "$val" 2>/dev/null; then echo "ok       $v = $val"
        else echo "CANNOT CREATE  $v = $val" >&2; bad=1; fi
    done
    if [ -f "$EDGE_MODEL/config.json" ]; then echo "ok       EDGE_MODEL = $EDGE_MODEL"
    else echo "MISSING  EDGE_MODEL = $EDGE_MODEL (setup.sh downloads it)"; fi
    if [ -s "$EDGE_METADB" ]; then echo "ok       EDGE_METADB = $EDGE_METADB"
    else echo "MISSING  EDGE_METADB = $EDGE_METADB (setup.sh fetches it)"; fi
    return $bad
}
