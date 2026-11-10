#!/usr/bin/env bash
# One-time H100 setup: uv environment, the Alpamayo 2 Edge checkpoint, the eval clip list,
# and a stack check. Re-runnable; every step skips what is already there.
#
#   bash setup.sh
#
# Needs: a Linux x86_64 host with an NVIDIA GPU driver, internet, and a Hugging Face login
# (`hf auth login` or HF_TOKEN) with access to nvidia/Alpamayo2-Edge and
# nvidia/PhysicalAI-Autonomous-Vehicles (accept both on huggingface.co first).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

echo "== paths =="
edge_check_paths || { echo; echo "Fix the paths above (or export them) and re-run." >&2; exit 1; }
mkdir -p "$EDGE_ROOT"/{logs,results}
echo

# --- 1. environment: Python 3.12, torch 2.8, transformers 5.17, flash-attn wheel ----------
echo "== environment =="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"
uv sync --python 3.12            # creates $UV_PROJECT_ENVIRONMENT, installs this package
# shellcheck disable=SC1091
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
echo "python: $(which python)"
echo

# --- 2. Hugging Face access ---------------------------------------------------------------
echo "== hugging face =="
if ! hf auth whoami >/dev/null 2>&1; then
    echo "Not logged in. Run 'hf auth login' (or export HF_TOKEN) and re-run." >&2
    exit 1
fi
hf auth whoami | head -1
echo

# --- 3. the checkpoint, at the revision NVIDIA's README pins (about 5.3 GB) ---------------
echo "== checkpoint =="
if [ -f "$EDGE_MODEL/config.json" ] && ls "$EDGE_MODEL"/*.safetensors >/dev/null 2>&1; then
    echo "have $EDGE_MODEL"
else
    hf download nvidia/Alpamayo2-Edge --revision "$EDGE_REVISION" --local-dir "$EDGE_MODEL"
fi
echo

# --- 4. NVIDIA's eval clip list (644 clips, ~73 kB), the same file flashdrive uses ---------
echo "== eval metadb =="
if [ -s "$EDGE_METADB" ]; then
    echo "have $EDGE_METADB"
else
    mkdir -p "$(dirname "$EDGE_METADB")"
    NAME="$(basename "$EDGE_METADB")"
    curl -fSL --retry 3 \
        "https://raw.githubusercontent.com/NVlabs/alpamayo-recipes/main/recipes/alpamayo1_5_quant/$NAME" \
        -o "$EDGE_METADB.part"
    mv "$EDGE_METADB.part" "$EDGE_METADB"
fi
echo

# --- 5. verify the stack, the checkpoint and the dataset reach -----------------------------
echo "== check =="
python - <<'PY'
import importlib.util, os, socket, sys
import torch, transformers
from alpamayo2_edge.checkpoint import validate_checkpoint
from pathlib import Path

print("python      :", sys.executable)
print("torch       :", torch.__version__, "cuda", torch.version.cuda)
print("transformers:", transformers.__version__)
print("cuda        :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      :", torch.cuda.get_device_name(0), "sm_%d%d" % torch.cuda.get_device_capability())
    print("bf16        :", torch.cuda.is_bf16_supported())
fa2 = importlib.util.find_spec("flash_attn") is not None
fa3 = importlib.util.find_spec("flash_attn_interface") is not None
print("flash_attn  :", "yes" if fa2 else "NO  (--mode compile/tuned unavailable)", "| fa3:", fa3)
validate_checkpoint(Path(os.environ["EDGE_MODEL"]))
print("checkpoint  :", os.environ["EDGE_MODEL"], "complete")
try:
    socket.create_connection(("huggingface.co", 443), timeout=10).close()
    print("dataset     : huggingface.co reachable (clips stream from there)")
except OSError:
    print("dataset     : NO route to huggingface.co; run on a node with internet or use a "
          "filled $EDGE_CLIPS cache")
print("\nOK")
PY

cat <<MSG

Ready. Each session:
  source $HERE/env.sh && source \$UV_PROJECT_ENVIRONMENT/bin/activate

Then:
  python scripts/infer.py --num-steps 10 --warmup-steps 2     # smoke test, eager
  python scripts/infer.py --mode tuned                         # accelerated, one clip
  python scripts/download_clips.py --num-clips 1               # cache windows (optional)
  bash scripts/run_clips.sh                                    # many clips, then summarize
MSG
