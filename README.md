# infer-alpamayo-edge

Inference benchmark for [NVIDIA Alpamayo 2 Edge](https://github.com/NVlabs/alpamayo2-edge)
(checkpoint [nvidia/Alpamayo2-Edge](https://huggingface.co/nvidia/Alpamayo2-Edge)) on an H100,
laid out like `rt-alapamyo/h100/flashdrive`: a clip is run as a stream of sliding windows,
and every run reports **minADE** against the ground-truth future and **per-window latency**
split into encode / prefill / decode / action.

```
pyproject.toml            uv project: alpamayo2-edge (pinned commit), torch 2.8, transformers 5.17, flash-attn wheel
env.sh                    cluster paths (EDGE_MODEL, EDGE_ROOT, EDGE_CLIPS, HF_HOME, compile caches)
setup.sh                  one-time: venv, checkpoint download, eval clip list, stack check
infer_edge/data.py        a window: NVIDIA's raw sample + ground-truth future, .npz cache, nav text, minADE
infer_edge/timing.py      CUDA-event stage timers hooked on the model's own calls
scripts/infer.py          the benchmark (one clip, one mode)
scripts/download_clips.py pre-fill the window cache for N clips (no GPU needed)
scripts/run_clips.sh      infer.py over the first N eval clips x modes, then summarize
scripts/summarize.py      one table over results/ (and flashdrive's result files, if wanted)
results/<model>/          <mode>_<clip8>.json, _reference.npy (pred xy), _gt.npy
```

## Setup (H100 node, once)

```bash
# accept the model and dataset terms on huggingface.co, then:
hf auth login            # or export HF_TOKEN
bash setup.sh
```

`setup.sh` installs `uv`, builds `.venv` from the lock, downloads the checkpoint to
`$EDGE_MODEL` at the revision NVIDIA pins, fetches the 644-clip eval list to `$EDGE_METADB`
and prints a stack check (torch/CUDA/bf16/flash-attn/checkpoint/dataset reach). Paths come
from `env.sh`; export any of them before running to override. The flash-attn prebuilt wheel
needs an x86_64 host with glibc >= 2.32; without it `--mode eager` still works.

Each session:

```bash
source env.sh && source $UV_PROJECT_ENVIRONMENT/bin/activate
```

## Run

```bash
python scripts/infer.py --num-steps 10 --warmup-steps 2    # smoke test (eager, streams the clip)
python scripts/infer.py                                     # eager: the released model, 117 windows
python scripts/infer.py --mode compile                      # NVIDIA --accel compile (torch.compile + CUDA graphs)
python scripts/infer.py --mode tuned                        # NVIDIA --accel tuned (also autotunes decode)
python scripts/infer.py --attn flash_attention_2            # eager with FlashAttention 2
python scripts/infer.py --num-traj-samples 6                # 6 CoT + trajectory candidates, minADE = best
python scripts/infer.py --nav-text "Go straight"            # fixed instruction instead of the oracle
bash scripts/run_clips.sh                                   # 10 clips x {eager, tuned}; NUM_CLIPS=100 MODES=... to change
python scripts/summarize.py                                 # table over results/
python scripts/summarize.py --compare ../rt-alapamyo/h100/flashdrive/results/*/*.json
```

Each run logs every window's chain of causation, minADE, token count and latency, then a
`LATENCY_SUMMARY` and `STAGE` lines, and writes `results/<model>/<mode>_<clip8>.json` with
the per-window arrays, the settings, the navigation texts, the CoTs and (accelerated modes)
NVIDIA's graph-build and fallback report.

### Window cache

Streaming a window from Hugging Face decodes 22 frames from six camera videos, so loading a
clip takes minutes. `infer.py` stores every window it streams under `$EDGE_CLIPS`
(`<clip>__<t0>.npz`) and reads it back next time. `scripts/download_clips.py` fills the cache
for many clips up front, shardable over several processes, and `--num-clips 1` first shows the
rate. Frames are stored at the model's front-camera input size (320 x 576 from 1080p, 11x less
disk); the four side cameras are then resized a second time by the processor to their smaller
budget, see `infer_edge.data.resize_frames`. `--raw` keeps 1080p.

## What differs from the flashdrive setup

| | flashdrive (Alpamayo R1 / 1.5) | this project (Alpamayo 2 Edge) |
| --- | --- | --- |
| cameras | 4 x 4 frames at 10 Hz | 6 cameras, 22 frames (front wide + tele: 5 each over 2 s; others: 3 each) |
| ego history | 16 poses, 1.6 s | 20 poses, 1.9 s |
| navigation | none | required text; default is an oracle instruction from the ground-truth future (`Turn right in 30m` / `Go straight`), `--nav-text` fixes it |
| windows | 120 from 1.7 s | 117 from 2.0 s (the model needs 2 s of camera history), same last window at 13.6 s |
| ground truth | from the R1 loader | computed the same way in `infer_edge.data.future_trajectory` (NVIDIA's Edge loader loads no future) |
| optimized path | FlashDrive steps | NVIDIA's own `--accel compile/tuned` (torch.compile + CUDA graphs + FA2) |
| sampling | model defaults | NVIDIA CLI defaults: seed 43, top-p 0.98, T 0.6, 256 tokens, 10 flow steps |

Windows are prepared (tokenized, image patches built) before the timed loop, as in flashdrive;
117 prepared windows take about 8 GB of host RAM.

### Stage timing

`infer_edge/timing.py` records CUDA events around the model's own calls, so it works on the
eager and the accelerated path alike: `encode` is the vision tower (always eager), `prefill` is
the first text-stack call (or the prefill graph) minus the encode nested in it, `decode` is the
rest of `_generate_with_shared_prefill`, `action` is `_sample_actions` (the flow-matching
expert), and `other` is the remaining wall time (token fusing, trajectory conversion, host
syncs). `--mode compile/tuned` compile and capture graphs on the first window; keep
`--warmup-steps >= 1` and the `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` from `env.sh` so
later processes reuse the kernels. A changed prompt length across a 256-token bucket (possible
with the oracle nav text) or a changed candidate count builds new graphs mid-run; the JSON's
`acceleration.builds` lists them.

## Tests NVIDIA ships

```bash
git clone https://github.com/NVlabs/alpamayo2-edge.git /tmp/alpamayo2-edge
cd /tmp/alpamayo2-edge && ALPAMAYO2_EDGE_CHECKPOINT="$EDGE_MODEL" python -m pytest
```
