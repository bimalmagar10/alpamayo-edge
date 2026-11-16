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
scripts/plot_results.py   figure per run: per-stage latency, latency per window, minADE / ADE (infer.py calls it)
scripts/run_clips.sh      infer.py over the eval set (644 clips), shardable over nodes
scripts/missing_clips.py  which eval clips have no finished result yet
scripts/summarize.py      one table over results/ (and flashdrive's result files, if wanted)
scripts/dissect_checkpoint.py  layers / widths / heads / params of a checkpoint from its config and tensor shapes
scripts/plot_architecture.py   NeurIPS-style block diagram of Alpamayo 1.5 vs Alpamayo 2 Edge (pdf + png)
results/<model>/          <mode>_<clip8>_n<k>.json, _reference.npy (pred xy), _gt.npy; figures/<mode>_<clip8>.png + .csv
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
python scripts/infer.py --num-traj-samples 1                # one candidate count only (default runs 1 and 6)
python scripts/plot_results.py results/Alpamayo2-Edge/tuned_87147a1b_n*.json   # redraw a figure
python scripts/infer.py --nav-text "Go straight"            # fixed instruction instead of the oracle
SHARD=0 NSHARDS=4 bash scripts/run_clips.sh                 # the 644 eval clips, --mode tuned, node 1 of 4 (see below)
python scripts/missing_clips.py --mode tuned                # which clips still have no result
python scripts/summarize.py                                 # table over results/
python scripts/summarize.py --compare ../rt-alapamyo/h100/flashdrive/results/*/*.json
```

By default a run goes over the clip twice on the same prepared windows: once with one
candidate and once with six (each candidate samples its own chain of causation and
trajectory). Every window logs its chain of causation, minADE, token count and latency,
then a `LATENCY_SUMMARY` and `STAGE` lines, and each count writes
`results/<model>/<mode>_<clip8>_n<k>.json` with the per-window arrays, the settings, the
navigation texts, the CoTs and (accelerated modes) NVIDIA's graph-build and fallback report.
minADE is the best candidate's ADE and ADE the mean candidate's; with one candidate they
are equal. At the end `scripts/plot_results.py` draws
`results/<model>/figures/<mode>_<clip8>.png` (per-stage latency per count, latency per
window, minADE / ADE per count) and the same numbers as a `.csv`. `--no-plot` skips it.

### The whole eval set

`infer.py` takes many clips in one process (`--clip-id a b ...`, `--clip-list file`, or the
first `--num-clips` of `$EDGE_METADB` split with `--shard k --nshards n`): the model is
loaded and compiled once, the next clip's windows stream in a background thread while the
GPU runs the current clip, clips whose result files already exist are skipped, and a clip
that fails is logged in `results/<model>/sweep.log` without stopping the rest.
`scripts/run_clips.sh` wraps that per node:

```bash
SHARD=0 NSHARDS=4 bash scripts/run_clips.sh     # node 1 ... SHARD=3 on node 4; logs in $EDGE_ROOT/logs
python scripts/missing_clips.py --mode tuned --out missing.txt   # after all shards: anything left?
CLIP_LIST=missing.txt bash scripts/run_clips.sh                  # rerun just those
python scripts/summarize.py                                      # mean over clips per mode / count
```

With more than one clip the window cache is read but never written (`--cache-read-only`):
644 clips x 117 windows is about 900 GB of windows. Per clip expect the streaming time of 117
windows (hidden behind the previous clip's inference once the pipeline is full) plus two passes
of 117 windows on the GPU, one per candidate count.

### CoT test: full vs empty reasoning

The flashdrive CoT test, on Edge. `--cot-condition full empty` runs every clip twice on the
same windows and seeds: `full` lets the model write its own chain of causation, `empty`
forces `<|traj_future_start|>` as the first token (NVIDIA's own mask processor, so tuned
mode stays on its graphs) and the action expert plans from the prompt alone, the model's
zero-length adaptive-reasoning case. One candidate by default. Outputs follow flashdrive's
layout so its `plot_cot_summary.py` / `plot_cot_reasoning.py` read them unchanged:

```
results/Alpamayo2-Edge/cot_test/<condition>/<clip>/tuned_<clip8>.json        metrics + per-window arrays
results/Alpamayo2-Edge/cot_test/<condition>/<clip>/tuned_<clip8>_cot.json    every window's CoT ids + text
results/Alpamayo2-Edge/cot_test/<condition>/<clip>/tuned_<clip8>_{reference,gt}.npy
results/Alpamayo2-Edge/cot_test/<condition>/<clip>/figs/tuned_<clip8>.png
```

`run_clips.sh` runs the CoT test by default (`CONDITIONS="full empty"`); `CONDITIONS=""`
gives the plain benchmark with 1 and 6 candidates. `summarize.py` shows a `cot` column and
`missing_clips.py --condition empty` checks one condition.

**`<NOMINAL>`.** Edge's adaptive reasoning: when the model decides the scene needs no
explanation it writes the single token `<NOMINAL>` and goes straight to the trajectory, which
is the model's own empty condition. A window's CoT is therefore either `<NOMINAL>` or a
chain of causation. Each result records `n_nominal`, a per-window `nominal` flag and minADE /
latency split by it, and `summarize.py` shows them as `nominal` (fraction of windows),
`ade_nom` / `ade_rsn` and `ms_nom` / `ms_rsn`: full vs empty is only informative on the
reasoned windows. Every window is seeded with `--seed` + window index (same seed for both
conditions of a window, different draws across windows); a fixed seed per window would turn
the reason-or-not choice into a constant threshold and give long runs of one or the other.

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

## Checkpoint anatomy

`scripts/dissect_checkpoint.py` prints every transformer declared in a checkpoint's
`config.json` (layers, hidden size, heads, KV heads, head dim, FFN, vocab) and, independently,
the same numbers derived from the safetensors tensor shapes, with parameter counts per
component. It reads only the config and shard headers, so it works on a Hub repo without
downloading weights, or on `$EDGE_MODEL`. The reports for both released models are in
`docs/checkpoint-dissection/`.

```bash
python scripts/dissect_checkpoint.py                                  # $EDGE_MODEL
python scripts/dissect_checkpoint.py --model nvidia/Alpamayo-1.5-10B --json a15.json
```

## Tests NVIDIA ships

```bash
git clone https://github.com/NVlabs/alpamayo2-edge.git /tmp/alpamayo2-edge
cd /tmp/alpamayo2-edge && ALPAMAYO2_EDGE_CHECKPOINT="$EDGE_MODEL" python -m pytest
```
