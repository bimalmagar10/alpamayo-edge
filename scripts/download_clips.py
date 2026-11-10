"""Cache the evaluation windows: N clips x 117 sliding windows at 10 Hz, one .npz each.

Frames are stored at the model's front-camera input size (320 x 576 for 1080p, 11x less
disk than raw; see infer_edge.data.resize_frames for the side cameras). Restartable:
existing windows are skipped. Runs without a GPU.

    python scripts/download_clips.py --num-clips 1   # check the download rate first
    python scripts/download_clips.py                 # 100 clips x 117 windows
    python scripts/download_clips.py --shard 0 --num-shards 4   # 4 copies in parallel
"""

import argparse
import os
import time

import pandas as pd
import physical_ai_av

from infer_edge.data import MIN_T0_US, load_window, token_px, window_path

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--metadb", default=os.environ.get("EDGE_METADB"),
                help="NVIDIA's eval metadb parquet (column 'key' = clip id)")
ap.add_argument("--clip-id", action="append", default=None,
                help="clip(s) instead of the metadb's first --num-clips")
ap.add_argument("--cache-dir", default=os.environ.get("EDGE_CLIPS"), required=False)
ap.add_argument("--model-path", default=os.environ.get("EDGE_MODEL"),
                help="checkpoint directory, to read the processor's patch/merge size")
ap.add_argument("--num-clips", type=int, default=100)
ap.add_argument("--num-steps", type=int, default=117)
ap.add_argument("--t0-us", type=int, default=MIN_T0_US)
ap.add_argument("--time-step-us", type=int, default=100_000)
ap.add_argument("--raw", action="store_true", help="store 1080p frames (11x the disk)")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
args = ap.parse_args()
if not args.cache_dir:
    ap.error("--cache-dir or $EDGE_CLIPS is required")

if args.clip_id:
    clips = args.clip_id
else:
    if not args.metadb:
        ap.error("--metadb or $EDGE_METADB is required (setup.sh fetches it)")
    clips = pd.read_parquet(args.metadb)["key"].astype(str).tolist()[: args.num_clips]
clips = clips[args.shard :: args.num_shards]
factor = token_px(args.model_path)
avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
total = len(clips) * args.num_steps
print(f"shard {args.shard}/{args.num_shards}: {len(clips)} clips x {args.num_steps} windows "
      f"= {total} -> {args.cache_dir} (token {factor} px)")

done = skipped = failed = 0
t_start = time.time()
for ci, clip in enumerate(clips, 1):
    for i in range(args.num_steps):
        t0 = args.t0_us + i * args.time_step_us
        out = window_path(args.cache_dir, clip, t0)
        if os.path.exists(out) and os.path.getsize(out):
            skipped += 1
            continue
        try:
            load_window(clip, t0, "placeholder", avdi, args.cache_dir,
                        resize_to_model=not args.raw, factor=factor)
        except Exception as exc:  # one bad window must not stop the run
            failed += 1
            print(f"  FAIL {clip[:12]} t0={t0}  {type(exc).__name__}: {str(exc)[:100]}",
                  flush=True)
            continue
        done += 1
        if done in (1, 2, 5, 10):
            print(f"  window {done}: {(time.time() - t_start) / done:.1f} s each so far",
                  flush=True)
    gb = sum(os.path.getsize(os.path.join(args.cache_dir, f))
             for f in os.listdir(args.cache_dir) if f.endswith(".npz")) / 1e9
    rate = (time.time() - t_start) / max(done, 1)
    print(f"[{ci}/{len(clips)}] saved {done} skipped {skipped} failed {failed}  {gb:.1f} GB  "
          f"{rate:.1f} s/window  ~{rate * (total - done - skipped) / 3600:.1f} h left",
          flush=True)

print(f"\nsaved {done}  had {skipped}  failed {failed}  in {(time.time() - t_start) / 3600:.1f} h")
