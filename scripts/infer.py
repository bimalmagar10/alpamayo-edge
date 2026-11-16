"""End-to-end Alpamayo 2 Edge inference: sliding windows per clip, minADE + per-stage latency.

    python scripts/infer.py                               # eager, the released model, one clip
    python scripts/infer.py --mode tuned                  # torch.compile + CUDA graphs (needs flash-attn)
    python scripts/infer.py --clip-id <uuid> --num-steps 20 --warmup-steps 2
    python scripts/infer.py --num-traj-samples 1          # one candidate count only (default: 1 and 6)
    python scripts/infer.py --mode tuned --num-clips 644 --shard 0 --nshards 4   # the eval set, node 1 of 4
    python scripts/infer.py --mode tuned --clip-list missing.txt                 # scripts/missing_clips.py
    python scripts/infer.py --mode tuned --cot-condition full empty --num-clips 644 --shard 0 --nshards 4

--cot-condition is the CoT test of the flashdrive project: "full" lets the model write its own
chain of causation, "empty" forces <|traj_future_start|> as the first token so the action
expert runs on the prompt alone (adaptive reasoning cut to zero). Both run in one process on
the same windows with the same seeds, one candidate by default, and write
results/<model>/cot_test/<condition>/<clip>/<mode>_<clip8>{.json,_cot.json,_reference.npy,_gt.npy}
plus figs/, the layout flashdrive's plot_cot_summary.py / plot_cot_reasoning.py read.

One process runs every clip it is given (--clip-id ..., --clip-list, or the first --num-clips of
$EDGE_METADB split by --shard/--nshards): the model is loaded and compiled once, the next clip's
windows are loaded while the GPU runs the current one, clips whose result files exist are
skipped, and a clip that fails is logged in <out-dir>/sweep.log and does not stop the rest.

Every candidate count in --num-traj-samples is run over the same windows and written to
results/<model>/<mode>_<clip8>_n<k>.json (metrics, settings, every window's CoT) with
_reference.npy (predicted xy per window) and _gt.npy (ground truth). At the end
scripts/plot_results.py draws per-stage latency, per-window latency and minADE / ADE of
all counts into results/<model>/figures/<mode>_<clip8>.png (+ .csv).
"""

import argparse
import concurrent.futures
import datetime
import gc
import json
import logging
import os
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import transformers
from alpamayo2_edge import helper
from alpamayo2_edge.checkpoint import load_model
from alpamayo2_edge.helper import prepare_model_inputs, to_device
from alpamayo2_edge.models.alpamayo2_edge import MaskTokenIdsLogitsProcessor
from transformers import LogitsProcessorList

from infer_edge.data import MIN_T0_US, load_window, min_ade
from infer_edge.timing import STAGES, StageTimer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_results import plot_runs  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

RESULTS_DIR = "results"
MODES = ("eager", "compile", "tuned")  # eager: as released; compile/tuned: NVIDIA's --accel
COT_CONDITIONS = ("full", "empty")  # the CoT test: the model's own CoT / no CoT at all
# NVIDIA's CLI defaults (alpamayo2_edge.inference), so eager runs reproduce its outputs.
SETTINGS = dict(seed=43, top_p=0.98, top_k=None, temperature=0.6,
                max_generation_length=256, diffusion_steps=10)
DEFAULT_CLIP = "87147a1b-3eef-4c25-94d2-ec7718a49a7a"  # flashdrive's default clip


def create_inputs(model: torch.nn.Module, args: argparse.Namespace,
                  clip_id: str) -> list[dict[str, Any]]:
    """Every window's prepared inputs (CPU) and ground truth, built before the timed loop."""
    import physical_ai_av

    processor = helper.get_processor(model.tokenizer, model.config)
    helper.get_processor = lambda tokenizer, config: processor  # built once, not per window
    factor = processor.image_processor.patch_size * processor.image_processor.merge_size
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    windows = []
    for i in range(args.num_steps):
        t0 = args.t0_us + i * args.time_step_us
        sample = load_window(clip_id, t0, args.nav_text, avdi, args.cache_dir,
                             resize_to_model=not args.raw, factor=factor,
                             write_cache=not args.cache_read_only)
        prepared = prepare_model_inputs(sample, model.config, model.tokenizer)
        logger.info("%s window %d/%d t0=%d us nav=%r prompt=%d tokens", clip_id[:8], i,
                    args.num_steps, t0, sample["nav_text"][0],
                    prepared["tokenized_data"]["input_ids"].shape[1])
        windows.append({"inputs": prepared, "t0_us": t0, "nav_text": sample["nav_text"][0],
                        "gt_xyz": sample["ego_future_xyz"]})
    return windows


@torch.inference_mode()
def run_window(model: torch.nn.Module, window: dict, args: argparse.Namespace,
               num_traj_samples: int) -> tuple:
    """One rollout, seeded like NVIDIA's CLI.

    Returns (minADE, ADE, pred xy, CoT text, #tokens, CoT token ids) of the best candidate."""
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xyz, _, _, extra = model.sample_trajectories_from_data(
            to_device(window["inputs"], device=args.device),
            num_traj_sets=args.num_traj_sets,
            num_traj_samples=num_traj_samples,
            top_p=SETTINGS["top_p"], top_k=SETTINGS["top_k"], temperature=SETTINGS["temperature"],
            max_generation_length=args.max_new_tokens,
            diffusion_kwargs={"inference_step": args.diffusion_steps, "generator": generator},
            return_extra=True,
        )
    best_ade, best, mean_ade = min_ade(window["gt_xyz"], xyz)
    cots = np.asarray(extra["cot"]).reshape(-1)
    ids = [int(t) for t in extra["generated_ids"][best].tolist() if t >= 0]
    pred = xyz.detach().float().cpu().numpy()[0].reshape(-1, xyz.shape[-2], 3)[..., :2]
    return best_ade, mean_ade, pred, str(cots[best]), len(ids), ids


def vocab_size(model: torch.nn.Module) -> int:
    head = model.get_output_embeddings()
    if head is not None and hasattr(head, "weight"):
        return int(head.weight.shape[0])
    return int(model.config.vlm_config.text_config.vocab_size)


def install_cot_condition(model: torch.nn.Module, condition: str, base_generate: Any) -> None:
    """Make every window decode the CoT of ``condition`` (see COT_CONDITIONS).

    ``base_generate`` is ``model._generate_with_shared_prefill`` as it was after the stage
    timer and the accelerator wrapped it, so both keep working. "empty" adds NVIDIA's own
    mask processor with every token but <|traj_future_start|> blocked: the first decode
    step emits the anchor, the stopping criterion ends generation one step later, and the
    expert reads a cache that holds the prompt plus the anchor, exactly the model's own
    zero-length-reasoning case. The accelerated sampler supports that processor, so tuned
    mode stays on its graphs.
    """
    if condition == "full":
        model._generate_with_shared_prefill = base_generate
        return
    anchor = int(model.config.traj_ids["future_start"])
    only_anchor = MaskTokenIdsLogitsProcessor([i for i in range(vocab_size(model)) if i != anchor])

    def empty_generate(tokenized_data: dict, generation_config: Any, n_samples_total: int,
                       **kwargs: Any) -> Any:
        processors = LogitsProcessorList(list(kwargs.get("logits_processor") or []))
        processors.append(only_anchor)
        return base_generate(tokenized_data, generation_config, n_samples_total,
                             **{**kwargs, "logits_processor": processors})

    model._generate_with_shared_prefill = empty_generate


def run_benchmark(model: torch.nn.Module, windows: list[dict], timer: StageTimer,
                  args: argparse.Namespace, out_dir: str, run_name: str,
                  num_traj_samples: int, accel: Any = None, clip_id: str = "",
                  condition: str | None = None) -> dict:
    """Warm up, time every window with ``num_traj_samples`` candidates, save and return the record.

    With ``condition`` (the CoT test) every window's CoT, warmup included, also goes to
    ``<run_name>_cot.json`` in flashdrive's format."""
    logger.info("Warmup (%d windows, %d candidates)", args.warmup_steps, num_traj_samples)
    cot_ids, cot_text = {}, {}
    for w, window in enumerate(windows[: args.warmup_steps]):
        *_, cot, _, ids = run_window(model, window, args, num_traj_samples)
        cot_ids[w], cot_text[w] = ids, cot
    logger.info("Warmup completed")

    min_ades, ades, latencies, stages, traj, gts, cots, tokens = [], [], [], [], [], [], [], []
    for idx, window in enumerate(windows[args.warmup_steps :], args.warmup_steps + 1):
        timer.start()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ade, mean_ade, pred, cot, n_tokens, ids = run_window(model, window, args,
                                                             num_traj_samples)
        cot_ids[idx - 1], cot_text[idx - 1] = ids, cot
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000.0
        stages.append(timer.stop(wall))
        latencies.append(wall)
        min_ades.append(ade)
        ades.append(mean_ade)
        traj.append(pred)
        gts.append(window["gt_xyz"].numpy()[0, :, :2])
        cots.append(cot)
        tokens.append(n_tokens)
        logger.info("Chain-of-Causation:\n%s", cot)
        logger.info("Step %d/%d completed | MinADE: %.4f | %d tokens | Latency: %.1f ms",
                    idx, len(windows), ade, n_tokens, wall)

    n = len(latencies)
    stage = {k: float(np.mean([s[k] for s in stages])) for k in STAGES}
    logger.info("Average MinADE: %.4f | Average ADE: %.4f (%d candidates)",
                np.mean(min_ades), np.mean(ades), num_traj_samples)
    logger.info("LATENCY_SUMMARY ms: mean=%.1f median=%.1f min=%.1f p90=%.1f n=%d",
                np.mean(latencies), np.median(latencies), np.min(latencies),
                np.percentile(latencies, 90), n)
    for name in STAGES:
        logger.info("STAGE %-8s %7.1f ms/window", name, stage[name])

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{run_name}_reference.npy"), np.stack(traj))
    np.save(os.path.join(out_dir, f"{run_name}_gt.npy"), np.stack(gts))
    record = {
        "mode": args.mode, "attn": args.attn, "model": args.model_path, "clip": clip_id,
        "t0_us": args.t0_us, "time_step_us": args.time_step_us, "n": n,
        "warmup_steps": args.warmup_steps,
        "settings": {**SETTINGS, "seed": args.seed, "max_generation_length": args.max_new_tokens,
                     "diffusion_steps": args.diffusion_steps,
                     "num_traj_sets": args.num_traj_sets,
                     "num_traj_samples": num_traj_samples},
        "nav_text": args.nav_text or "oracle",
        "mean_min_ade": float(np.mean(min_ades)),
        "mean_ade": float(np.mean(ades)),
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "stage_ms": stage,
        "mean_tokens": float(np.mean(tokens)),
        "min_ades": min_ades,
        "ades": ades,
        "latencies_ms": latencies,
        "tokens": tokens,
        "stage_ms_per_window": {k: [s[k] for s in stages] for k in STAGES},
        "nav_texts": [w["nav_text"] for w in windows[args.warmup_steps :]],
        "cots": cots,
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "device": torch.cuda.get_device_name(args.device)},
    }
    if accel is not None:
        record["acceleration"] = accel.describe()
    if condition is not None:
        record["cot_condition"] = condition
    with open(os.path.join(out_dir, f"{run_name}.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    if condition is not None:
        with open(os.path.join(out_dir, f"{run_name}_cot.json"), "w") as fh:
            json.dump({"cot_condition": condition, "step": args.mode, "mode": args.mode,
                       "clip": clip_id, "t0_us": args.t0_us, "time_step_us": args.time_step_us,
                       "ids": cot_ids, "text": cot_text}, fh, indent=2)
    logger.info("Saved %s/%s.json, %s_reference.npy and %s_gt.npy", out_dir, run_name,
                run_name, run_name)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default=os.environ.get("EDGE_MODEL", "nvidia/Alpamayo2-Edge"),
                        help="local checkpoint directory or HF repo id")
    parser.add_argument("--revision", default=os.environ.get("EDGE_REVISION"),
                        help="HF revision when --model-path is a repo id")
    parser.add_argument("--mode", default="eager", choices=MODES,
                        help="eager: the released path; compile / tuned: NVIDIA's --accel "
                             "(torch.compile + CUDA graphs; tuned autotunes the decode step)")
    parser.add_argument("--attn", default="sdpa", choices=("sdpa", "flash_attention_2"),
                        help="attention of the eager parts (the vision tower, and everything "
                             "in --mode eager)")
    parser.add_argument("--clip-id", nargs="+", default=None,
                        help=f"PhysicalAI-AV clip(s) to evaluate (default {DEFAULT_CLIP[:8]}...)")
    parser.add_argument("--clip-list", default=None, help="file with one clip id per line")
    parser.add_argument("--metadb", default=os.environ.get("EDGE_METADB"),
                        help="eval metadb parquet (column 'key'), used with --num-clips")
    parser.add_argument("--num-clips", type=int, default=None,
                        help="the first N clips of --metadb (644 = the whole eval set)")
    parser.add_argument("--shard", type=int, default=0, help="with --nshards: this node's part")
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--force", action="store_true",
                        help="rerun clips whose result files already exist")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="load each clip's windows only after the previous clip finished")
    parser.add_argument("--t0-us", type=int, default=MIN_T0_US,
                        help="first window (us); the model needs 2 s of camera history")
    parser.add_argument("--time-step-us", type=int, default=100_000, help="window stride (us)")
    parser.add_argument("--num-steps", type=int, default=117,
                        help="windows per clip; 117 from 2.0 s ends at 13.6 s like flashdrive's "
                             "120 from 1.7 s, and the last 6.4 s future stays inside a 20 s clip")
    parser.add_argument("--warmup-steps", type=int, default=3,
                        help="initial windows excluded from the metrics (compile/tuned build "
                             "their graphs there)")
    parser.add_argument("--num-traj-sets", type=int, default=1)
    parser.add_argument("--num-traj-samples", type=int, nargs="+", default=None,
                        help="candidates per set, each with its own CoT; every count listed is "
                             "run over the same windows (minADE = best candidate, ADE = mean). "
                             "Default 1 and 6; 1 with --cot-condition")
    parser.add_argument("--cot-condition", nargs="+", default=[], choices=COT_CONDITIONS,
                        help="the CoT test: run these conditions on every clip and write "
                             "results/<model>/cot_test/<condition>/<clip>/ (see the top)")
    parser.add_argument("--nav-text", default=None,
                        help="fixed navigation instruction for every window (default: the "
                             "oracle instruction derived from the ground-truth future)")
    parser.add_argument("--max-new-tokens", type=int, default=SETTINGS["max_generation_length"])
    parser.add_argument("--diffusion-steps", type=int, default=SETTINGS["diffusion_steps"])
    parser.add_argument("--seed", type=int, default=SETTINGS["seed"])
    parser.add_argument("--cache-dir", default=os.environ.get("EDGE_CLIPS"),
                        help="window cache (.npz per window); streams from HF when a window is "
                             "missing and stores it")
    parser.add_argument("--no-cache", action="store_true", help="stream every window, store nothing")
    parser.add_argument("--cache-read-only", action="store_true",
                        help="read cached windows, never write new ones (a 644-clip sweep is "
                             "~900 GB of windows; the default for more than one clip)")
    parser.add_argument("--raw", action="store_true", help="keep 1080p frames (and cache them)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None,
                        help="default results/<model basename>")
    parser.add_argument("--run-name", default=None, help="default <mode>[_fa2]")
    parser.add_argument("--no-plot", action="store_true", help="skip the figure at the end")
    args = parser.parse_args()
    if args.no_cache:
        args.cache_dir = None
    clips = select_clips(args)
    if len(clips) > 1:
        args.cache_read_only = True
    conditions = list(dict.fromkeys(args.cot_condition))
    if args.num_traj_samples is None:
        args.num_traj_samples = [1] if conditions else [1, 6]
    if args.mode != "eager" and not args.warmup_steps:
        raise SystemExit("--mode compile/tuned: use --warmup-steps >= 1, the first window "
                         "compiles and captures the graphs")

    is_local = os.path.isdir(args.model_path)
    model, loading_info = load_model(
        args.model_path, device=args.device, dtype="bfloat16", attn_implementation=args.attn,
        revision=None if is_local else args.revision)
    model.eval()
    logger.info("loaded %s (%s, %s)", args.model_path, args.attn,
                {k: len(v) for k, v in loading_info.items() if isinstance(v, list)})
    accel = None
    if args.mode != "eager":
        from alpamayo2_edge.accel import AccelConfig, enable_acceleration

        accel = enable_acceleration(model, AccelConfig(mode=args.mode))
        if accel.skipped:
            raise SystemExit(f"--mode {args.mode} runs eagerly: {accel.skipped}; install "
                             "flash-attn (see README)")
        logger.info("acceleration: parts=%s kernels=%s", accel.parts, accel.kernels)
    timer = StageTimer()
    timer.install(model)
    base_generate = model._generate_with_shared_prefill  # timed (and accelerated) generation

    run_base = args.run_name or (args.mode + ("_fa2" if args.attn == "flash_attention_2" else ""))
    out_dir = args.out_dir or os.path.join(
        RESULTS_DIR, os.path.basename(args.model_path.rstrip("/")) or "model")
    os.makedirs(out_dir, exist_ok=True)

    cot_root = os.path.join(out_dir, "cot_test")

    def run_names(clip_id: str) -> list[tuple[int, str]]:
        """(candidate count, run name): the CoT test with one count keeps flashdrive's
        <mode>_<clip8> name, everything else gets _n<k>."""
        if conditions and len(args.num_traj_samples) == 1:
            return [(args.num_traj_samples[0], f"{run_base}_{clip_id[:8]}")]
        return [(k, f"{run_base}_{clip_id[:8]}_n{k}") for k in args.num_traj_samples]

    def done(clip_id: str) -> bool:
        if conditions:
            return all(os.path.exists(os.path.join(cot_root, c, clip_id, f"{name}_cot.json"))
                       for c in conditions for _, name in run_names(clip_id))
        return all(os.path.exists(os.path.join(out_dir, f"{name}.json"))
                   for _, name in run_names(clip_id))

    todo = [c for c in clips if args.force or not done(c)]
    logger.info("%d clips, %d to run (shard %d/%d)", len(clips), len(todo), args.shard,
                args.nshards)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending: dict[str, concurrent.futures.Future] = {}  # clip id -> its windows, loading

    def prefetch(clip_id: str) -> None:
        if clip_id not in pending:
            pending[clip_id] = pool.submit(create_inputs, model, args, clip_id)

    if todo:
        prefetch(todo[0])
    n_failed = 0
    for ci, clip_id in enumerate(todo):
        start = time.time()
        status = "ok"
        try:
            logger.info("[%d/%d] loading inputs of %s", ci + 1, len(todo), clip_id)
            windows = pending.pop(clip_id).result()
            if ci + 1 < len(todo):  # the next clip's windows load while this one runs
                prefetch(todo[ci + 1])
                if args.no_prefetch:
                    pending[todo[ci + 1]].result()
            fig_name = f"{run_base}_{clip_id[:8]}.png"
            if conditions:
                for condition in conditions:
                    cdir = os.path.join(cot_root, condition, clip_id)
                    if not args.force and all(os.path.exists(os.path.join(cdir, f"{name}_cot.json"))
                                              for _, name in run_names(clip_id)):
                        logger.info("CoT condition %s of %s is done, skipping", condition, clip_id)
                        continue
                    logger.info("CoT condition: %s", condition)
                    install_cot_condition(model, condition, base_generate)
                    records = [run_benchmark(model, windows, timer, args, cdir, name, k, accel,
                                             clip_id, condition) for k, name in run_names(clip_id)]
                    if not args.no_plot:
                        fig = plot_runs(records, os.path.join(cdir, "figs", fig_name))
                        logger.info("Saved %s and its .csv", fig)
                model._generate_with_shared_prefill = base_generate
            else:
                records = [run_benchmark(model, windows, timer, args, out_dir, name, k, accel,
                                         clip_id) for k, name in run_names(clip_id)]
                if not args.no_plot:
                    fig = plot_runs(records, os.path.join(out_dir, "figures", fig_name))
                    logger.info("Saved %s and its .csv", fig)
            del windows, records
        except Exception:
            status, n_failed = "FAILED", n_failed + 1
            logger.error("clip %s failed:\n%s", clip_id, traceback.format_exc())
            if ci + 1 < len(todo):
                prefetch(todo[ci + 1])
            gc.collect()
            torch.cuda.empty_cache()
        with open(os.path.join(out_dir, "sweep.log"), "a") as fh:
            fh.write(f"{datetime.datetime.now():%F %T}  {status}  {clip_id}  {args.mode}  "
                     f"[{' '.join(conditions) or 'benchmark'}]  shard {args.shard}/{args.nshards}  "
                     f"{(time.time() - start) / 60:.1f} min\n")
    pool.shutdown()
    logger.info("done: %d clips run, %d failed (see %s/sweep.log; scripts/missing_clips.py "
                "lists what is still missing)", len(todo), n_failed, out_dir)


def select_clips(args: argparse.Namespace) -> list[str]:
    """--clip-id, else --clip-list, else the first --num-clips of the metadb, sharded."""
    if args.clip_id:
        clips = list(args.clip_id)
    elif args.clip_list:
        with open(args.clip_list) as fh:
            clips = [line.strip() for line in fh if line.strip()]
    elif args.num_clips:
        if not args.metadb:
            raise SystemExit("--num-clips needs --metadb or $EDGE_METADB")
        import pandas as pd

        clips = pd.read_parquet(args.metadb)["key"].astype(str).tolist()[: args.num_clips]
    else:
        clips = [DEFAULT_CLIP]
    return clips[args.shard :: args.nshards]


if __name__ == "__main__":
    main()
