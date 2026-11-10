"""End-to-end Alpamayo 2 Edge inference on one clip: sliding windows, minADE + per-stage latency.

    python scripts/infer.py                               # eager, the released model
    python scripts/infer.py --mode tuned                  # torch.compile + CUDA graphs (needs flash-attn)
    python scripts/infer.py --clip-id <uuid> --num-steps 20 --warmup-steps 2
    python scripts/infer.py --num-traj-samples 1          # one candidate count only (default: 1 and 6)

Every candidate count in --num-traj-samples is run over the same windows and written to
results/<model>/<mode>_<clip8>_n<k>.json (metrics, settings, every window's CoT) with
_reference.npy (predicted xy per window) and _gt.npy (ground truth). At the end
scripts/plot_results.py draws per-stage latency, per-window latency and minADE / ADE of
all counts into results/<model>/figures/<mode>_<clip8>.png (+ .csv).
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import transformers
from alpamayo2_edge import helper
from alpamayo2_edge.checkpoint import load_model
from alpamayo2_edge.helper import prepare_model_inputs, to_device

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
# NVIDIA's CLI defaults (alpamayo2_edge.inference), so eager runs reproduce its outputs.
SETTINGS = dict(seed=43, top_p=0.98, top_k=None, temperature=0.6,
                max_generation_length=256, diffusion_steps=10)
DEFAULT_CLIP = "87147a1b-3eef-4c25-94d2-ec7718a49a7a"  # flashdrive's default clip


def create_inputs(model: torch.nn.Module, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Every window's prepared inputs (CPU) and ground truth, built before the timed loop."""
    import physical_ai_av

    processor = helper.get_processor(model.tokenizer, model.config)
    helper.get_processor = lambda tokenizer, config: processor  # built once, not per window
    factor = processor.image_processor.patch_size * processor.image_processor.merge_size
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    windows = []
    for i in range(args.num_steps):
        t0 = args.t0_us + i * args.time_step_us
        sample = load_window(args.clip_id, t0, args.nav_text, avdi, args.cache_dir,
                             resize_to_model=not args.raw, factor=factor)
        prepared = prepare_model_inputs(sample, model.config, model.tokenizer)
        logger.info("window %d/%d t0=%d us nav=%r prompt=%d tokens", i, args.num_steps, t0,
                    sample["nav_text"][0], prepared["tokenized_data"]["input_ids"].shape[1])
        windows.append({"inputs": prepared, "t0_us": t0, "nav_text": sample["nav_text"][0],
                        "gt_xyz": sample["ego_future_xyz"]})
    return windows


@torch.inference_mode()
def run_window(model: torch.nn.Module, window: dict, args: argparse.Namespace,
               num_traj_samples: int) -> tuple:
    """One rollout, seeded like NVIDIA's CLI. Returns (minADE, ADE, pred xy, CoT, #tokens)."""
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
    tokens = int((extra["generated_ids"][best] >= 0).sum())
    pred = xyz.detach().float().cpu().numpy()[0].reshape(-1, xyz.shape[-2], 3)[..., :2]
    return best_ade, mean_ade, pred, str(cots[best]), tokens


def run_benchmark(model: torch.nn.Module, windows: list[dict], timer: StageTimer,
                  args: argparse.Namespace, out_dir: str, run_name: str,
                  num_traj_samples: int, accel: Any = None) -> dict:
    """Warm up, time every window with ``num_traj_samples`` candidates, save and return the record."""
    logger.info("Warmup (%d windows, %d candidates)", args.warmup_steps, num_traj_samples)
    for window in windows[: args.warmup_steps]:
        run_window(model, window, args, num_traj_samples)
    logger.info("Warmup completed")

    min_ades, ades, latencies, stages, traj, gts, cots, tokens = [], [], [], [], [], [], [], []
    for idx, window in enumerate(windows[args.warmup_steps :], args.warmup_steps + 1):
        timer.start()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ade, mean_ade, pred, cot, n_tokens = run_window(model, window, args, num_traj_samples)
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
        "mode": args.mode, "attn": args.attn, "model": args.model_path, "clip": args.clip_id,
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
    with open(os.path.join(out_dir, f"{run_name}.json"), "w") as fh:
        json.dump(record, fh, indent=2)
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
    parser.add_argument("--clip-id", default=DEFAULT_CLIP, help="PhysicalAI-AV clip to evaluate")
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
    parser.add_argument("--num-traj-samples", type=int, nargs="+", default=[1, 6],
                        help="candidates per set, each with its own CoT; every count listed is "
                             "run over the same windows (minADE = best candidate, ADE = mean)")
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
    parser.add_argument("--raw", action="store_true", help="keep 1080p frames (and cache them)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None,
                        help="default results/<model basename>")
    parser.add_argument("--run-name", default=None, help="default <mode>[_fa2]")
    parser.add_argument("--no-plot", action="store_true", help="skip the figure at the end")
    args = parser.parse_args()
    if args.no_cache:
        args.cache_dir = None
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

    logger.info("Loading inputs...")
    windows = create_inputs(model, args)

    run_name = args.run_name or (args.mode + ("_fa2" if args.attn == "flash_attention_2" else ""))
    run_name = f"{run_name}_{args.clip_id[:8]}"
    out_dir = args.out_dir or os.path.join(
        RESULTS_DIR, os.path.basename(args.model_path.rstrip("/")) or "model")
    records = [run_benchmark(model, windows, timer, args, out_dir, f"{run_name}_n{k}", k, accel)
               for k in args.num_traj_samples]
    if not args.no_plot:
        fig = plot_runs(records, os.path.join(out_dir, "figures", f"{run_name}.png"))
        logger.info("Saved %s and its .csv", fig)


if __name__ == "__main__":
    main()
