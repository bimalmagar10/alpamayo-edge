"""Plot one clip's runs: per-stage latency, per-window latency, and minADE / ADE per candidate count.

infer.py calls ``plot_runs`` when it finishes. Standalone:

    python scripts/plot_results.py results/Alpamayo2-Edge/tuned_87147a1b_n1.json \\
                                   results/Alpamayo2-Edge/tuned_87147a1b_n6.json
    python scripts/plot_results.py results/Alpamayo2-Edge/tuned_87147a1b_n*.json -o fig.png

Writes <out> (default results/<model>/figures/<mode>_<clip8>.png) and a .csv of the numbers.
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STAGES = ("encode", "prefill", "decode", "action", "other")
# Validated categorical palette (dataviz reference, light surface), fixed order.
STAGE_COLORS = dict(zip(STAGES, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"),
                        strict=True))
RUN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # one per candidate count
ADE_COLORS = {"minADE": "#2a78d6", "ADE": "#eb6834"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _label(run: dict) -> str:
    s = run.get("settings", {})
    n = s.get("num_traj_sets", 1) * s.get("num_traj_samples", 1)
    return f"{n} traj"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_runs(runs: list[dict], out_path: str) -> str:
    """``runs`` are result JSONs of one clip and mode with different candidate counts."""
    runs = sorted(runs, key=lambda r: r.get("settings", {}).get("num_traj_samples", 1)
                  * r.get("settings", {}).get("num_traj_sets", 1))
    labels = [_label(r) for r in runs]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1.6, 1]})
    title = f"{os.path.basename(str(runs[0].get('model', '')).rstrip('/'))}  " \
            f"{runs[0].get('mode', '')}  clip {str(runs[0].get('clip', ''))[:8]}  " \
            f"({runs[0]['n']} windows, {runs[0].get('runtime', {}).get('device', '')})"
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")

    # 1. per-stage latency: one stacked bar per candidate count, 2 px gaps, labelled segments
    ax = axes[0]
    _style(ax)
    x = np.arange(len(runs))
    bottom = np.zeros(len(runs))
    for stage in STAGES:
        vals = np.array([r["stage_ms"].get(stage, 0.0) for r in runs])
        ax.bar(x, vals, 0.55, bottom=bottom, color=STAGE_COLORS[stage], label=stage,
               edgecolor=SURFACE, linewidth=1.5)
        for xi, v, b in zip(x, vals, bottom, strict=True):
            if v >= 0.06 * max(r["mean_ms"] for r in runs):
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if stage != "action" else INK)
        bottom += vals
    for xi, r in zip(x, runs, strict=True):
        ax.text(xi, r["mean_ms"] * 1.01, f"{r['mean_ms']:.0f} ms", ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12 * max(r["mean_ms"] for r in runs))
    ax.set_ylabel("mean ms / window", color=INK_2)
    ax.set_title("per-stage latency", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left",
              bbox_to_anchor=(1.0, 1.0))

    # 2. per-window total latency
    ax = axes[1]
    _style(ax)
    for r, c in zip(runs, RUN_COLORS, strict=False):
        lat = np.asarray(r["latencies_ms"])
        w = np.arange(r.get("warmup_steps", 0) + 1, r.get("warmup_steps", 0) + 1 + len(lat))
        ax.plot(w, lat, color=c, linewidth=2, label=f"{_label(r)}  median {np.median(lat):.0f} ms")
    ax.set_xlabel("window", color=INK_2)
    ax.set_ylabel("ms", color=INK_2)
    ax.set_title("latency per window", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)

    # 3. minADE and ADE per candidate count
    ax = axes[2]
    _style(ax)
    width = 0.36
    for i, (name, key) in enumerate((("minADE", "mean_min_ade"), ("ADE", "mean_ade"))):
        vals = [r.get(key, r["mean_min_ade"]) for r in runs]
        bars = ax.bar(x + (i - 0.5) * (width + 0.02), vals, width, color=ADE_COLORS[name],
                      label=name, edgecolor=SURFACE, linewidth=1.5)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.25 * max(r.get("mean_ade", r["mean_min_ade"]) for r in runs))
    ax.set_ylabel("metres (mean over windows)", color=INK_2)
    ax.set_title("minADE (best candidate) / ADE (mean)", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    with open(os.path.splitext(out_path)[0] + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "windows", "min_ade", "ade", "mean_ms", "median_ms", "p90_ms",
                    "tokens", *STAGES])
        for r, label in zip(runs, labels, strict=True):
            w.writerow([label, r["n"], f"{r['mean_min_ade']:.4f}",
                        f"{r.get('mean_ade', r['mean_min_ade']):.4f}", f"{r['mean_ms']:.1f}",
                        f"{r['median_ms']:.1f}", f"{r['p90_ms']:.1f}",
                        f"{r.get('mean_tokens', float('nan')):.1f}",
                        *(f"{r['stage_ms'].get(s, 0.0):.1f}" for s in STAGES)])
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="result JSONs of one clip and mode")
    ap.add_argument("-o", "--out", default=None, help="output .png")
    args = ap.parse_args()
    runs = []
    for path in args.files:
        with open(path) as fh:
            runs.append(json.load(fh))
    if args.out is None:
        first = args.files[0]
        stem = os.path.basename(first).rsplit("_n", 1)[0]
        args.out = os.path.join(os.path.dirname(first), "figures", stem + ".png")
    print("wrote", plot_runs(runs, args.out), "and its .csv")


if __name__ == "__main__":
    main()
