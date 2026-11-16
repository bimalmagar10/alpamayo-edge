"""One table over every run in results/: minADE and latency per mode, averaged over clips.

    python scripts/summarize.py                                  # results/*/*.json
    python scripts/summarize.py --compare ../flashdrive/results/*.json   # other runs too

flashdrive's result files share the keys (its "step" is this project's "mode"), so they
line up in the same table. The nominal / ade_nom / ade_rsn / ms_nom / ms_rsn columns split an
Edge run by the model's adaptive reasoning: windows answered <NOMINAL> (no chain of
causation) against windows with a written one; blank for runs without that field.
"""

import argparse
import collections
import glob
import json
import os

import numpy as np

STAGES = ("encode", "prefill", "decode", "action", "other")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", default=None, help="result JSONs (default results/*/*.json)")
    ap.add_argument("--compare", nargs="*", default=[], help="result JSONs of other projects")
    ap.add_argument("--csv", default=None, help="also write the table here")
    args = ap.parse_args()

    files = args.files or sorted(glob.glob(os.path.join("results", "*", "*.json"))
                                 + glob.glob(os.path.join("results", "*", "cot_test", "*", "*",
                                                          "*.json")))
    groups = collections.defaultdict(list)
    for path in files + list(args.compare):
        with open(path) as fh:
            d = json.load(fh)
        if "mean_ms" not in d:
            continue
        parts = os.path.abspath(path).split(os.sep)
        if "cot_test" in parts:  # results/<model>/cot_test/<condition>/<clip>/<run>.json
            i = parts.index("cot_test")
            model, condition = parts[i - 1], parts[i + 1]
        else:
            model, condition = os.path.basename(os.path.dirname(path)), "-"
        s = d.get("settings", {})
        traj = s.get("num_traj_sets", 1) * s.get("num_traj_samples", 1)
        groups[(model, d.get("mode") or d.get("step"), condition, traj)].append(d)

    rows = []
    for (model, mode, condition, traj), runs in sorted(groups.items()):
        row = {"model": model, "mode": mode, "cot": condition, "traj": traj, "clips": len(runs),
               "windows": int(sum(r["n"] for r in runs)),
               "min_ade": np.mean([r["mean_min_ade"] for r in runs]),
               "ade": np.mean([r.get("mean_ade", r["mean_min_ade"]) for r in runs]),
               "mean_ms": np.mean([r["mean_ms"] for r in runs]),
               "median_ms": np.mean([r["median_ms"] for r in runs]),
               "p90_ms": np.mean([r["p90_ms"] for r in runs]),
               "tokens": np.mean([r.get("mean_tokens", np.nan) for r in runs]),
               "nominal": _frac_nominal(runs),
               "ade_nom": _mean_or_nan(runs, "mean_min_ade_nominal"),
               "ade_rsn": _mean_or_nan(runs, "mean_min_ade_reasoned"),
               "ms_nom": _mean_or_nan(runs, "mean_ms_nominal"),
               "ms_rsn": _mean_or_nan(runs, "mean_ms_reasoned")}
        for s in STAGES:
            row[s] = np.mean([r.get("stage_ms", {}).get(s, 0.0) for r in runs])
        rows.append(row)

    cols = ["model", "mode", "cot", "traj", "clips", "windows", "min_ade", "ade", "mean_ms",
            "median_ms", "p90_ms", "tokens", "nominal", "ade_nom", "ade_rsn", "ms_nom", "ms_rsn",
            *STAGES]
    widths = {c: max(len(c), *(len(_fmt(r[c])) for r in rows)) for c in cols} if rows else {}
    print("  ".join(c.rjust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(_fmt(r[c]).rjust(widths[c]) for c in cols))
    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows({c: _fmt(r[c]) for c in cols} for r in rows)
        print(f"wrote {args.csv}")


def _frac_nominal(runs: list) -> float:
    """Fraction of windows the model answered <NOMINAL> (Edge's adaptive reasoning); nan if the
    runs predate that field (flashdrive, older Edge results)."""
    nom = [r["n_nominal"] for r in runs if "n_nominal" in r]
    if not nom:
        return np.nan
    return float(sum(nom) / sum(r["n"] for r in runs if "n_nominal" in r))


def _mean_or_nan(runs: list, key: str) -> float:
    vals = [r[key] for r in runs if r.get(key) is not None]
    return float(np.mean(vals)) if vals else np.nan


def _fmt(v) -> str:
    if isinstance(v, float):
        return "" if np.isnan(v) else (f"{v:.4f}" if v < 10 else f"{v:.1f}")
    return str(v)


if __name__ == "__main__":
    main()
