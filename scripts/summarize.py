"""One table over every run in results/: minADE and latency per mode, averaged over clips.

    python scripts/summarize.py                                  # results/*/*.json
    python scripts/summarize.py --compare ../flashdrive/results/*.json   # other runs too

flashdrive's result files share the keys (its "step" is this project's "mode"), so they
line up in the same table.
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

    files = args.files or sorted(glob.glob(os.path.join("results", "*", "*.json")))
    groups = collections.defaultdict(list)
    for path in files + list(args.compare):
        with open(path) as fh:
            d = json.load(fh)
        if "mean_ms" not in d:
            continue
        model = os.path.basename(os.path.dirname(path))
        s = d.get("settings", {})
        traj = s.get("num_traj_sets", 1) * s.get("num_traj_samples", 1)
        groups[(model, d.get("mode") or d.get("step"), traj)].append(d)

    rows = []
    for (model, mode, traj), runs in sorted(groups.items()):
        row = {"model": model, "mode": mode, "traj": traj, "clips": len(runs),
               "windows": int(sum(r["n"] for r in runs)),
               "min_ade": np.mean([r["mean_min_ade"] for r in runs]),
               "ade": np.mean([r.get("mean_ade", r["mean_min_ade"]) for r in runs]),
               "mean_ms": np.mean([r["mean_ms"] for r in runs]),
               "median_ms": np.mean([r["median_ms"] for r in runs]),
               "p90_ms": np.mean([r["p90_ms"] for r in runs]),
               "tokens": np.mean([r.get("mean_tokens", np.nan) for r in runs])}
        for s in STAGES:
            row[s] = np.mean([r.get("stage_ms", {}).get(s, 0.0) for r in runs])
        rows.append(row)

    cols = ["model", "mode", "traj", "clips", "windows", "min_ade", "ade", "mean_ms",
            "median_ms", "p90_ms", "tokens", *STAGES]
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


def _fmt(v) -> str:
    if isinstance(v, float):
        return "" if np.isnan(v) else (f"{v:.4f}" if v < 10 else f"{v:.1f}")
    return str(v)


if __name__ == "__main__":
    main()
