"""Score a predictions file against the benchmark's held-out truth.

Input: a JSONL file where each line is
    {"instance_id": "<id from instances.jsonl>", "mean": <float>, "sd": <float>}
i.e. a Gaussian posterior over Theta for that instance. Every instance in
benchmark/data/instances.jsonl must be predicted exactly once; missing or
duplicated instance_ids are an error (no silent partial scoring).

Output: per-cell metrics identical in definition to the paper's own analysis
(src/analyze.py): RMSE, empirical 95% credible-interval coverage, mean
negative log score, and the calibration ratio C = RMSE / mean posterior sd,
each with a 95% cluster bootstrap CI by world (1000 resamples, seed 0).
Written to <predictions>.summary.json and printed as a markdown table.

Usage:
    python benchmark/score.py <predictions.jsonl>
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmark" / "data"

Z95 = 1.959963984540054


def nll(theta: float, mean: float, sd: float) -> float:
    var = sd * sd
    return 0.5 * math.log(2.0 * math.pi * var) + (theta - mean) ** 2 / (2.0 * var)


def bootstrap_ci(errs, sds, covered, statistic, n_boot=1000, seed=0):
    rng = random.Random(seed)
    n = len(errs)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = list(range(n))
    stats = []
    for _ in range(n_boot):
        sample = [rng.choice(idx) for _ in range(n)]
        e = [errs[i] for i in sample]
        s = [sds[i] for i in sample]
        c = [covered[i] for i in sample]
        if statistic == "rmse":
            stats.append(math.sqrt(sum(x**2 for x in e) / n))
        elif statistic == "coverage":
            stats.append(sum(c) / n)
        elif statistic == "calibration_ratio":
            rmse = math.sqrt(sum(x**2 for x in e) / n)
            mean_sd = sum(s) / n
            stats.append(rmse / mean_sd if mean_sd > 0 else float("nan"))
    stats.sort()
    return (stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1])


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python benchmark/score.py <predictions.jsonl>")
    pred_path = Path(sys.argv[1])

    instances = {}
    for line in open(DATA_DIR / "instances.jsonl", encoding="utf-8"):
        r = json.loads(line)
        instances[r["instance_id"]] = (r["grid"], json.dumps(r["cell"]), r["world_id"])

    theta = {}
    for line in open(DATA_DIR / "truth.jsonl", encoding="utf-8"):
        r = json.loads(line)
        theta[r["world_id"]] = r["theta"]

    preds = {}
    for line in open(pred_path, encoding="utf-8"):
        r = json.loads(line)
        iid = r["instance_id"]
        if iid not in instances:
            sys.exit(f"Unknown instance_id in predictions: {iid}")
        if iid in preds:
            sys.exit(f"Duplicate prediction for instance_id: {iid}")
        preds[iid] = (float(r["mean"]), float(r["sd"]))
    missing = set(instances) - set(preds)
    if missing:
        sys.exit(f"{len(missing)} instances missing predictions, e.g. {sorted(missing)[:3]}")

    cells: dict = {}
    for iid, (grid, cell_key, wid) in instances.items():
        mean, sd = preds[iid]
        err = mean - theta[wid]
        cells.setdefault((grid, cell_key), []).append(
            (err, sd, 1.0 if abs(err) <= Z95 * sd else 0.0)
        )

    summary = {}
    rows = []
    for (grid, cell_key), triples in sorted(cells.items()):
        errs = [t[0] for t in triples]
        sds = [t[1] for t in triples]
        covered = [t[2] for t in triples]
        thetas_used = None  # errs already relative to theta
        n = len(errs)
        rmse = math.sqrt(sum(e**2 for e in errs) / n)
        mean_sd = sum(sds) / n
        coverage = sum(covered) / n
        mean_nll = sum(
            nll(0.0, -e, s) for e, s in zip(errs, sds)
        ) / n  # nll depends only on (theta - mean, sd) = (-err, sd)
        cr = rmse / mean_sd if mean_sd > 0 else float("nan")
        cell_summary = {
            "n_worlds": n,
            "rmse": rmse,
            "rmse_ci95": list(bootstrap_ci(errs, sds, covered, "rmse")),
            "coverage": coverage,
            "coverage_ci95": list(bootstrap_ci(errs, sds, covered, "coverage")),
            "mean_nll": mean_nll,
            "calibration_ratio": cr,
            "calibration_ratio_ci95": list(bootstrap_ci(errs, sds, covered, "calibration_ratio")),
        }
        summary[f"{grid} {cell_key}"] = cell_summary
        rows.append((grid, cell_key, n, coverage, cr, mean_nll, rmse))

    out_path = pred_path.with_suffix(pred_path.suffix + ".summary.json")
    json.dump(summary, open(out_path, "w"), indent=2)

    print(f"\nScored {len(preds)} predictions from {pred_path}\n")
    print("| Grid | Cell | Worlds | Coverage@95 | Calib. ratio | Mean NLL | RMSE |")
    print("|---|---|---|---|---|---|---|")
    for grid, cell_key, n, cov, cr, mnll, rmse in rows:
        print(f"| {grid} | {cell_key} | {n} | {cov:.3f} | {cr:.3f} | {mnll:.3f} | {rmse:.2f} |")
    print(f"\nFull summary with bootstrap CIs: {out_path}")


if __name__ == "__main__":
    main()
