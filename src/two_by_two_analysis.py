"""2x2 balanced-threshold calibration and evaluation analysis.

Per the pre-registration frozen in EXPERIMENT_NOTES.md before any of this
ran: balanced threshold is fit exclusively on the 60-world calibration set,
minimizing mean NLL with equal weight across cells A/B/C/D (every
calibration world contributes exactly one instance of each cell, so plain
pooling already weights them equally). Frozen before evaluation collection
begins. No re-optimization against false-merge/false-split rates.

Usage:
    python src/two_by_two_analysis.py fit-threshold   # calibration only, freezes the threshold
    python src/two_by_two_analysis.py evaluate         # 200-world eval, predefined metrics
    python src/two_by_two_analysis.py oracle-sweep      # diagnostic only
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import worldgen
import gates
import aggregate
import embed
from two_by_two import load_pool, build_cells, CELLS, CELL_DESIGN

ROOT = Path(__file__).resolve().parents[1]


def load_cfg_and_ab_fit():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    fit = json.loads((ROOT / "results" / "processed" / "calibration_fit.json").read_text())
    return cfg, fit


def cells_with_theta(cfg, range_name):
    pool = load_pool(cfg, range_name)
    cells_by_world = build_cells(pool)
    master_seed = cfg["master_seed"]
    theta_by_world = {wid: worldgen.generate_world(master_seed, wid, cfg).theta for wid in cells_by_world}
    return cells_by_world, theta_by_world


# ---------------------------------------------------------------------------
# Balanced threshold fit (calibration only)
# ---------------------------------------------------------------------------

def fit_balanced_threshold(cfg, ab_fit, cells_by_world, theta_by_world, threshold_grid) -> dict:
    prior_mean, prior_sd = cfg["dgp"]["prior_mean"], cfg["dgp"]["prior_sd"]
    sigma_r2 = ab_fit["sigma_r2"]

    per_threshold = []
    for t in threshold_grid:
        per_cell_nlls = {"A": [], "B": [], "C": [], "D": []}
        for wid, cells in cells_by_world.items():
            theta = theta_by_world[wid]
            for cell in ["A", "B", "C", "D"]:
                if cell not in cells:
                    continue
                values = [v for v, _ in cells[cell]]
                rationales = [r for _, r in cells[cell]]
                post, _ = aggregate.dedup_pool(values, rationales, prior_mean, prior_sd, sigma_r2, t)
                per_cell_nlls[cell].append(aggregate.nll(theta, post))
        cell_means = {c: sum(v) / len(v) for c, v in per_cell_nlls.items() if v}
        balanced_mean_nll = sum(cell_means.values()) / len(cell_means)  # equal weight per cell type
        per_threshold.append({"threshold": t, "cell_mean_nll": cell_means, "balanced_mean_nll": balanced_mean_nll})

    best = min(per_threshold, key=lambda r: r["balanced_mean_nll"])
    return {"sweep": per_threshold, "selected_threshold": best["threshold"], "selected_row": best}


def cmd_fit_threshold():
    cfg, ab_fit = load_cfg_and_ab_fit()
    cells_by_world, theta_by_world = cells_with_theta(cfg, "two_by_two_dedup_calibration")
    print(f"calibration worlds available: {len(cells_by_world)}")

    result = fit_balanced_threshold(
        cfg, ab_fit, cells_by_world, theta_by_world,
        cfg["two_by_two"]["dedup_threshold_grid_expanded"],
    )
    for row in result["sweep"]:
        print(f"t={row['threshold']:.2f}  balanced_mean_nll={row['balanced_mean_nll']:.4f}  "
              f"per_cell={ {k: round(v,3) for k,v in row['cell_mean_nll'].items()} }")
    print(f"\nFROZEN balanced threshold: {result['selected_threshold']}")

    out = {
        "balanced_dedup_threshold": result["selected_threshold"],
        "sweep": result["sweep"],
        "n_calibration_worlds": len(cells_by_world),
    }
    out_path = ROOT / "results" / "processed" / "two_by_two_calibration_fit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path} (frozen -- do not re-fit after seeing evaluation data)")


# ---------------------------------------------------------------------------
# Evaluation: predefined primary metrics
# ---------------------------------------------------------------------------

def _cluster_stats(values, rationales, threshold):
    vecs = embed.embed_texts(rationales)
    clusters = aggregate._cluster_by_threshold(vecs, threshold)
    return len(clusters), clusters


def _pair_flags(n, clusters):
    cluster_of = {}
    for ci, idxs in enumerate(clusters):
        for i in idxs:
            cluster_of[i] = ci
    same = []
    for i in range(n):
        for j in range(i + 1, n):
            same.append(cluster_of[i] == cluster_of[j])
    return same


def evaluate_cell_metrics(theta_by_world, posteriors_by_world) -> dict:
    z95 = 1.959963984540054
    errs, sds, nlls, covered = [], [], [], []
    for wid, post in posteriors_by_world.items():
        theta = theta_by_world[wid]
        err = post.mean - theta
        errs.append(err); sds.append(post.sd)
        nlls.append(aggregate.nll(theta, post))
        covered.append(1.0 if abs(err) <= z95 * post.sd else 0.0)
    n = len(errs)
    if n == 0:
        return {"n_worlds": 0}
    rmse = math.sqrt(sum(e ** 2 for e in errs) / n)
    mean_sd = sum(sds) / n
    return {
        "n_worlds": n, "rmse": rmse, "mean_posterior_sd": mean_sd,
        "coverage": sum(covered) / n, "mean_nll": sum(nlls) / n,
        "calibration_ratio": rmse / mean_sd if mean_sd > 0 else float("nan"),
    }


def run_evaluation(cfg, ab_fit, tt_calib_fit, cells_by_world, theta_by_world) -> dict:
    prior_mean, prior_sd = cfg["dgp"]["prior_mean"], cfg["dgp"]["prior_sd"]
    sigma_r2, sigma_hat2, nu_hat2 = ab_fit["sigma_r2"], ab_fit["sigma_hat2"], ab_fit["nu_hat2"]
    gamma_cal = ab_fit["gamma_cal"]
    legacy_t = ab_fit["dedup_threshold"]
    balanced_t = tt_calib_fit["balanced_dedup_threshold"]

    aggregator_results = {name: {c: {} for c in ["A", "B", "C", "D"]}
                           for name in ["naive", "provenance", "provenance_gamma", "dedup_legacy", "dedup_balanced"]}

    # mechanism diagnostics
    b_merge_flags, c_split_flags = [], []
    b_cluster_counts, c_cluster_counts, a_cluster_counts, d_cluster_counts = [], [], [], []

    per_cell_posteriors = {name: {c: {} for c in ["A", "B", "C", "D"]}
                            for name in aggregator_results}

    for wid, cells in cells_by_world.items():
        for cell in ["A", "B", "C", "D"]:
            if cell not in cells:
                continue
            values = [v for v, _ in cells[cell]]
            rationales = [r for _, r in cells[cell]]
            ancestry = CELL_DESIGN[cell]["ancestry"]

            per_cell_posteriors["naive"][cell][wid] = aggregate.naive_pool(values, prior_mean, prior_sd, sigma_r2)

            if ancestry == "shared_root":
                by_root = {0: values}
            else:
                by_root = {i: [v] for i, v in enumerate(values)}
            per_cell_posteriors["provenance"][cell][wid] = aggregate.provenance_pool(
                by_root, prior_mean, prior_sd, sigma_hat2, nu_hat2)
            per_cell_posteriors["provenance_gamma"][cell][wid] = aggregate.provenance_pool_gamma(
                by_root, prior_mean, prior_sd, sigma_hat2, nu_hat2, gamma_cal)

            post_legacy, n_clusters_legacy = aggregate.dedup_pool(values, rationales, prior_mean, prior_sd, sigma_r2, legacy_t)
            per_cell_posteriors["dedup_legacy"][cell][wid] = post_legacy
            post_balanced, n_clusters_balanced = aggregate.dedup_pool(values, rationales, prior_mean, prior_sd, sigma_r2, balanced_t)
            per_cell_posteriors["dedup_balanced"][cell][wid] = post_balanced

            # mechanism diagnostics use the BALANCED threshold (the one actually used for evaluation)
            _, clusters_b = (_cluster_stats(values, rationales, balanced_t) if cell == "B" else (None, None))
            if cell == "B":
                n_c, clusters = _cluster_stats(values, rationales, balanced_t)
                b_cluster_counts.append(n_c)
                b_merge_flags.extend(_pair_flags(len(values), clusters))
            if cell == "C":
                n_c, clusters = _cluster_stats(values, rationales, balanced_t)
                c_cluster_counts.append(n_c)
                c_split_flags.extend([not x for x in _pair_flags(len(values), clusters)])
            if cell == "A":
                n_c, _ = _cluster_stats(values, rationales, balanced_t)
                a_cluster_counts.append(n_c)
            if cell == "D":
                n_c, _ = _cluster_stats(values, rationales, balanced_t)
                d_cluster_counts.append(n_c)

    metrics = {}
    for name, by_cell in per_cell_posteriors.items():
        metrics[name] = {c: evaluate_cell_metrics(theta_by_world, posts) for c, posts in by_cell.items()}

    mechanism = {
        "false_merge_rate_cell_b": sum(b_merge_flags) / len(b_merge_flags) if b_merge_flags else float("nan"),
        "false_split_rate_cell_c": sum(c_split_flags) / len(c_split_flags) if c_split_flags else float("nan"),
        "mean_inferred_clusters": {
            "A_true_roots_1": sum(a_cluster_counts) / len(a_cluster_counts) if a_cluster_counts else float("nan"),
            "B_true_roots_4": sum(b_cluster_counts) / len(b_cluster_counts) if b_cluster_counts else float("nan"),
            "C_true_roots_1": sum(c_cluster_counts) / len(c_cluster_counts) if c_cluster_counts else float("nan"),
            "D_true_roots_4": sum(d_cluster_counts) / len(d_cluster_counts) if d_cluster_counts else float("nan"),
        },
        "balanced_threshold_used": balanced_t,
    }
    return {"aggregator_metrics": metrics, "mechanism": mechanism}


def cmd_evaluate():
    cfg, ab_fit = load_cfg_and_ab_fit()
    tt_calib_fit_path = ROOT / "results" / "processed" / "two_by_two_calibration_fit.json"
    if not tt_calib_fit_path.exists():
        raise RuntimeError("run 'fit-threshold' first -- the balanced threshold must be frozen before evaluation")
    tt_calib_fit = json.loads(tt_calib_fit_path.read_text())

    cells_by_world, theta_by_world = cells_with_theta(cfg, "two_by_two_eval")
    print(f"evaluation worlds available: {len(cells_by_world)}")

    result = run_evaluation(cfg, ab_fit, tt_calib_fit, cells_by_world, theta_by_world)

    print("\n=== Mechanism diagnostics (balanced threshold = "
          f"{tt_calib_fit['balanced_dedup_threshold']}) ===")
    for k, v in result["mechanism"].items():
        print(f"{k}: {v}")

    print("\n=== Per-cell metrics ===")
    for name, by_cell in result["aggregator_metrics"].items():
        print(f"\n-- {name} --")
        for cell in ["A", "B", "C", "D"]:
            m = by_cell[cell]
            print(f"  {cell}: n={m.get('n_worlds')} cov={m.get('coverage'):.3f} "
                  f"NLL={m.get('mean_nll'):.3f} C={m.get('calibration_ratio'):.3f}"
                  if m.get("n_worlds") else f"  {cell}: no data")

    out_path = ROOT / "results" / "processed" / "two_by_two_evaluation.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


# ---------------------------------------------------------------------------
# Oracle threshold sweep (diagnostic only)
# ---------------------------------------------------------------------------

def cmd_oracle_sweep():
    cfg, ab_fit = load_cfg_and_ab_fit()
    cells_by_world, theta_by_world = cells_with_theta(cfg, "two_by_two_eval")
    grid = cfg["two_by_two"]["dedup_threshold_grid_expanded"]

    rows = []
    for t in grid:
        b_flags, c_flags = [], []
        for wid, cells in cells_by_world.items():
            if "B" in cells:
                values = [v for v, _ in cells["B"]]
                rationales = [r for _, r in cells["B"]]
                _, clusters = _cluster_stats(values, rationales, t)
                b_flags.extend(_pair_flags(len(values), clusters))
            if "C" in cells:
                values = [v for v, _ in cells["C"]]
                rationales = [r for _, r in cells["C"]]
                _, clusters = _cluster_stats(values, rationales, t)
                c_flags.extend([not x for x in _pair_flags(len(values), clusters)])
        fmr_b = sum(b_flags) / len(b_flags) if b_flags else float("nan")
        fsr_c = sum(c_flags) / len(c_flags) if c_flags else float("nan")
        rows.append({"threshold": t, "false_merge_rate_b": fmr_b, "false_split_rate_c": fsr_c,
                      "distance_from_origin": math.sqrt(fmr_b ** 2 + fsr_c ** 2)})
        print(f"t={t:.2f}  FMR_B={fmr_b:.3f}  FSR_C={fsr_c:.3f}  dist={rows[-1]['distance_from_origin']:.3f}")

    best = min(rows, key=lambda r: r["distance_from_origin"])
    print(f"\nBest joint point (diagnostic only, not a selection criterion): "
          f"t={best['threshold']}, FMR_B={best['false_merge_rate_b']:.3f}, "
          f"FSR_C={best['false_split_rate_c']:.3f}, dist={best['distance_from_origin']:.3f}")

    out_path = ROOT / "results" / "processed" / "two_by_two_oracle_sweep.json"
    out_path.write_text(json.dumps({"sweep": rows, "closest_to_origin": best}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == "fit-threshold":
        cmd_fit_threshold()
    elif cmd == "evaluate":
        cmd_evaluate()
    elif cmd == "oracle-sweep":
        cmd_oracle_sweep()
    else:
        print("usage: python src/two_by_two_analysis.py {fit-threshold|evaluate|oracle-sweep}")
        sys.exit(1)
