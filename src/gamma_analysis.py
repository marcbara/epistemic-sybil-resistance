"""Out-of-sample gamma validation, gamma-aware aggregation, bias sensitivity.

Methodological round requested after the first diagnostics: the initial
gamma_hat = 0.703 was estimated from Grid A's own within-document variance
and then used to explain Grid A's precision curve -- valid as a post-hoc
diagnostic, but in-sample. This module does the clean version, with no new
API calls:

1. gamma_cal: estimated EXCLUSIVELY from the 100 calibration worlds
   (8 reports/world), frozen, then used to predict Grid A's 300-world
   precision curve out-of-sample. Stored into calibration_fit.json.
2. Gamma-aware provenance aggregator (aggregate.provenance_pool_gamma with
   gamma_cal) run on Grid A and Grid B alongside naive / provenance /
   dedup -- exploratory, clearly labeled, parameters all calibration-only.
3. Bias sensitivity: the calibration bias estimate subtracted from every
   report, all aggregators re-run. Secondary check, not a change to the
   main analysis (the brief said subtract "if material"; it was left in,
   disclosed, and here quantified).

Usage:
    python src/gamma_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import worldgen
import gates
import aggregate
from analyze import load_pool, evaluate_cell, _strip_private, GRID_A_NS, GRID_B_KS, GRID_B_N
from diagnostics import precision_curve, fig8_precision_curve

ROOT = Path(__file__).resolve().parents[1]


def load_cfg_fit():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    fit = json.loads((ROOT / "results" / "processed" / "calibration_fit.json").read_text())
    return cfg, fit


def estimate_gamma_cal(cfg, fit) -> dict:
    """gamma from calibration worlds only: 1 - (mean within-doc var)/nu_hat2,
    both factors computed on the calibration split."""
    recs = [json.loads(l) for l in open(ROOT / "results" / "raw" / "calibration.jsonl", encoding="utf-8")]
    seen, ded = set(), []
    for r in recs:
        if r["cache_key"] in seen:
            continue
        seen.add(r["cache_key"]); ded.append(r)
    reports = [r for r in ded if r["call_kind"] == "report" and r["valid"]]

    by_world: dict[int, list[float]] = {}
    for r in reports:
        by_world.setdefault(r["world_id"], []).append(r["parsed_estimate"])

    within_vars = [gates.sample_variance(v) for v in by_world.values() if len(v) > 1]
    mean_within = sum(within_vars) / len(within_vars)
    nu2 = fit["nu_hat2"]  # calibration-split Var(R - E)
    gamma_cal = 1.0 - mean_within / nu2
    return {
        "n_worlds": len(within_vars),
        "reports_per_world": 8,
        "mean_within_document_variance": mean_within,
        "nu_hat2": nu2,
        "gamma_cal": gamma_cal,
    }


def out_of_sample_check(cfg, fit, pool, gamma_cal: float) -> list[dict]:
    rows, _ = precision_curve(cfg, fit, pool)
    sigma2, nu2 = fit["sigma_hat2"], fit["nu_hat2"]
    for r in rows:
        m = r["m"]
        pred = sigma2 + gamma_cal * nu2 + (1 - gamma_cal) * nu2 / m
        r["predicted_var_gamma_cal"] = pred
        r["gamma_cal_pred_in_ci"] = r["empirical_var_ci95"][0] <= pred <= r["empirical_var_ci95"][1]
    return rows


def run_grids_extended(cfg, pool, fit, gamma_cal: float, bias: float = 0.0) -> dict:
    """Grid A and B with naive / provenance / gamma-aware provenance, with an
    optional bias subtracted from every report value (bias=0 -> main analysis)."""
    master_seed = cfg["master_seed"]
    prior_mean, prior_sd = cfg["dgp"]["prior_mean"], cfg["dgp"]["prior_sd"]
    sigma_r2, sigma_hat2, nu_hat2 = fit["sigma_r2"], fit["sigma_hat2"], fit["nu_hat2"]

    thetas = {wid: worldgen.generate_world(master_seed, wid, cfg).theta for wid in pool}

    grid_a = {}
    for n in GRID_A_NS:
        naive_p, prov_p, gamma_p, theta_w = {}, {}, {}, {}
        for wid, roots in pool.items():
            if 0 not in roots or len(roots[0]) < n:
                continue
            values = [v - bias for v, _, _ in roots[0][:n]]
            theta_w[wid] = thetas[wid]
            naive_p[wid] = aggregate.naive_pool(values, prior_mean, prior_sd, sigma_r2)
            prov_p[wid] = aggregate.provenance_pool({0: values}, prior_mean, prior_sd, sigma_hat2, nu_hat2)
            gamma_p[wid] = aggregate.provenance_pool_gamma(
                {0: values}, prior_mean, prior_sd, sigma_hat2, nu_hat2, gamma_cal
            )
        grid_a[n] = {
            "naive": evaluate_cell(theta_w, naive_p),
            "provenance": evaluate_cell(theta_w, prov_p),
            "provenance_gamma": evaluate_cell(theta_w, gamma_p),
        }

    grid_b = {}
    for k in GRID_B_KS:
        m = GRID_B_N // k
        naive_p, prov_p, gamma_p, theta_w = {}, {}, {}, {}
        for wid, roots in pool.items():
            root_ids = list(range(k))
            if not all(rid in roots and len(roots[rid]) >= m for rid in root_ids):
                continue
            by_root = {rid: [v - bias for v, _, _ in roots[rid][:m]] for rid in root_ids}
            all_values = [v for rid in root_ids for v in by_root[rid]]
            theta_w[wid] = thetas[wid]
            naive_p[wid] = aggregate.naive_pool(all_values, prior_mean, prior_sd, sigma_r2)
            prov_p[wid] = aggregate.provenance_pool(by_root, prior_mean, prior_sd, sigma_hat2, nu_hat2)
            gamma_p[wid] = aggregate.provenance_pool_gamma(
                by_root, prior_mean, prior_sd, sigma_hat2, nu_hat2, gamma_cal
            )
        grid_b[k] = {
            "naive": evaluate_cell(theta_w, naive_p),
            "provenance": evaluate_cell(theta_w, prov_p),
            "provenance_gamma": evaluate_cell(theta_w, gamma_p),
        }

    return {
        "grid_a": {n: {a: _strip_private(c) for a, c in cells.items()} for n, cells in grid_a.items()},
        "grid_b": {k: {a: _strip_private(c) for a, c in cells.items()} for k, cells in grid_b.items()},
    }


def main():
    cfg, fit = load_cfg_fit()
    pool = load_pool(cfg)

    print("=== 1. gamma_cal from calibration worlds only ===")
    gcal_info = estimate_gamma_cal(cfg, fit)
    for k, v in gcal_info.items():
        print(f"{k}: {v}")
    gamma_cal = gcal_info["gamma_cal"]

    # freeze into calibration_fit.json alongside the other calibration-only params
    fit["gamma_cal"] = gamma_cal
    fit_path = ROOT / "results" / "processed" / "calibration_fit.json"
    fit_path.write_text(json.dumps(fit, indent=2))

    print("\n=== 2. Out-of-sample precision-curve check on Grid A (300 worlds) ===")
    rows = out_of_sample_check(cfg, fit, pool, gamma_cal)
    for r in rows:
        print(f"m={r['m']:2d}  empirical={r['empirical_var']:8.1f} "
              f"[{r['empirical_var_ci95'][0]:8.1f}, {r['empirical_var_ci95'][1]:8.1f}]  "
              f"indep_pred={r['predicted_var']:8.1f}  "
              f"gamma_cal_pred={r['predicted_var_gamma_cal']:8.1f}  "
              f"in_CI={r['gamma_cal_pred_in_ci']}")

    # redraw fig8 with the calibration-frozen gamma (out-of-sample version)
    gamma_info_for_fig = {
        "gamma_hat": gamma_cal,
        "nu_hat2_total": fit["nu_hat2"],
        "independent_model_ceiling_sigma2": fit["sigma_hat2"],
    }
    p8 = fig8_precision_curve(rows, gamma_info_for_fig)
    print(f"rewrote {p8} (gamma curve now uses gamma_cal, out-of-sample)")

    print("\n=== 3. Gamma-aware provenance aggregator (exploratory, gamma_cal) ===")
    main_res = run_grids_extended(cfg, pool, fit, gamma_cal, bias=0.0)
    print("Grid A coverage / calibration ratio C:")
    print(" n | naive_cov | prov_cov | gamma_cov | naive_C | prov_C | gamma_C")
    for n in GRID_A_NS:
        c = main_res["grid_a"][n]
        print(f"{n:2d} | {c['naive']['coverage']:.3f} | {c['provenance']['coverage']:.3f} | "
              f"{c['provenance_gamma']['coverage']:.3f} | {c['naive']['calibration_ratio']:.3f} | "
              f"{c['provenance']['calibration_ratio']:.3f} | {c['provenance_gamma']['calibration_ratio']:.3f}")
    print("Grid B coverage / NLL:")
    print(" k | prov_cov | gamma_cov | prov_nll | gamma_nll")
    for k in GRID_B_KS:
        c = main_res["grid_b"][k]
        print(f"{k:2d} | {c['provenance']['coverage']:.3f} | {c['provenance_gamma']['coverage']:.3f} | "
              f"{c['provenance']['mean_nll']:.3f} | {c['provenance_gamma']['mean_nll']:.3f}")

    print("\n=== 4. Bias sensitivity (bias_cal subtracted from every report) ===")
    bias = fit["report_bias"]
    bias_res = run_grids_extended(cfg, pool, fit, gamma_cal, bias=bias)
    print(f"(bias_cal = {bias:.2f})")
    print("Grid A coverage:")
    print(" n | naive | provenance | provenance_gamma")
    for n in GRID_A_NS:
        c = bias_res["grid_a"][n]
        print(f"{n:2d} | {c['naive']['coverage']:.3f} | {c['provenance']['coverage']:.3f} | "
              f"{c['provenance_gamma']['coverage']:.3f}")
    print("Grid B coverage:")
    print(" k | naive | provenance | provenance_gamma")
    for k in GRID_B_KS:
        c = bias_res["grid_b"][k]
        print(f"{k:2d} | {c['naive']['coverage']:.3f} | {c['provenance']['coverage']:.3f} | "
              f"{c['provenance_gamma']['coverage']:.3f}")

    out = {
        "gamma_cal": gcal_info,
        "out_of_sample_precision_curve": rows,
        "main": main_res,
        "bias_subtracted": bias_res,
    }
    out_path = ROOT / "results" / "processed" / "gamma_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
