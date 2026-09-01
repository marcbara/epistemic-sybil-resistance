"""Calibration split (EXPERIMENT.md Section 4).

Collects data on a disjoint set of ~100 worlds, never used for evaluation
(Grid A/B use world_id range `grid_ab`, this uses `calibration` -- see
config.yaml's `world_id_ranges`). At the frozen pilot settings (k=1, n=8,
T=0.7), estimates:
  - sigma_r^2 = Var(report - Theta)          (naive aggregator's per-report variance)
  - rho_hat via the known-DGP variance decomposition (Corollary 2's rho)
  - sigma_hat^2, nu_hat^2                    (variance components)
  - per-report bias

Also runs the leakage control's full statistical gate (Section 3), which
the brief specifies must run on ~100 calibration worlds, not just the pilot.

Writes results/processed/calibration_fit.json, the frozen parameterization
consumed by aggregate.py's aggregators. No parameter here is fitted on
Grid A/B's evaluation worlds.

Usage:
    python src/calibration.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import worldgen
import gates
from elicit import Elicitor, CostTracker, BudgetExceededError, append_jsonl, ElicitResult

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


async def run_calibration(cfg: dict) -> dict:
    id_range = cfg["world_id_ranges"]["calibration"]
    pilot_cfg = cfg["pilot"]  # frozen settings: k=1, n=8, T=0.7
    elicit_cfg = cfg["elicitation"]
    cost_cfg = cfg["cost"]
    master_seed = cfg["master_seed"]

    raw_log_path = ROOT / cfg["paths"]["results_dir"] / "calibration.jsonl"
    cache_dir = ROOT / cfg["paths"]["cache_dir"]

    tracker = CostTracker(
        hard_cap_usd=cost_cfg["hard_cap_usd"],
        price_in=cost_cfg["price_per_input_token"],
        price_out=cost_cfg["price_per_output_token"],
    )
    elicitor = Elicitor(cfg, cache_dir, tracker, max_concurrency=8)

    world_ids = range(id_range["base"], id_range["base"] + id_range["count"])
    worlds = [worldgen.generate_world(master_seed, wid, cfg) for wid in world_ids]
    docs = [worldgen.generate_root_for_world(w, 0, master_seed, cfg) for w in worlds]

    report_tasks = []
    for w, d in zip(worlds, docs):
        for i in range(pilot_cfg["n"]):
            report_tasks.append(
                elicitor.elicit(
                    world_id=w.world_id, root_id=0, call_kind="report",
                    document_text=d.text, model=elicit_cfg["main_model"],
                    temperature=pilot_cfg["temperature"], seed=i,
                    jsonl_path=raw_log_path,
                )
            )
    leakage_tasks = []
    for w in worlds:
        for i in range(cfg["leakage_control"]["calls_per_world"]):
            leakage_tasks.append(
                elicitor.elicit(
                    world_id=w.world_id, root_id=None, call_kind="leakage",
                    document_text=w.entity_name, model=elicit_cfg["main_model"],
                    temperature=pilot_cfg["temperature"], seed=1000 + i,
                    jsonl_path=raw_log_path,
                )
            )

    report_results: list[ElicitResult] = []
    leakage_results: list[ElicitResult] = []
    halted_early = False
    try:
        for coro in asyncio.as_completed(report_tasks):
            r = await coro
            report_results.append(r)
        for coro in asyncio.as_completed(leakage_tasks):
            r = await coro
            leakage_results.append(r)
    except BudgetExceededError as e:
        halted_early = True
        print(f"[calibration] halted early: {e}")

    return {
        "worlds": worlds, "docs": docs,
        "report_results": report_results, "leakage_results": leakage_results,
        "cost_spent_usd": tracker.spent_usd, "halted_early": halted_early,
    }


def fit_calibration(cfg: dict, worlds, docs, report_results, leakage_results) -> dict:
    docs_by_world = {d.world_id: d for d in docs}
    worlds_by_id = {w.world_id: w for w in worlds}

    valid_reports = [r for r in report_results if r.valid]
    valid_leakage = [r for r in leakage_results if r.valid]

    report_values = [r.parsed_estimate for r in valid_reports]
    report_targets_theta = [worlds_by_id[r.world_id].theta for r in valid_reports]
    report_targets_e = [docs_by_world[r.world_id].e_true_float() for r in valid_reports]

    sigma_r2 = gates.sample_variance([r - t for r, t in zip(report_values, report_targets_theta)])
    report_bias = gates.bias(report_values, report_targets_e)

    eps_values = [d.eps for d in docs]
    extraction_noise = [r.parsed_estimate - docs_by_world[r.world_id].e_true_float() for r in valid_reports]
    decomp = gates.variance_decomposition(eps_values, extraction_noise)

    leak_targets = [worlds_by_id[r.world_id].theta for r in valid_leakage]
    leak_values = [r.parsed_estimate for r in valid_leakage]
    leak_result = gates.leakage_control(
        leak_values, leak_targets,
        prior_mean=cfg["dgp"]["prior_mean"],
        practical_threshold=cfg["leakage_control"]["practical_rmse_improvement_max"],
        statistical_corr_threshold=cfg["leakage_control"]["statistical_corr_threshold_at_100"],
        n_perm=20_000,
    )

    return {
        "n_worlds": len(worlds),
        "n_valid_reports": len(valid_reports),
        "n_valid_leakage": len(valid_leakage),
        "validity_rate": gates.validity_rate([r.valid for r in report_results]),
        "sigma_r2": sigma_r2,
        "report_bias": report_bias,
        "sigma_hat2": decomp.sigma_hat2,
        "nu_hat2": decomp.nu_hat2,
        "rho_hat": decomp.predicted_rho,
        "leakage": {
            "rmse_leakage": leak_result.rmse_leakage,
            "rmse_baseline": leak_result.rmse_baseline,
            "practical_improvement": leak_result.practical_improvement,
            "corr": leak_result.corr,
            "permutation_p": leak_result.permutation_p,
            "practical_pass": leak_result.practical_pass,
            "statistical_pass": leak_result.statistical_pass,
        },
    }


async def main():
    cfg = load_config()
    result = await run_calibration(cfg)
    fit = fit_calibration(
        cfg, result["worlds"], result["docs"],
        result["report_results"], result["leakage_results"],
    )
    out_path = ROOT / cfg["paths"]["processed_dir"] / "calibration_fit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fit, indent=2))
    print(json.dumps(fit, indent=2))
    print(f"\n[calibration] total cost: ${result['cost_spent_usd']:.4f}")
    if result["halted_early"]:
        print("[calibration] WARNING: run halted early due to cost cap; results are partial.")


if __name__ == "__main__":
    asyncio.run(main())
