"""Phase 0 pilot runner (EXPERIMENT.md Section 8).

Generates 30 worlds (k=1, n=8 reports/world, T=0.7) plus 3 leakage-control
calls per world, computes gates 1-6, and writes a pilot report. Does NOT
freeze anything and does NOT touch Grids A-D -- per the brief, a human must
review the gates before spending on the full grids.

Usage:
    python src/pilot.py
"""
from __future__ import annotations

import asyncio
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


async def run_pilot(cfg: dict) -> dict:
    pilot_cfg = cfg["pilot"]
    elicit_cfg = cfg["elicitation"]
    cost_cfg = cfg["cost"]
    master_seed = cfg["master_seed"]

    raw_log_path = ROOT / cfg["paths"]["results_dir"] / "pilot.jsonl"
    cache_dir = ROOT / cfg["paths"]["cache_dir"]

    tracker = CostTracker(
        hard_cap_usd=cost_cfg["hard_cap_usd"],
        price_in=cost_cfg["price_per_input_token"],
        price_out=cost_cfg["price_per_output_token"],
    )
    elicitor = Elicitor(cfg, cache_dir, tracker, max_concurrency=8)

    worlds = [worldgen.generate_world(master_seed, wid, cfg) for wid in range(pilot_cfg["w_worlds"])]
    docs = [worldgen.generate_root_for_world(w, 0, master_seed, cfg) for w in worlds]

    report_tasks = []
    for w, d in zip(worlds, docs):
        for i in range(pilot_cfg["n"]):
            report_tasks.append(
                elicitor.elicit(
                    world_id=w.world_id,
                    root_id=0,
                    call_kind="report",
                    document_text=d.text,
                    model=elicit_cfg["main_model"],
                    temperature=pilot_cfg["temperature"],
                    seed=i,
                    jsonl_path=raw_log_path,
                )
            )

    leakage_tasks = []
    for w in worlds:
        for i in range(cfg["leakage_control"]["calls_per_world"]):
            leakage_tasks.append(
                elicitor.elicit(
                    world_id=w.world_id,
                    root_id=None,
                    call_kind="leakage",
                    document_text=w.entity_name,
                    model=elicit_cfg["main_model"],
                    temperature=pilot_cfg["temperature"],
                    seed=1000 + i,
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
        print(f"[pilot] halted early: {e}")

    return {
        "worlds": worlds,
        "docs": docs,
        "report_results": report_results,
        "leakage_results": leakage_results,
        "cost_spent_usd": tracker.spent_usd,
        "halted_early": halted_early,
    }


def compute_gates(cfg: dict, worlds, docs, report_results, leakage_results) -> str:
    pilot_cfg = cfg["pilot"]
    docs_by_world = {d.world_id: d for d in docs}
    worlds_by_id = {w.world_id: w for w in worlds}

    valid_reports = [r for r in report_results if r.valid]
    valid_leakage = [r for r in leakage_results if r.valid]

    # --- Gate 1: parse / validity rate ---
    validity = gates.validity_rate([r.valid for r in report_results])

    # --- report vs E, grouped by world ---
    report_targets = [docs_by_world[r.world_id].e_true_float() for r in valid_reports]
    report_values = [r.parsed_estimate for r in valid_reports]

    by_world: dict[int, list[float]] = {}
    for r in valid_reports:
        by_world.setdefault(r.world_id, []).append(r.parsed_estimate)

    # --- Gate 2: within-document report sd meaningfully > 0 ---
    within_doc_sds = [gates.sample_sd(vals) for vals in by_world.values() if len(vals) > 1]
    mean_within_doc_sd = gates.mean(within_doc_sds) if within_doc_sds else 0.0

    # --- Gate 3: rho_hat via known-DGP variance decomposition ---
    eps_values = [d.eps for d in docs]
    extraction_noise = [r.parsed_estimate - docs_by_world[r.world_id].e_true_float() for r in valid_reports]
    decomp = gates.variance_decomposition(eps_values, extraction_noise)

    # --- Gate 4: leakage control (practical criterion; statistical gate needs ~100 worlds) ---
    leak_targets = [worlds_by_id[r.world_id].theta for r in valid_leakage]
    leak_values = [r.parsed_estimate for r in valid_leakage]
    leak_result = gates.leakage_control(
        leak_values, leak_targets,
        prior_mean=cfg["dgp"]["prior_mean"],
        practical_threshold=cfg["leakage_control"]["practical_rmse_improvement_max"],
        statistical_corr_threshold=cfg["leakage_control"]["statistical_corr_threshold_at_100"],
    )

    # --- Gate 5: bias around E, small and stable ---
    report_bias = gates.bias(report_values, report_targets)

    # --- Gate 6: informative but nontrivial ---
    report_rmse = gates.rmse(report_values, report_targets)
    prior_sd = cfg["dgp"]["prior_sd"]
    rmse_ratio = report_rmse / prior_sd

    gate_1_pass = validity >= 0.95
    gate_2_pass = mean_within_doc_sd > 1.0  # meaningfully > 0, not clone regime
    gate_3_pass = pilot_cfg["rho_hat_band"][0] <= decomp.predicted_rho <= pilot_cfg["rho_hat_band"][1]
    gate_4_pass = leak_result.practical_pass  # statistical gate deferred to full 100-world calibration run
    gate_5_pass = abs(report_bias) < 0.15 * prior_sd
    gate_6_pass = rmse_ratio < pilot_cfg["rmse_vs_prior_sd_max"] and mean_within_doc_sd > 1.0

    lines = []
    lines.append("# Phase 0 pilot report\n")
    lines.append(f"Worlds: {len(worlds)}, reports/world: {pilot_cfg['n']}, "
                  f"temperature: {pilot_cfg['temperature']}\n")
    lines.append(f"Total report calls: {len(report_results)}, valid: {len(valid_reports)}\n")
    lines.append(f"Total leakage calls: {len(leakage_results)}, valid: {len(valid_leakage)}\n\n")

    lines.append("## Gate 1: parse + validity rate >= 95%\n")
    lines.append(f"Validity rate: {validity:.3f} -- {'PASS' if gate_1_pass else 'FAIL'}\n\n")

    lines.append("## Gate 2: extraction noise is real (not clone regime)\n")
    lines.append(f"Mean within-document report sd: {mean_within_doc_sd:.2f} M EUR -- "
                  f"{'PASS' if gate_2_pass else 'FAIL'}\n\n")

    lines.append("## Gate 3: rho_hat in workable band [0.2, 0.95]\n")
    lines.append(f"sigma_hat^2 = {decomp.sigma_hat2:.2f} (Var(E-Theta))\n")
    lines.append(f"nu_hat^2 = {decomp.nu_hat2:.2f} (Var(report-E))\n")
    lines.append(f"rho_hat (predicted, sigma^2/(sigma^2+nu^2)) = {decomp.predicted_rho:.3f} -- "
                  f"{'PASS' if gate_3_pass else 'FAIL'}\n\n")

    lines.append("## Gate 4: leakage control\n")
    lines.append(f"RMSE(no-doc estimate, Theta) = {leak_result.rmse_leakage:.2f}\n")
    lines.append(f"RMSE(constant prior mean, Theta) = {leak_result.rmse_baseline:.2f}\n")
    lines.append(f"Practical improvement over baseline: {leak_result.practical_improvement:.1%} "
                  f"(must be < {cfg['leakage_control']['practical_rmse_improvement_max']:.0%}) -- "
                  f"{'PASS' if gate_4_pass else 'FAIL'}\n")
    lines.append(f"corr(no-doc estimate, Theta) = {leak_result.corr:.3f}, "
                  f"permutation p = {leak_result.permutation_p:.4f} (supplementary; statistical "
                  f"gate requires the full ~100-world calibration set, not this 30-world pilot)\n\n")

    lines.append("## Gate 5: reports approximately unbiased around E\n")
    lines.append(f"Bias (mean report - E): {report_bias:.2f} M EUR -- "
                  f"{'PASS' if gate_5_pass else 'FAIL'}\n\n")

    lines.append("## Gate 6: task informative but nontrivial\n")
    lines.append(f"RMSE(report, E) = {report_rmse:.2f} M EUR, prior sd = {prior_sd:.1f} M EUR, "
                  f"ratio = {rmse_ratio:.3f} (must be < {pilot_cfg['rmse_vs_prior_sd_max']}) -- "
                  f"{'PASS' if gate_6_pass else 'FAIL'}\n\n")

    all_pass = all([gate_1_pass, gate_2_pass, gate_3_pass, gate_4_pass, gate_5_pass, gate_6_pass])
    lines.append(f"## Overall: {'ALL GATES PASS' if all_pass else 'SOME GATES FAILED'}\n")
    lines.append("\nPer EXPERIMENT.md Section 8: STOP here for human review. Do not freeze "
                  "templates or proceed to Grids A-D until a human has read this report.\n")

    return "".join(lines)


async def main():
    cfg = load_config()
    result = await run_pilot(cfg)
    report_text = compute_gates(
        cfg, result["worlds"], result["docs"],
        result["report_results"], result["leakage_results"],
    )
    out_path = ROOT / cfg["paths"]["processed_dir"] / "pilot_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\n[pilot] total cost: ${result['cost_spent_usd']:.4f}")
    if result["halted_early"]:
        print("[pilot] WARNING: run halted early due to cost cap; results are partial.")


if __name__ == "__main__":
    asyncio.run(main())
