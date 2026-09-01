"""Grid A + Grid B data collection (EXPERIMENT.md Section 3).

Grid A (prediction 13.1): k=1, n in {1,2,4,8,16,32}, 32 reports on root 1
per world, nested (n=8 is the first 8 of the 32, etc.).

Grid B (12.4 analogue): n=16 fixed, k in {1,2,4,8,16}, same worlds as Grid A.
Root 1 reuses Grid A's 32-report pool; roots 2..16 need at most
8, 4, 4, 2, 2, 2, 2, 1x8 reports respectively (nested so e.g. root 2's first
4 reports serve both k=4 and k=2's larger allocation) -- 32 extra calls per
world, exactly matching the brief's cost formula (64 calls/world total).

This module only COLLECTS the report pool; subsetting into the n/k grid
cells for analysis happens in analyze.py, which doesn't need new API calls
to explore different (n,k) slices of the same pool.

Usage:
    python src/grid_ab.py [--worlds N] [--dry-run-cost]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import worldgen
from elicit import Elicitor, CostTracker, BudgetExceededError, append_jsonl, ElicitResult

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config.yaml"

# Reports needed per root (0-indexed) to cover both grids' nesting.
# root 0: Grid A's full n=32 pool (also serves Grid B k=1's n=16 subset).
# roots 1..15: Grid B's extra roots, per the brief's nesting counts.
REPORTS_PER_ROOT = [32, 8, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
assert len(REPORTS_PER_ROOT) == 16
assert sum(REPORTS_PER_ROOT) == 64  # matches Section 3's "about 64 calls/world"


def load_config() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def build_tasks(worlds, docs_by_world_root, elicitor, elicit_cfg, pilot_cfg, jsonl_path):
    tasks = []
    for w in worlds:
        for root_id, count in enumerate(REPORTS_PER_ROOT):
            doc = docs_by_world_root[(w.world_id, root_id)]
            for i in range(count):
                tasks.append(
                    elicitor.elicit(
                        world_id=w.world_id, root_id=root_id, call_kind="report",
                        document_text=doc.text, model=elicit_cfg["main_model"],
                        temperature=pilot_cfg["temperature"], seed=i,
                        jsonl_path=jsonl_path,
                    )
                )
    return tasks


def estimate_cost(cfg: dict, n_worlds: int) -> float:
    # Calibration run's actual observed rate: $0.743 / 1100 calls ~ $0.00068/call,
    # with a 50% margin for report-call-heavier mix here (leakage calls, which
    # are shorter, made up ~27% of calibration's calls but 0% of grid_ab's).
    avg_cost_per_call = 0.001
    n_calls = n_worlds * sum(REPORTS_PER_ROOT)
    return n_calls * avg_cost_per_call


async def run_grid_ab(cfg: dict, n_worlds: int) -> dict:
    id_range = cfg["world_id_ranges"]["grid_ab"]
    pilot_cfg = cfg["pilot"]
    elicit_cfg = cfg["elicitation"]
    cost_cfg = cfg["cost"]
    master_seed = cfg["master_seed"]

    raw_log_path = ROOT / cfg["paths"]["results_dir"] / "grid_ab.jsonl"
    cache_dir = ROOT / cfg["paths"]["cache_dir"]

    tracker = CostTracker(
        hard_cap_usd=cost_cfg.get("grid_ab_hard_cap_usd", cost_cfg["hard_cap_usd"]),
        price_in=cost_cfg["price_per_input_token"],
        price_out=cost_cfg["price_per_output_token"],
    )
    elicitor = Elicitor(cfg, cache_dir, tracker, max_concurrency=16)

    world_ids = range(id_range["base"], id_range["base"] + min(n_worlds, id_range["count"]))
    worlds = [worldgen.generate_world(master_seed, wid, cfg) for wid in world_ids]

    docs_by_world_root = {}
    for w in worlds:
        for root_id in range(len(REPORTS_PER_ROOT)):
            docs_by_world_root[(w.world_id, root_id)] = worldgen.generate_root_for_world(
                w, root_id, master_seed, cfg
            )

    tasks = build_tasks(worlds, docs_by_world_root, elicitor, elicit_cfg, pilot_cfg, raw_log_path)

    results: list[ElicitResult] = []
    halted_early = False
    try:
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            r = await coro
            results.append(r)
            if (i + 1) % 500 == 0:
                print(f"[grid_ab] {i + 1}/{len(tasks)} calls done, "
                      f"spent=${tracker.spent_usd:.2f}")
    except BudgetExceededError as e:
        halted_early = True
        print(f"[grid_ab] halted early: {e}")

    return {
        "worlds": worlds, "results": results,
        "cost_spent_usd": tracker.spent_usd, "halted_early": halted_early,
        "n_planned_calls": len(tasks),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=None, help="override W (default: config)")
    parser.add_argument("--dry-run-cost", action="store_true", help="print cost estimate and exit")
    args = parser.parse_args()

    cfg = load_config()
    n_worlds = args.worlds or cfg["world_id_ranges"]["grid_ab"]["count"]

    if args.dry_run_cost:
        est = estimate_cost(cfg, n_worlds)
        n_calls = n_worlds * sum(REPORTS_PER_ROOT)
        print(f"worlds={n_worlds} calls={n_calls} estimated_cost=${est:.2f} "
              f"hard_cap=${cfg['cost']['hard_cap_usd']}")
        return

    result = await run_grid_ab(cfg, n_worlds)
    print(f"\n[grid_ab] done: {len(result['results'])}/{result['n_planned_calls']} calls, "
          f"spent=${result['cost_spent_usd']:.4f}")
    if result["halted_early"]:
        print("[grid_ab] WARNING: halted early due to cost cap; results are partial.")


if __name__ == "__main__":
    asyncio.run(main())
