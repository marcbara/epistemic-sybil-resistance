"""2x2 similarity x ancestry design (prediction 13.4).

Paired by world: every world generates 4 roots (root_id 0..3), each an
independent draw E_r = Theta + eps_r from the same world's Theta. Ancestry
is manipulated by which roots feed a cell's 4 reports.

Representation similarity is NOT manipulated at elicitation time -- see
config.yaml's two_by_two block and src/render.py's module docstring for why
(a pilot attempt at prompt-level style phrasing changed extraction RMSE by
0.80x, a real confound). Instead: ONE frozen elicitation (the same prompt
used everywhere, `elicit.report_system_prompt("S1")`, byte-identical to the
frozen A/B prompt) produces estimate + raw_rationale; a deterministic
post-hoc renderer (never reads the estimate or raw rationale, so it cannot
move the estimate) produces the S1/S2 text actually embedded.

    A (similar,    shared root):       root0 x4, rendered S1,S1,S1,S1
    B (similar,    independent roots): root0-3 x1, rendered S1,S1,S1,S1
    C (dissimilar, shared root):       root0 x4, rendered S1,S1,S2,S2
    D (dissimilar, independent roots): root0-3 x1, rendered S1,S1,S2,S2

Per-world call plan (7 calls -- one elicitation per root/seed slot needed,
no elicitation-time style variants, cheaper than the earlier design):

    root0: seeds 0-3 @ frozen prompt  -> 4 calls
    root1: seed 0    @ frozen prompt  -> 1 call
    root2: seed 0    @ frozen prompt  -> 1 call
    root3: seed 0    @ frozen prompt  -> 1 call

Usage:
    python src/two_by_two.py pilot            # 30 worlds, prints gate report
    python src/two_by_two.py collect --range two_by_two_dedup_calibration
    python src/two_by_two.py collect --range two_by_two_eval --dry-run-cost
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import worldgen
import gates
import aggregate
import embed
import render
from elicit import Elicitor, CostTracker, BudgetExceededError, ElicitResult

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config.yaml"

# (root_id, seed) -- single frozen elicitation, no style parameter
CALL_PLAN = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0)]
assert len(CALL_PLAN) == 7

# cell -> list of (root_id, seed, render_style)
CELLS = {
    "A": [(0, 0, "S1"), (0, 1, "S1"), (0, 2, "S1"), (0, 3, "S1")],
    "B": [(0, 0, "S1"), (1, 0, "S1"), (2, 0, "S1"), (3, 0, "S1")],
    "C": [(0, 0, "S1"), (0, 1, "S1"), (0, 2, "S2"), (0, 3, "S2")],
    "D": [(0, 0, "S1"), (1, 0, "S1"), (2, 0, "S2"), (3, 0, "S2")],
}
CELL_DESIGN = {
    "A": {"similarity": "similar", "ancestry": "shared_root"},
    "B": {"similarity": "similar", "ancestry": "independent_roots"},
    "C": {"similarity": "dissimilar", "ancestry": "shared_root"},
    "D": {"similarity": "dissimilar", "ancestry": "independent_roots"},
}


def load_config() -> dict:
    return yaml.safe_load(open(CFG_PATH))


def build_tasks(worlds, docs_by_world_root, elicitor, elicit_cfg, temperature, jsonl_path):
    tasks = []
    for w in worlds:
        for root_id, seed in CALL_PLAN:
            doc = docs_by_world_root[(w.world_id, root_id)]
            tasks.append(
                elicitor.elicit(
                    world_id=w.world_id, root_id=root_id, call_kind="report",
                    document_text=doc.text, model=elicit_cfg["main_model"],
                    temperature=temperature, seed=seed, jsonl_path=jsonl_path,
                    # rationale_style defaults to "S1" == the frozen A/B prompt;
                    # not varied here, representation is manipulated post-hoc.
                )
            )
    return tasks


async def collect(cfg: dict, range_name: str, n_worlds: int, cost_cap_key: str = "hard_cap_usd",
                   max_concurrency: int = 12) -> dict:
    id_range = cfg["world_id_ranges"][range_name]
    elicit_cfg = cfg["elicitation"]
    cost_cfg = cfg["cost"]
    master_seed = cfg["master_seed"]
    temperature = cfg["pilot"]["temperature"]

    raw_log_path = ROOT / cfg["paths"]["results_dir"] / f"{range_name}.jsonl"
    cache_dir = ROOT / cfg["paths"]["cache_dir"]

    tracker = CostTracker(
        hard_cap_usd=cost_cfg.get(cost_cap_key, cost_cfg["hard_cap_usd"]),
        price_in=cost_cfg["price_per_input_token"],
        price_out=cost_cfg["price_per_output_token"],
    )
    elicitor = Elicitor(cfg, cache_dir, tracker, max_concurrency=max_concurrency)

    world_ids = range(id_range["base"], id_range["base"] + min(n_worlds, id_range["count"]))
    worlds = [worldgen.generate_world(master_seed, wid, cfg) for wid in world_ids]

    docs_by_world_root = {}
    for w in worlds:
        for root_id in range(cfg["two_by_two"]["k_roots_available"]):
            docs_by_world_root[(w.world_id, root_id)] = worldgen.generate_root_for_world(
                w, root_id, master_seed, cfg
            )

    tasks = build_tasks(worlds, docs_by_world_root, elicitor, elicit_cfg, temperature, raw_log_path)

    results: list[ElicitResult] = []
    halted_early = False
    try:
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            r = await coro
            results.append(r)
            if (i + 1) % 200 == 0:
                print(f"[two_by_two:{range_name}] {i + 1}/{len(tasks)} calls, "
                      f"spent=${tracker.spent_usd:.2f}")
    except BudgetExceededError as e:
        halted_early = True
        print(f"[two_by_two:{range_name}] halted early: {e}")

    return {
        "worlds": worlds, "docs_by_world_root": docs_by_world_root,
        "results": results, "cost_spent_usd": tracker.spent_usd,
        "halted_early": halted_early, "n_planned_calls": len(tasks),
    }


def load_pool(cfg: dict, range_name: str) -> dict:
    """{world_id: {(root_id, seed): estimate}} -- the single frozen
    elicitation's numeric output. Representation (rationale text) is never
    read from the raw log for cell construction; it's produced fresh by
    render.render_rationale, deterministically, per (root_id, seed)."""
    path = ROOT / cfg["paths"]["results_dir"] / f"{range_name}.jsonl"
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    seen, ded = set(), []
    for r in recs:
        if r["cache_key"] in seen:
            continue
        seen.add(r["cache_key"]); ded.append(r)

    from elicit import cache_key_for, report_system_prompt
    elicit_cfg = cfg["elicitation"]
    master_seed = cfg["master_seed"]

    by_world_root: dict = {}
    for r in ded:
        if not r["valid"]:
            continue
        by_world_root.setdefault((r["world_id"], r["root_id"]), []).append(r)

    worlds_needed = sorted({wid for wid, _ in by_world_root})
    worlds = {wid: worldgen.generate_world(master_seed, wid, cfg) for wid in worlds_needed}
    system = report_system_prompt("S1")  # the one frozen elicitation

    pool: dict = {}
    for wid in worlds_needed:
        w = worlds[wid]
        pool[wid] = {}
        for root_id, seed in CALL_PLAN:
            doc = worldgen.generate_root_for_world(w, root_id, master_seed, cfg)
            prompt = f"Memo:\n\n{doc.text}"
            key = cache_key_for(elicit_cfg["main_model"], cfg["pilot"]["temperature"],
                                 f"{system}\n{prompt}", seed)
            match = next((r for r in by_world_root.get((wid, root_id), []) if r["cache_key"] == key), None)
            if match is not None:
                pool[wid][(root_id, seed)] = match["parsed_estimate"]
    return pool


def build_cells(pool: dict) -> dict:
    """{world_id: {cell: [(value, rendered_rationale), ...]}}."""
    out = {}
    for wid, entries in pool.items():
        cell_data = {}
        for cell, keys in CELLS.items():
            if not all((root_id, seed) in entries for root_id, seed, _ in keys):
                continue
            cell_data[cell] = [
                (entries[(root_id, seed)], render.render_rationale(style, f"{wid}-{root_id}-{seed}"))
                for root_id, seed, style in keys
            ]
        out[wid] = cell_data
    return out


# ---------------------------------------------------------------------------
# Pilot gates
# ---------------------------------------------------------------------------

def gate_validity(results: list[ElicitResult]) -> dict:
    rate = gates.validity_rate([r.valid for r in results])
    return {"validity_rate": rate, "pass": rate >= 0.95}


def gate_render_style_separation(cells_by_world: dict, min_gap: float) -> dict:
    """Sanity check on the deterministic renderer itself (not the model):
    S1/S2 template pools should be clearly separable in embedding space."""
    within, cross = [], []
    for wid, cells in cells_by_world.items():
        for cell in ["A", "B", "C", "D"]:
            if cell not in cells:
                continue
            by_style: dict[str, list[str]] = {"S1": [], "S2": []}
            for value, text in cells[cell]:
                # style is recoverable from which template pool the text is in
                style = "S1" if text in render.S1_TEMPLATES else "S2"
                by_style[style].append(text)
            for s in ["S1", "S2"]:
                texts = by_style[s]
                vecs = embed.embed_texts(texts)
                for i in range(len(vecs)):
                    for j in range(i + 1, len(vecs)):
                        within.append(embed.cosine_similarity(vecs[i], vecs[j]))
            v1 = embed.embed_texts(by_style["S1"])
            v2 = embed.embed_texts(by_style["S2"])
            for a in v1:
                for b in v2:
                    cross.append(embed.cosine_similarity(a, b))
    mean_within = sum(within) / len(within) if within else float("nan")
    mean_cross = sum(cross) / len(cross) if cross else float("nan")
    gap = mean_within - mean_cross
    return {
        "mean_within_style_cosine": mean_within,
        "mean_cross_style_cosine": mean_cross,
        "gap": gap,
        "pass": gap > min_gap,
    }


def _mean_pairwise_cosine(texts: list[str]) -> list[float]:
    vecs = embed.embed_texts(texts)
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(embed.cosine_similarity(vecs[i], vecs[j]))
    return sims


def gate_delta_bc_inversion(cells_by_world: dict, min_margin: float, n_boot: int = 2000, seed: int = 0) -> dict:
    """Delta_BC = E[cos(cell B pairs)] - E[cos(cell C pairs)]. B is
    independent-roots-but-similar-representation; C is shared-root-but-
    dissimilar-representation. A positive Delta_BC means report-only
    similarity ranks B as MORE alike than C -- the exact inversion of what
    it should track if it tracked evidential ancestry. Per-world mean
    pairwise cosine (not raw pairs) is the unit of the cluster bootstrap,
    since pairs within a world are not independent."""
    b_world_means, c_world_means = [], []
    for wid, cells in cells_by_world.items():
        if "B" in cells:
            sims = _mean_pairwise_cosine([t for _, t in cells["B"]])
            if sims:
                b_world_means.append(sum(sims) / len(sims))
        if "C" in cells:
            sims = _mean_pairwise_cosine([t for _, t in cells["C"]])
            if sims:
                c_world_means.append(sum(sims) / len(sims))

    mean_b = sum(b_world_means) / len(b_world_means)
    mean_c = sum(c_world_means) / len(c_world_means)
    delta = mean_b - mean_c

    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        bb = [b_world_means[rng.randrange(len(b_world_means))] for _ in b_world_means]
        cc = [c_world_means[rng.randrange(len(c_world_means))] for _ in c_world_means]
        boots.append(sum(bb) / len(bb) - sum(cc) / len(cc))
    boots.sort()
    ci = [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]]

    return {
        "mean_cos_B": mean_b, "mean_cos_C": mean_c, "delta_bc": delta,
        "delta_bc_ci95": ci, "n_worlds_b": len(b_world_means), "n_worlds_c": len(c_world_means),
        "pass": delta > min_margin and ci[0] > 0,
    }


def run_pilot_gates(cfg: dict, collect_result: dict) -> str:
    pool = load_pool(cfg, "two_by_two_pilot")
    cells_by_world = build_cells(pool)
    tt_cfg = cfg["two_by_two"]

    g1 = gate_validity(collect_result["results"])
    g2 = gate_render_style_separation(cells_by_world, tt_cfg["render_style_separation_min_gap"])
    g3 = gate_delta_bc_inversion(cells_by_world, tt_cfg["delta_bc_min_margin"])

    lines = ["# 2x2 pilot report (redesigned: post-hoc deterministic rendering)\n\n"]
    lines.append(f"Worlds: {len(pool)}\n\n")

    lines.append("## Gate 1: validity >= 95%\n")
    lines.append(f"{g1}\n{'PASS' if g1['pass'] else 'FAIL'}\n\n")

    lines.append("## Gate 2: renderer template separation (sanity check, not model-dependent)\n")
    lines.append(f"gap > {tt_cfg['render_style_separation_min_gap']}\n")
    lines.append(f"{g2}\n{'PASS' if g2['pass'] else 'FAIL'}\n\n")

    lines.append("## Gate 3: Delta_BC inversion (threshold-independent mechanism gate)\n")
    lines.append(f"Delta_BC = E[cos(B)] - E[cos(C)], margin > {tt_cfg['delta_bc_min_margin']}, "
                 "bootstrap CI lower bound > 0\n")
    lines.append(f"{g3}\n{'PASS' if g3['pass'] else 'FAIL'}\n\n")

    lines.append("## Note: estimate-quality equivalence across styles\n")
    lines.append("Not applicable in this design -- there is a single frozen elicitation; "
                 "the estimate is identical regardless of which representation is rendered "
                 "downstream, by construction (render.py never reads the estimate).\n\n")

    all_pass = g1["pass"] and g2["pass"] and g3["pass"]
    lines.append(f"## Overall: {'ALL GATES PASS' if all_pass else 'SOME GATES FAILED'}\n")
    lines.append("\nSTOP here for human review before running the balanced dedup "
                 "calibration or the full 2x2 evaluation.\n")
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--range", required=True,
                            choices=["two_by_two_pilot", "two_by_two_dedup_calibration", "two_by_two_eval"])
    p_collect.add_argument("--worlds", type=int, default=None)
    p_collect.add_argument("--dry-run-cost", action="store_true")
    p_collect.add_argument("--cost-cap-key", default="hard_cap_usd")

    sub.add_parser("pilot")

    args = parser.parse_args()
    cfg = load_config()

    if args.cmd == "collect":
        n_worlds = args.worlds or cfg["world_id_ranges"][args.range]["count"]
        if args.dry_run_cost:
            n_calls = n_worlds * len(CALL_PLAN)
            est = n_calls * 0.001
            print(f"range={args.range} worlds={n_worlds} calls={n_calls} estimated_cost=${est:.2f}")
            return
        result = asyncio.run(collect(cfg, args.range, n_worlds, args.cost_cap_key))
        print(f"[two_by_two:{args.range}] done: {len(result['results'])}/{result['n_planned_calls']} calls, "
              f"spent=${result['cost_spent_usd']:.4f}")
        if result["halted_early"]:
            print(f"[two_by_two:{args.range}] WARNING: halted early; results partial.")

    elif args.cmd == "pilot":
        n_worlds = cfg["world_id_ranges"]["two_by_two_pilot"]["count"]
        result = asyncio.run(collect(cfg, "two_by_two_pilot", n_worlds))
        print(f"[two_by_two:pilot] collected {len(result['results'])}/{result['n_planned_calls']} calls, "
              f"spent=${result['cost_spent_usd']:.4f}\n")
        report = run_pilot_gates(cfg, result)
        out_path = ROOT / cfg["paths"]["processed_dir"] / "two_by_two_pilot_report.md"
        out_path.write_text(report, encoding="utf-8")
        print(report)


if __name__ == "__main__":
    main()
