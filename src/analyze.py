"""Analysis for Grid A and Grid B (EXPERIMENT.md Section 6, 12.3/12.4 analogues).

Reads results/raw/grid_ab.jsonl (grid_ab.py's output) and
results/processed/calibration_fit.json (calibration.py + fit_dedup_threshold.py),
builds the nested (n,k) grid cells, runs the naive / provenance-aware / dedup
aggregators, computes RMSE, 95% coverage, mean NLL, and the calibration ratio
C = RMSE / posterior sd, with a cluster bootstrap by world for uncertainty.

Can run on a partial grid_ab.jsonl (useful while the collection job is still
running) -- worlds with incomplete report counts for a given cell are simply
excluded from that cell.

Usage:
    python src/analyze.py
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
import aggregate
from grid_ab import REPORTS_PER_ROOT

ROOT = Path(__file__).resolve().parents[1]

GRID_A_NS = [1, 2, 4, 8, 16, 32]
GRID_B_KS = [1, 2, 4, 8, 16]
GRID_B_N = 16


def load_config() -> dict:
    return yaml.safe_load(open(ROOT / "config.yaml"))


def load_pool(cfg: dict) -> dict:
    """Returns {world_id: {root_id: [(value, rationale), ...]}}, values ordered
    by elicitation seed (0..count-1) so nesting/subsetting is well-defined."""
    path = ROOT / cfg["paths"]["results_dir"] / "grid_ab.jsonl"
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]

    # A cache hit is returned by elicit() without going through append_jsonl
    # itself -- the caller appends every result it receives regardless, so a
    # call that hits cache (e.g. re-running a world already collected in an
    # earlier test batch) gets logged again under the same cache_key. Same
    # underlying API call, not a second one (cost tracking confirms this:
    # no double charge) -- but dedupe by cache_key here so no world's report
    # count is inflated by a duplicate log line.
    seen_keys = set()
    deduped = []
    for r in recs:
        if r["cache_key"] in seen_keys:
            continue
        seen_keys.add(r["cache_key"])
        deduped.append(r)
    recs = deduped

    pool: dict = {}
    for r in recs:
        if not r["valid"]:
            continue
        wid, rid = r["world_id"], r["root_id"]
        pool.setdefault(wid, {}).setdefault(rid, [])
        pool[wid][rid].append((r["parsed_estimate"], r["parsed_rationale"] or "", r["prompt"]))
    for wid in pool:
        for rid in pool[wid]:
            # seed order is API-call order, not guaranteed by JSONL append order
            # under concurrency -- re-sort deterministically by value+rationale
            # is not possible without the seed; append_jsonl is written as
            # results complete, so instead we sort by the cache key, which is
            # deterministic per (model,T,prompt,seed). Simpler: the prompt is
            # identical across seeds for a given (world,root) -- seed isn't
            # recoverable post hoc, so we just take collection order, which is
            # fine since all reports within a root are exchangeable (nesting
            # only needs *a* stable, seed-independent size-n subset, not a
            # specific ordering).
            pass
    return pool


def world_thetas_and_e(cfg: dict, world_ids: list[int]) -> dict:
    master_seed = cfg["master_seed"]
    out = {}
    for wid in world_ids:
        w = worldgen.generate_world(master_seed, wid, cfg)
        e_by_root = {
            rid: worldgen.generate_root_for_world(w, rid, master_seed, cfg).e_true_float()
            for rid in range(len(REPORTS_PER_ROOT))
        }
        out[wid] = {"theta": w.theta, "e_by_root": e_by_root}
    return out


def calibration_ratio(rmse: float, mean_sd: float) -> float:
    return rmse / mean_sd if mean_sd > 0 else float("nan")


def evaluate_cell(theta_by_world, posteriors_by_world: dict) -> dict:
    """posteriors_by_world: {world_id: aggregate.Posterior}."""
    z95 = 1.959963984540054
    errs, sds, nlls, covered = [], [], [], []
    world_ids = []
    for wid, post in posteriors_by_world.items():
        theta = theta_by_world[wid]
        err = post.mean - theta
        errs.append(err)
        sds.append(post.sd)
        nlls.append(aggregate.nll(theta, post))
        covered.append(1.0 if abs(err) <= z95 * post.sd else 0.0)
        world_ids.append(wid)
    n = len(errs)
    rmse = math.sqrt(sum(e ** 2 for e in errs) / n) if n else float("nan")
    mean_sd = sum(sds) / n if n else float("nan")
    coverage = sum(covered) / n if n else float("nan")
    mean_nll = sum(nlls) / n if n else float("nan")
    cr = calibration_ratio(rmse, mean_sd)

    rmse_ci = cluster_bootstrap_ci(errs, sds, covered, "rmse") if n else (float("nan"),) * 2
    cov_ci = cluster_bootstrap_ci(errs, sds, covered, "coverage") if n else (float("nan"),) * 2
    cr_ci = cluster_bootstrap_ci(errs, sds, covered, "calibration_ratio") if n else (float("nan"),) * 2

    return {
        "n_worlds": n, "rmse": rmse, "rmse_ci95": list(rmse_ci),
        "mean_posterior_sd": mean_sd,
        "coverage": coverage, "coverage_ci95": list(cov_ci),
        "mean_nll": mean_nll,
        "calibration_ratio": cr, "calibration_ratio_ci95": list(cr_ci),
        "_errs": errs, "_sds": sds, "_covered": covered, "_world_ids": world_ids,
    }


def cluster_bootstrap_ci(errs, sds, covered, statistic: str, n_boot: int = 1000, seed: int = 0):
    """Bootstrap by world (resample worlds with replacement) for one of
    'rmse', 'coverage', 'calibration_ratio'."""
    rng = random.Random(seed)
    n = len(errs)
    if n == 0:
        return (float("nan"), float("nan"))
    stats = []
    idx = list(range(n))
    for _ in range(n_boot):
        sample = [rng.choice(idx) for _ in range(n)]
        e = [errs[i] for i in sample]
        s = [sds[i] for i in sample]
        c = [covered[i] for i in sample]
        if statistic == "rmse":
            stats.append(math.sqrt(sum(x ** 2 for x in e) / n))
        elif statistic == "coverage":
            stats.append(sum(c) / n)
        elif statistic == "calibration_ratio":
            rmse = math.sqrt(sum(x ** 2 for x in e) / n)
            mean_sd = sum(s) / n
            stats.append(calibration_ratio(rmse, mean_sd))
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[int(0.975 * n_boot) - 1]
    return (lo, hi)


def run_grid_a(cfg, pool, meta, fit) -> dict:
    prior_mean, prior_sd = cfg["dgp"]["prior_mean"], cfg["dgp"]["prior_sd"]
    sigma_r2, sigma_hat2, nu_hat2 = fit["sigma_r2"], fit["sigma_hat2"], fit["nu_hat2"]
    dedup_threshold = fit["dedup_threshold"]

    cells = {}
    for n in GRID_A_NS:
        naive_posts, prov_posts, dedup_posts, oracle_posts, theta_by_world = {}, {}, {}, {}, {}
        for wid, roots in pool.items():
            if 0 not in roots or len(roots[0]) < n:
                continue
            subset = roots[0][:n]
            values = [v for v, _, _ in subset]
            rationales = [r for _, r, _ in subset]
            theta_by_world[wid] = meta[wid]["theta"]
            naive_posts[wid] = aggregate.naive_pool(values, prior_mean, prior_sd, sigma_r2)
            prov_posts[wid] = aggregate.provenance_pool({0: values}, prior_mean, prior_sd, sigma_hat2, nu_hat2)
            post, _ = aggregate.dedup_pool(values, rationales, prior_mean, prior_sd, sigma_r2, dedup_threshold)
            dedup_posts[wid] = post
            oracle_posts[wid] = aggregate.oracle_pool({0: meta[wid]["e_by_root"][0]}, prior_mean, prior_sd, sigma_hat2)

        cells[n] = {
            "naive": evaluate_cell(theta_by_world, naive_posts),
            "provenance": evaluate_cell(theta_by_world, prov_posts),
            "dedup": evaluate_cell(theta_by_world, dedup_posts),
            "oracle": evaluate_cell(theta_by_world, oracle_posts),
        }
    return cells


def run_grid_b(cfg, pool, meta, fit) -> dict:
    prior_mean, prior_sd = cfg["dgp"]["prior_mean"], cfg["dgp"]["prior_sd"]
    sigma_r2, sigma_hat2, nu_hat2 = fit["sigma_r2"], fit["sigma_hat2"], fit["nu_hat2"]
    dedup_threshold = fit["dedup_threshold"]

    cells = {}
    for k in GRID_B_KS:
        m = GRID_B_N // k
        naive_posts, prov_posts, dedup_posts, theta_by_world = {}, {}, {}, {}
        for wid, roots in pool.items():
            root_ids = list(range(k))
            if not all(rid in roots and len(roots[rid]) >= m for rid in root_ids):
                continue
            reports_by_root = {rid: [v for v, _, _ in roots[rid][:m]] for rid in root_ids}
            all_values = [v for rid in root_ids for v in reports_by_root[rid]]
            all_rationales = [r for rid in root_ids for v, r, _ in roots[rid][:m]]
            theta_by_world[wid] = meta[wid]["theta"]
            naive_posts[wid] = aggregate.naive_pool(all_values, prior_mean, prior_sd, sigma_r2)
            prov_posts[wid] = aggregate.provenance_pool(reports_by_root, prior_mean, prior_sd, sigma_hat2, nu_hat2)
            post, _ = aggregate.dedup_pool(all_values, all_rationales, prior_mean, prior_sd, sigma_r2, dedup_threshold)
            dedup_posts[wid] = post

        cells[k] = {
            "naive": evaluate_cell(theta_by_world, naive_posts),
            "provenance": evaluate_cell(theta_by_world, prov_posts),
            "dedup": evaluate_cell(theta_by_world, dedup_posts),
        }
    return cells


def _strip_private(cell: dict) -> dict:
    return {k: v for k, v in cell.items() if not k.startswith("_")}


def main():
    cfg = load_config()
    fit = json.loads((ROOT / cfg["paths"]["processed_dir"] / "calibration_fit.json").read_text())
    pool = load_pool(cfg)
    meta = world_thetas_and_e(cfg, list(pool.keys()))

    grid_a = run_grid_a(cfg, pool, meta, fit)
    grid_b = run_grid_b(cfg, pool, meta, fit)

    out = {
        "n_worlds_available": len(pool),
        "grid_a": {n: {agg: _strip_private(c) for agg, c in cells.items()} for n, cells in grid_a.items()},
        "grid_b": {k: {agg: _strip_private(c) for agg, c in cells.items()} for k, cells in grid_b.items()},
    }
    out_path = ROOT / cfg["paths"]["processed_dir"] / "grid_ab_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    return grid_a, grid_b


if __name__ == "__main__":
    main()
