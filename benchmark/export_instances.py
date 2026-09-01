"""Export the frozen Grid A / Grid B evaluation data as benchmark instances.

Reads the frozen raw model outputs (results/raw/grid_ab.jsonl) and the
world generator (src/worldgen.py, seeded by config.yaml's master_seed), and
writes two files under benchmark/data/:

  instances.jsonl  -- one line per (cell, world) evaluation instance. This is
                      the only file an evaluated aggregator may read.
  truth.jsonl      -- one line per world with the latent state Theta. Read
                      ONLY by the scorer (benchmark/score.py), never by the
                      aggregator under evaluation.

The instance pools are identical to the ones the paper's own analysis
(src/analyze.py) evaluates: Grid A takes nested prefixes of root 0's report
pool at n in {1,2,4,8,16,32}; Grid B takes m = 16/k reports from each of the
first k roots, k in {1,2,4,8,16}. Calibration parameters shipped inside each
instance come from the frozen calibration fit
(results/processed/calibration_fit.json), which was estimated on a disjoint
100-world calibration split -- never on these evaluation worlds.

Usage:
    python benchmark/export_instances.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import worldgen  # noqa: E402

GRID_A_NS = [1, 2, 4, 8, 16, 32]
GRID_B_KS = [1, 2, 4, 8, 16]
GRID_B_N = 16

DATA_DIR = ROOT / "benchmark" / "data"


def load_pool(cfg: dict) -> dict:
    """{world_id: {root_id: [(value, rationale), ...]}} deduped by cache_key,
    mirroring src/analyze.py's load_pool exactly."""
    path = ROOT / cfg["paths"]["results_dir"] / "grid_ab.jsonl"
    seen, pool = set(), {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["cache_key"] in seen:
            continue
        seen.add(r["cache_key"])
        if not r["valid"]:
            continue
        pool.setdefault(r["world_id"], {}).setdefault(r["root_id"], []).append(
            (r["parsed_estimate"], r["parsed_rationale"] or "")
        )
    return pool


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    fit = json.load(open(ROOT / "results" / "processed" / "calibration_fit.json"))
    pool = load_pool(cfg)

    prior = {"mean": cfg["dgp"]["prior_mean"], "sd": cfg["dgp"]["prior_sd"]}
    calibration = {
        "sigma_r2": fit["sigma_r2"],
        "sigma_hat2": fit["sigma_hat2"],
        "nu_hat2": fit["nu_hat2"],
        "dedup_threshold": fit["dedup_threshold"],
        "gamma_cal": fit["gamma_cal"],
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n_inst = 0
    with open(DATA_DIR / "instances.jsonl", "w", encoding="utf-8") as f:
        for n in GRID_A_NS:
            for wid in sorted(pool):
                roots = pool[wid]
                if 0 not in roots or len(roots[0]) < n:
                    continue
                reports = [
                    {"value": v, "rationale": r, "root": 0} for v, r in roots[0][:n]
                ]
                inst = {
                    "instance_id": f"A-n{n:02d}-w{wid}",
                    "grid": "A",
                    "cell": {"n": n},
                    "world_id": wid,
                    "prior": prior,
                    "calibration": calibration,
                    "reports": reports,
                }
                f.write(json.dumps(inst) + "\n")
                n_inst += 1
        for k in GRID_B_KS:
            m = GRID_B_N // k
            for wid in sorted(pool):
                roots = pool[wid]
                root_ids = list(range(k))
                if not all(rid in roots and len(roots[rid]) >= m for rid in root_ids):
                    continue
                reports = [
                    {"value": v, "rationale": r, "root": rid}
                    for rid in root_ids
                    for v, r in roots[rid][:m]
                ]
                inst = {
                    "instance_id": f"B-k{k:02d}-w{wid}",
                    "grid": "B",
                    "cell": {"k": k, "m": m},
                    "world_id": wid,
                    "prior": prior,
                    "calibration": calibration,
                    "reports": reports,
                }
                f.write(json.dumps(inst) + "\n")
                n_inst += 1

    master_seed = cfg["master_seed"]
    with open(DATA_DIR / "truth.jsonl", "w", encoding="utf-8") as f:
        for wid in sorted(pool):
            w = worldgen.generate_world(master_seed, wid, cfg)
            f.write(json.dumps({"world_id": wid, "theta": w.theta}) + "\n")

    print(f"Wrote {n_inst} instances and {len(pool)} truth records to {DATA_DIR}")


if __name__ == "__main__":
    main()
