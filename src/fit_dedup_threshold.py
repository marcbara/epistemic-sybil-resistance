"""Fit and freeze the report-space dedup cosine threshold (Section 5.2) on
the calibration split's already-collected data. No new API calls -- reads
results/raw/calibration.jsonl (from calibration.py) and worldgen to
reconstruct per-world (theta, values, rationales), then updates
calibration_fit.json with the frozen threshold.

Usage:
    python src/fit_dedup_threshold.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import worldgen
import aggregate

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    master_seed = cfg["master_seed"]
    id_range = cfg["world_id_ranges"]["calibration"]
    world_ids = range(id_range["base"], id_range["base"] + id_range["count"])
    worlds = {wid: worldgen.generate_world(master_seed, wid, cfg) for wid in world_ids}

    recs = [json.loads(l) for l in open(ROOT / "results" / "raw" / "calibration.jsonl", encoding="utf-8")]
    reports = [r for r in recs if r["call_kind"] == "report" and r["valid"]]

    by_world: dict[int, dict] = {}
    for r in reports:
        wid = r["world_id"]
        by_world.setdefault(wid, {"values": [], "rationales": []})
        by_world[wid]["values"].append(r["parsed_estimate"])
        by_world[wid]["rationales"].append(r["parsed_rationale"] or "")

    worlds_data = [
        {"theta": worlds[wid].theta, "values": d["values"], "rationales": d["rationales"]}
        for wid, d in by_world.items()
    ]

    fit_path = ROOT / cfg["paths"]["processed_dir"] / "calibration_fit.json"
    fit = json.loads(fit_path.read_text())

    threshold = aggregate.fit_dedup_threshold(
        worlds_data,
        prior_mean=cfg["dgp"]["prior_mean"],
        prior_sd=cfg["dgp"]["prior_sd"],
        sigma_r2=fit["sigma_r2"],
        threshold_grid=cfg["aggregators"]["dedup"]["threshold_grid"],
    )
    fit["dedup_threshold"] = threshold
    fit_path.write_text(json.dumps(fit, indent=2))
    print(f"frozen dedup threshold: {threshold}")
    print(json.dumps(fit, indent=2))


if __name__ == "__main__":
    main()
