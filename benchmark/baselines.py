"""Reference baseline adapters: minimal integration examples.

Each baseline is a function instance -> (mean, sd). This is the full
contract an external aggregator must satisfy; see benchmark/README.md for
the track rules (which instance fields a report-only vs. provenance-track
method may read).

Running this module produces one predictions file per baseline under
benchmark/, then you can score them:

    python benchmark/baselines.py
    python benchmark/score.py benchmark/predictions_naive.jsonl
    python benchmark/score.py benchmark/predictions_provenance.jsonl

`naive` (report-only track) and `provenance` (provenance track) are the same
aggregators evaluated in the paper (src/aggregate.py), wired through the
benchmark interface; their scored metrics reproduce the paper's Grid A /
Grid B tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import aggregate  # noqa: E402

DATA_DIR = ROOT / "benchmark" / "data"


def naive_baseline(inst: dict) -> tuple[float, float]:
    """Report-only track: reads report values only; assumes independence."""
    values = [r["value"] for r in inst["reports"]]
    post = aggregate.naive_pool(
        values,
        inst["prior"]["mean"],
        inst["prior"]["sd"],
        inst["calibration"]["sigma_r2"],
    )
    return post.mean, post.sd


def provenance_baseline(inst: dict) -> tuple[float, float]:
    """Provenance track: additionally reads each report's root label and
    applies the shared-root block precision (paper, Proposition 2)."""
    by_root: dict = {}
    for r in inst["reports"]:
        by_root.setdefault(r["root"], []).append(r["value"])
    post = aggregate.provenance_pool(
        by_root,
        inst["prior"]["mean"],
        inst["prior"]["sd"],
        inst["calibration"]["sigma_hat2"],
        inst["calibration"]["nu_hat2"],
    )
    return post.mean, post.sd


BASELINES = {
    "naive": naive_baseline,
    "provenance": provenance_baseline,
}


def main() -> None:
    instances = [
        json.loads(line)
        for line in open(DATA_DIR / "instances.jsonl", encoding="utf-8")
    ]
    for name, fn in BASELINES.items():
        out = ROOT / "benchmark" / f"predictions_{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for inst in instances:
                mean, sd = fn(inst)
                f.write(
                    json.dumps(
                        {"instance_id": inst["instance_id"], "mean": mean, "sd": sd}
                    )
                    + "\n"
                )
        print(f"Wrote {len(instances)} predictions to {out}")


if __name__ == "__main__":
    main()
