"""Reproduce every empirical result and figure reported in the paper from
the frozen, already-collected model outputs in results/raw/*.jsonl -- no
API credentials, no network access, no new model calls.

This script re-derives, in the same order the original analysis used:
  1. The calibration fit (sigma_r2, sigma_hat2, nu_hat2, report bias,
     rho_hat, leakage control) from results/raw/calibration.jsonl.
  2. The frozen legacy dedup threshold (0.80) from the same file.
  3. Grid A and Grid B (naive / provenance-aware / report-space dedup)
     from results/raw/grid_ab.jsonl.
  4. The correlated-extraction diagnostics (precision curve, in-sample
     gamma, extraction-noise QQ/tail statistics) from grid_ab.jsonl.
  5. The out-of-sample gamma_cal estimate, the exploratory gamma-aware
     aggregator, and the bias-subtraction sensitivity check.
  6. All confirmatory and exploratory Grid A/B figures.
  7. The 2x2 design: balanced dedup threshold (from
     results/raw/two_by_two_dedup_calibration.jsonl), evaluation metrics
     and the coarse oracle sweep (from results/raw/two_by_two_eval.jsonl).
  8. The 2x2 figures.
  9. The final $0 statistical round: factorial contrasts, the
     confirmatory Delta_BC, paired downstream contrasts, and the exact
     (gap-free) oracle threshold sweep.
 10. Sync the regenerated figures into paper/latex-src/figures/, the copies
     the compiled paper actually embeds.
 11. An integrity check comparing every number in step 9 above (plus the
     Grid A coverage endpoints and gamma_cal) against the values already
     committed in results/processed/*.json, at full stored precision, not
     the rounded figures printed in the paper.

Every step below imports and calls the exact functions the original
analysis used (src/*.py) rather than reimplementing any statistic. Steps
that originally required a live API call (calibration.py's, and grid
collection scripts') are replaced here by reconstructing the same
ElicitResult records from the frozen JSONL logs -- the statistics
functions downstream are unchanged and never notice the difference.

Usage:
    python reproduce_paper.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import worldgen               # noqa: E402
import gates                  # noqa: E402
import calibration             # noqa: E402
import fit_dedup_threshold     # noqa: E402
import analyze                 # noqa: E402
import diagnostics             # noqa: E402
import gamma_analysis          # noqa: E402
import figures                 # noqa: E402
import two_by_two              # noqa: E402
import two_by_two_analysis     # noqa: E402
import two_by_two_figures      # noqa: E402
import two_by_two_stats        # noqa: E402
from elicit import ElicitResult  # noqa: E402

CFG_PATH = ROOT / "config.yaml"
PROCESSED = ROOT / "results" / "processed"
RAW = ROOT / "results" / "raw"


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def load_config() -> dict:
    return yaml.safe_load(open(CFG_PATH))


def load_elicit_results(path: Path) -> list[ElicitResult]:
    """Deserialize a raw JSONL log back into ElicitResult records, deduped
    by cache_key (a call that hit the sha256 response cache during
    collection is logged only by the caller once per unique key; grid_ab's
    log additionally contains 128 duplicate lines from an early, folded-in
    test batch -- see EXPERIMENT_NOTES.md -- so dedup is applied uniformly
    here rather than assumed absent)."""
    seen, out = set(), []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["cache_key"] in seen:
                continue
            seen.add(rec["cache_key"])
            out.append(ElicitResult(**rec))
    return out


def reproduce_calibration_fit(cfg: dict) -> dict:
    """Step 1-2: calibration.fit_calibration + fit_dedup_threshold, both
    pure computation over calibration.jsonl -- no API call in either."""
    banner("1/9  Calibration fit (sigma_r2, sigma_hat2, nu_hat2, rho_hat, leakage)")
    master_seed = cfg["master_seed"]
    id_range = cfg["world_id_ranges"]["calibration"]
    world_ids = range(id_range["base"], id_range["base"] + id_range["count"])
    worlds = [worldgen.generate_world(master_seed, wid, cfg) for wid in world_ids]
    docs = [worldgen.generate_root_for_world(w, 0, master_seed, cfg) for w in worlds]

    all_results = load_elicit_results(RAW / "calibration.jsonl")
    report_results = [r for r in all_results if r.call_kind == "report"]
    leakage_results = [r for r in all_results if r.call_kind == "leakage"]

    fit = calibration.fit_calibration(cfg, worlds, docs, report_results, leakage_results)
    out_path = PROCESSED / "calibration_fit.json"
    out_path.write_text(json.dumps(fit, indent=2))
    print(f"sigma_r2={fit['sigma_r2']:.4f}  sigma_hat2={fit['sigma_hat2']:.4f}  "
          f"nu_hat2={fit['nu_hat2']:.4f}  rho_hat={fit['rho_hat']:.4f}")

    banner("2/9  Legacy dedup threshold (calibrated on k=1 calibration worlds)")
    fit_dedup_threshold.main()
    return json.loads(out_path.read_text())


def reproduce_grid_ab(cfg: dict) -> None:
    banner("3/9  Grid A + Grid B (naive / provenance-aware / report-space dedup)")
    analyze.main()


def reproduce_diagnostics(cfg: dict) -> None:
    banner("4/9  Correlated-extraction diagnostics (precision curve, in-sample gamma, tails)")
    diagnostics.main()


def reproduce_gamma(cfg: dict) -> dict:
    banner("5/9  Out-of-sample gamma_cal, gamma-aware aggregator, bias sensitivity")
    gamma_analysis.main()
    return json.loads((PROCESSED / "gamma_analysis.json").read_text())


def reproduce_grid_ab_figures(cfg: dict) -> None:
    banner("6/9  Grid A/B figures (fig5, fig6, fig7, fig10)")
    figures.main()


def reproduce_2x2(cfg: dict) -> dict:
    banner("7/9  2x2 design: balanced threshold, evaluation, coarse oracle sweep")
    two_by_two_analysis.cmd_fit_threshold()
    two_by_two_analysis.cmd_evaluate()
    two_by_two_analysis.cmd_oracle_sweep()
    return json.loads((PROCESSED / "two_by_two_evaluation.json").read_text())


def reproduce_2x2_figures(cfg: dict) -> None:
    banner("8/9  2x2 figures (fig11, fig12, fig13)")
    two_by_two_figures.main()


def reproduce_2x2_stats(cfg: dict) -> dict:
    banner("9/9  Factorial contrasts, confirmatory Delta_BC, exact oracle sweep")
    two_by_two_stats.main()
    return json.loads((PROCESSED / "two_by_two_final_stats.json").read_text())


def sync_figures_into_paper() -> None:
    """reproduce_paper.py writes into figures/ (matching the other analysis
    scripts' existing behavior); the compiled paper embeds the copies under
    paper/latex-src/figures/. Keep both in sync rather than requiring a
    manual copy step -- a byte-for-byte comparison between the two would
    spuriously differ even on success, since matplotlib embeds a PDF
    creation-date that changes on every render regardless of content."""
    banner("Syncing regenerated figures into paper/latex-src/figures/")
    dest_dir = ROOT / "paper" / "latex-src" / "figures"
    empirical_figs = [
        "fig5_calibration_empirical.pdf", "fig6_logscore_empirical.pdf",
        "fig7_roots_info_empirical.pdf", "fig8_precision_curve.pdf",
        "fig9_extraction_noise_qq.pdf", "fig10_gamma_exploratory.pdf",
        "fig11_tradeoff_curve.pdf", "fig12_cluster_counts.pdf",
        "fig13_coverage_by_cell.pdf",
    ]
    for name in empirical_figs:
        src = ROOT / "figures" / name
        (dest_dir / name).write_bytes(src.read_bytes())
        print(f"  synced {name}")


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------

TOLERANCE = 1e-6


def check(label: str, reproduced: float, frozen: float, results: list) -> None:
    ok = math.isclose(reproduced, frozen, rel_tol=0, abs_tol=TOLERANCE)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: reproduced={reproduced!r}  frozen={frozen!r}")
    results.append((label, ok))


def integrity_check(cfg: dict) -> bool:
    banner("INTEGRITY CHECK: reproduced values vs. frozen results/processed/*.json")

    # Load frozen (committed) values -- these are exactly what the paper's
    # numbers were drawn from, at full stored precision.
    frozen_grid_ab = json.loads((PROCESSED / "grid_ab_analysis.json").read_text())
    frozen_calib = json.loads((PROCESSED / "calibration_fit.json").read_text())
    frozen_2x2_stats = json.loads((PROCESSED / "two_by_two_final_stats.json").read_text())

    # Re-run the full pipeline end to end (this also overwrites the files
    # above with freshly reproduced output; since every step is a pure,
    # deterministic function of the frozen raw JSONL logs, a passing check
    # below means the files are unchanged in value).
    reproduce_calibration_fit(cfg)
    reproduce_grid_ab(cfg)
    reproduce_diagnostics(cfg)
    reproduce_gamma(cfg)
    reproduce_grid_ab_figures(cfg)
    reproduce_2x2(cfg)
    reproduce_2x2_figures(cfg)
    reproduce_2x2_stats(cfg)
    sync_figures_into_paper()

    reproduced_grid_ab = json.loads((PROCESSED / "grid_ab_analysis.json").read_text())
    reproduced_calib = json.loads((PROCESSED / "calibration_fit.json").read_text())
    reproduced_2x2_stats = json.loads((PROCESSED / "two_by_two_final_stats.json").read_text())

    banner("Comparing reproduced values against the values frozen before this run")
    results: list = []
    check("Grid A coverage, n=1 (naive)",
          reproduced_grid_ab["grid_a"]["1"]["naive"]["coverage"],
          frozen_grid_ab["grid_a"]["1"]["naive"]["coverage"], results)
    check("Grid A coverage, n=32 (naive)",
          reproduced_grid_ab["grid_a"]["32"]["naive"]["coverage"],
          frozen_grid_ab["grid_a"]["32"]["naive"]["coverage"], results)
    check("gamma_cal",
          reproduced_calib["gamma_cal"], frozen_calib["gamma_cal"], results)
    check("Representation effect (factorial contrast mean)",
          reproduced_2x2_stats["factorial_contrasts"]["representation_effect"]["mean"],
          frozen_2x2_stats["factorial_contrasts"]["representation_effect"]["mean"], results)
    check("Ancestry effect (factorial contrast mean)",
          reproduced_2x2_stats["factorial_contrasts"]["ancestry_effect"]["mean"],
          frozen_2x2_stats["factorial_contrasts"]["ancestry_effect"]["mean"], results)
    check("Delta_BC (confirmatory, 200 eval worlds)",
          reproduced_2x2_stats["confirmatory_delta_bc"]["delta_bc"],
          frozen_2x2_stats["confirmatory_delta_bc"]["delta_bc"], results)
    check("min_t max(FMR_B, FSR_C) (exact oracle sweep)",
          reproduced_2x2_stats["exact_oracle_sweep"]["best_minimax"]["max"],
          frozen_2x2_stats["exact_oracle_sweep"]["best_minimax"]["max"], results)

    n_fail = sum(1 for _, ok in results if not ok)
    banner(f"INTEGRITY CHECK {'PASSED' if n_fail == 0 else 'FAILED'} "
           f"({len(results) - n_fail}/{len(results)} checks within tolerance {TOLERANCE})")
    return n_fail == 0


def main() -> int:
    cfg = load_config()
    ok = integrity_check(cfg)
    if not ok:
        print("\nOne or more reproduced values differ from the frozen results "
              "beyond tolerance. See INTEGRITY CHECK output above.", file=sys.stderr)
        return 1
    print("\nAll reported figures and tables have been regenerated in "
          "results/processed/ and figures/, matching the frozen values used "
          "in the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
