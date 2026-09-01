"""Figures for the 2x2 similarity x ancestry design."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.dpi": 150,
})
C_BAYES = "#1f4e79"
C_NAIVE = "#b03a2e"
C_GAMMA = "#4b7a3a"
C_LEGACY = "#7a5c00"
C_BALANCED = "#8a3ea3"


def fig11_tradeoff_curve():
    sweep = json.loads((ROOT / "results" / "processed" / "two_by_two_oracle_sweep.json").read_text())["sweep"]
    fmr = [r["false_merge_rate_b"] for r in sweep]
    fsr = [r["false_split_rate_c"] for r in sweep]
    thresholds = [r["threshold"] for r in sweep]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fmr, fsr, "o-", color=C_BAYES, lw=1.4, ms=4)
    for t, x, y in zip(thresholds, fmr, fsr):
        if t in (0.30, 0.50, 0.70, 0.80, 0.95):
            ax.annotate(f"t={t:.2f}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.plot(0, 0, "*", color="black", ms=12, label="Ideal (0,0)")
    legacy = next(r for r in sweep if r["threshold"] == 0.80)
    ax.plot(legacy["false_merge_rate_b"], legacy["false_split_rate_c"], "s", color=C_LEGACY, ms=8,
            label="Legacy threshold (0.80)")
    balanced_t = 0.70
    balanced = next(r for r in sweep if r["threshold"] == balanced_t)
    ax.plot(balanced["false_merge_rate_b"], balanced["false_split_rate_c"], "D", color=C_BALANCED, ms=8,
            label="Balanced-calibrated threshold (0.70)")
    ax.set_xlabel("False merge rate, cell B (independent roots, similar representation)")
    ax.set_ylabel("False split rate, cell C (shared root, dissimilar representation)")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="center right", fontsize=7.5)
    fig.tight_layout()
    out = ROOT / "figures" / "fig11_tradeoff_curve.pdf"
    fig.savefig(out); plt.close(fig)
    return out


def fig12_cluster_counts():
    ev = json.loads((ROOT / "results" / "processed" / "two_by_two_evaluation.json").read_text())
    mc = ev["mechanism"]["mean_inferred_clusters"]
    cells = ["A", "B", "C", "D"]
    true_roots = {"A": 1, "B": 4, "C": 1, "D": 4}
    inferred = {"A": mc["A_true_roots_1"], "B": mc["B_true_roots_4"],
                "C": mc["C_true_roots_1"], "D": mc["D_true_roots_4"]}

    x = np.arange(len(cells))
    width = 0.35
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.bar(x - width / 2, [true_roots[c] for c in cells], width, color="gray", alpha=0.6, label="True root count")
    ax.bar(x + width / 2, [inferred[c] for c in cells], width, color=C_BAYES, label="Mean inferred clusters")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({CELL_LABEL[c]})" for c in cells], fontsize=7.5)
    ax.set_ylabel("Count")
    fig.tight_layout()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              fontsize=8, frameon=False)
    fig.subplots_adjust(top=0.84)
    out = ROOT / "figures" / "fig12_cluster_counts.pdf"
    fig.savefig(out); plt.close(fig)
    return out


CELL_LABEL = {
    "A": "similar,\nshared root",
    "B": "similar,\nindep. roots",
    "C": "dissimilar,\nshared root",
    "D": "dissimilar,\nindep. roots",
}


def fig13_coverage_by_cell():
    ev = json.loads((ROOT / "results" / "processed" / "two_by_two_evaluation.json").read_text())
    am = ev["aggregator_metrics"]
    cells = ["A", "B", "C", "D"]
    aggs = [("naive", C_NAIVE, "s"), ("provenance", C_BAYES, "o"),
            ("dedup_legacy", C_LEGACY, "^"), ("dedup_balanced", C_BALANCED, "D")]

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    x = np.arange(len(cells))
    for i, (name, color, marker) in enumerate(aggs):
        vals = [am[name][c]["coverage"] for c in cells]
        ax.plot(x + (i - 1.5) * 0.06, vals, marker, color=color, ms=6, label=name, linestyle="none")
    ax.axhline(0.95, color="gray", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({CELL_LABEL[c]})" for c in cells], fontsize=7.5)
    ax.set_ylabel("Empirical 95% coverage")
    ax.set_ylim(0.6, 1.02)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    out = ROOT / "figures" / "fig13_coverage_by_cell.pdf"
    fig.savefig(out); plt.close(fig)
    return out


def main():
    for fn in [fig11_tradeoff_curve, fig12_cluster_counts, fig13_coverage_by_cell]:
        print(f"wrote {fn()}")


if __name__ == "__main__":
    main()
