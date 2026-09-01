"""Figures for Grid A/B, in make_figures.py's exact style (rcParams, sizes,
color convention) -- reuses paper/latex-src/make_figures.py's palette,
extended with one color for the dedup baseline (not present in the paper's
simulation figures, which only compare Bayes vs naive).

Usage:
    python src/figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import GRID_A_NS, GRID_B_KS

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
C_BAYES = "#1f4e79"    # dark blue -- provenance-aware (this paper's dependence-aware analogue)
C_NAIVE = "#b03a2e"    # brick red -- naive independent pooling
C_DEDUP = "#7a5c00"    # dark gold -- report-space dedup baseline (Section 5.2)


def load_analysis():
    return json.loads((ROOT / "results" / "processed" / "grid_ab_analysis.json").read_text())


def fig5_calibration(analysis):
    ns = GRID_A_NS
    naive = [analysis["grid_a"][str(n)]["naive"]["calibration_ratio"] for n in ns]
    prov = [analysis["grid_a"][str(n)]["provenance"]["calibration_ratio"] for n in ns]
    dedup = [analysis["grid_a"][str(n)]["dedup"]["calibration_ratio"] for n in ns]

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(ns, prov, "o-", color=C_BAYES, lw=1.4, ms=4, label="Provenance-aware")
    ax.plot(ns, naive, "s--", color=C_NAIVE, lw=1.4, ms=4, label="Naive independent pooling")
    ax.plot(ns, dedup, "^:", color=C_DEDUP, lw=1.4, ms=4, label="Report-space dedup")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_xlabel(r"Number of reports $n$ (one primitive root, $k=1$)")
    ax.set_ylabel(r"Calibration ratio $C=\mathrm{RMSE}/\mathrm{posterior\ s.d.}$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = ROOT / "figures" / "fig5_calibration_empirical.pdf"
    fig.savefig(out); plt.close(fig)
    return out


def fig6_logscore(analysis):
    ns = GRID_A_NS
    naive = [analysis["grid_a"][str(n)]["naive"]["mean_nll"] for n in ns]
    prov = [analysis["grid_a"][str(n)]["provenance"]["mean_nll"] for n in ns]
    dedup = [analysis["grid_a"][str(n)]["dedup"]["mean_nll"] for n in ns]

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(ns, prov, "o-", color=C_BAYES, lw=1.4, ms=4, label="Provenance-aware")
    ax.plot(ns, naive, "s--", color=C_NAIVE, lw=1.4, ms=4, label="Naive independent pooling")
    ax.plot(ns, dedup, "^:", color=C_DEDUP, lw=1.4, ms=4, label="Report-space dedup")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_xlabel(r"Number of reports $n$ (one primitive root, $k=1$)")
    ax.set_ylabel("Mean negative log score")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = ROOT / "figures" / "fig6_logscore_empirical.pdf"
    fig.savefig(out); plt.close(fig)
    return out


def fig7_roots(analysis):
    ks = GRID_B_KS
    naive_rmse = [analysis["grid_b"][str(k)]["naive"]["rmse"] for k in ks]
    prov_rmse = [analysis["grid_b"][str(k)]["provenance"]["rmse"] for k in ks]
    prov_nll = [analysis["grid_b"][str(k)]["provenance"]["mean_nll"] for k in ks]
    naive_nll_at_k1 = analysis["grid_b"]["1"]["naive"]["mean_nll"]

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(ks, prov_nll, "o-", color=C_BAYES, lw=1.4, ms=4,
            label="Provenance-aware mean NLL")
    ax.axhline(naive_nll_at_k1, color=C_NAIVE, lw=1.4, ls="--",
               label="Naive NLL (report count fixed, ignores $k$)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel(r"Number of independent primitive roots $k$ (fixed $n=16$)")
    ax.set_ylabel("Mean negative log score")
    fig.tight_layout()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1,
              fontsize=8.5, frameon=False)
    fig.subplots_adjust(top=0.80)
    out = ROOT / "figures" / "fig7_roots_info_empirical.pdf"
    fig.savefig(out); plt.close(fig)
    return out


C_GAMMA = "#4b7a3a"    # green -- provenance + correlated extraction (exploratory, Section 6.3)


def fig10_gamma_exploratory():
    """Exploratory extension figure, deliberately separate from the
    confirmatory fig5/fig6: naive vs provenance vs provenance+gamma_cal on
    Grid A, coverage (left) and calibration ratio (right). Data from
    gamma_analysis.json; all gamma-aggregator parameters calibration-only."""
    ga = json.loads((ROOT / "results" / "processed" / "gamma_analysis.json").read_text())
    ns = GRID_A_NS
    cells = {n: ga["main"]["grid_a"][str(n)] for n in ns}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    for ax, metric, ylabel, ref in (
        (ax1, "coverage", "Empirical 95% coverage", 0.95),
        (ax2, "calibration_ratio", r"Calibration ratio $C$", 1.0),
    ):
        ax.plot(ns, [cells[n]["provenance_gamma"][metric] for n in ns], "D-",
                color=C_GAMMA, lw=1.4, ms=4, label=r"Provenance + $\gamma_{\mathrm{cal}}$ (exploratory)")
        ax.plot(ns, [cells[n]["provenance"][metric] for n in ns], "o-",
                color=C_BAYES, lw=1.4, ms=4, label="Provenance-aware")
        ax.plot(ns, [cells[n]["naive"][metric] for n in ns], "s--",
                color=C_NAIVE, lw=1.4, ms=4, label="Naive independent pooling")
        ax.axhline(ref, color="gray", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns); ax.set_xticklabels(ns)
        ax.set_xlabel(r"Number of reports $n$ ($k=1$)")
        ax.set_ylabel(ylabel)
    ax1.legend(loc="lower left", fontsize=7.5)

    fig.tight_layout()
    out = ROOT / "figures" / "fig10_gamma_exploratory.pdf"
    fig.savefig(out); plt.close(fig)
    return out


def main():
    analysis = load_analysis()
    paths = [fig5_calibration(analysis), fig6_logscore(analysis), fig7_roots(analysis),
             fig10_gamma_exploratory()]
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
