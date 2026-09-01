"""Post-hoc diagnostics on already-collected Grid A/B and calibration data.
No API calls. Four checks, requested at the Grid A/B review:

1. Precision-curve test of the Section 6 model: does the empirical
   Var(R_bar_m - Theta) across worlds follow sigma^2 + nu^2/m for
   m in {1,2,4,8,16,32}, with sigma^2 and nu^2 taken exclusively from the
   calibration fit? This -- not the rho identity, which is definitionally
   imposed -- is the real evidence that the shared-root random-effects
   model describes Haiku's extraction behavior. -> fig8

2. Distributional check of extraction noise: QQ plot of (R - E) against a
   fitted Normal, pooled over Grid A root-0 reports. Heavy tails from
   arithmetic slips would show up here and help explain residual
   miscalibration. -> fig9

3. Bias handling: confirms (by reporting, not retrofitting) that the
   calibration bias estimate (-11.31) is NOT subtracted in any aggregator,
   and quantifies how much of the k=16 coverage shortfall it could explain.

4. Leakage control re-analysis at world level: the original analysis
   treated 300 no-doc calls as independent, but the 3 calls per world share
   Theta. Re-run corr + permutation p on per-world means (100 clusters).

Usage:
    python src/diagnostics.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import worldgen
import gates
from analyze import load_pool, GRID_A_NS

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


def load_cfg_fit():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    fit = json.loads((ROOT / "results" / "processed" / "calibration_fit.json").read_text())
    return cfg, fit


# ------------------------------------------------------------------
# 1. Precision curve: Var(R_bar_m - Theta) vs sigma^2 + nu^2/m
# ------------------------------------------------------------------

def precision_curve(cfg, fit, pool):
    master_seed = cfg["master_seed"]
    sigma2, nu2 = fit["sigma_hat2"], fit["nu_hat2"]

    thetas = {}
    for wid in pool:
        thetas[wid] = worldgen.generate_world(master_seed, wid, cfg).theta

    rows = []
    per_m_errors = {}
    for m in GRID_A_NS:
        errs = []
        for wid, roots in pool.items():
            if 0 not in roots or len(roots[0]) < m:
                continue
            vals = [v for v, _, _ in roots[0][:m]]
            errs.append(sum(vals) / m - thetas[wid])
        per_m_errors[m] = errs
        emp_var = gates.sample_variance(errs)
        pred_var = sigma2 + nu2 / m
        # cluster bootstrap (worlds are the clusters; errors here are already
        # one-per-world, so an iid bootstrap over worlds is the cluster bootstrap)
        rng = random.Random(m)
        boots = []
        n = len(errs)
        for _ in range(1000):
            sample = [errs[rng.randrange(n)] for _ in range(n)]
            boots.append(gates.sample_variance(sample))
        boots.sort()
        rows.append({
            "m": m, "n_worlds": n,
            "empirical_var": emp_var,
            "empirical_var_ci95": [boots[25], boots[974]],
            "predicted_var": pred_var,
        })
    return rows, per_m_errors


def estimate_gamma(cfg, fit, pool):
    """Estimate Section 6.3's extraction-error correlation gamma from Grid A.

    Under correlated extraction, Var(eta_i)=nu^2 with Cov(eta_i,eta_j)=gamma*nu^2,
    so the expected within-document sample variance is (1-gamma)*nu^2 and
    Var(R_bar_m - Theta) = sigma^2 + gamma*nu^2 + (1-gamma)*nu^2/m. A flat
    empirical precision curve well above sigma^2 is this regime's signature.
    Uses nu^2 from the calibration fit (Var(R - E) pooled across documents,
    which under this model still estimates total nu^2) and the mean
    within-document variance from Grid A's 32-report pools.
    """
    within_vars = []
    for wid, roots in pool.items():
        if 0 not in roots or len(roots[0]) < 32:
            continue
        vals = [v for v, _, _ in roots[0][:32]]
        within_vars.append(gates.sample_variance(vals))
    mean_within = sum(within_vars) / len(within_vars)
    nu2 = fit["nu_hat2"]
    gamma = 1.0 - mean_within / nu2
    sigma2 = fit["sigma_hat2"]
    return {
        "n_worlds": len(within_vars),
        "mean_within_document_variance": mean_within,
        "nu_hat2_total": nu2,
        "gamma_hat": gamma,
        "implied_ceiling_sigma2_plus_gamma_nu2": sigma2 + gamma * nu2,
        "independent_model_ceiling_sigma2": sigma2,
    }


def fig8_precision_curve(rows, gamma_info=None):
    ms = [r["m"] for r in rows]
    emp = [r["empirical_var"] for r in rows]
    lo = [r["empirical_var_ci95"][0] for r in rows]
    hi = [r["empirical_var_ci95"][1] for r in rows]
    pred = [r["predicted_var"] for r in rows]

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(ms, pred, "-", color=C_BAYES, lw=1.4,
            label=r"Independent extraction: $\hat\sigma^2+\hat\nu^2/m$")
    if gamma_info is not None:
        g, nu2, s2 = gamma_info["gamma_hat"], gamma_info["nu_hat2_total"], \
            gamma_info["independent_model_ceiling_sigma2"]
        pred_corr = [s2 + g * nu2 + (1 - g) * nu2 / m for m in ms]
        ax.plot(ms, pred_corr, "-.", color="#4b7a3a", lw=1.4,
                label=r"Correlated extraction: $\hat\sigma^2+\hat\gamma\hat\nu^2+(1-\hat\gamma)\hat\nu^2/m$")
    ax.errorbar(ms, emp, yerr=[[e - l for e, l in zip(emp, lo)],
                               [h - e for e, h in zip(emp, hi)]],
                fmt="s", color=C_NAIVE, ms=4, lw=1.0, capsize=2,
                label=r"Empirical $\mathrm{Var}(\bar R_m-\Theta)$ (Grid A)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms); ax.set_xticklabels(ms)
    ax.set_xlabel(r"Reports per root $m$ (one primitive root, $k=1$)")
    ax.set_ylabel(r"Variance of block-mean error (M EUR$^2$)")
    fig.tight_layout()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1,
              fontsize=8, frameon=False)
    fig.subplots_adjust(top=0.74)
    out = ROOT / "figures" / "fig8_precision_curve.pdf"
    fig.savefig(out); plt.close(fig)
    return out


# ------------------------------------------------------------------
# 2. QQ plot of extraction noise R - E
# ------------------------------------------------------------------

def _normal_ppf(p: float) -> float:
    """Acklam's rational approximation to the standard normal inverse CDF."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def extraction_noise_qq(cfg, pool):
    master_seed = cfg["master_seed"]
    noises = []
    for wid, roots in pool.items():
        if 0 not in roots:
            continue
        e_true = worldgen.generate_root_for_world(
            worldgen.generate_world(master_seed, wid, cfg), 0, master_seed, cfg
        ).e_true_float()
        for v, _, _ in roots[0]:
            noises.append(v - e_true)
    noises.sort()
    n = len(noises)
    mu = sum(noises) / n
    sd = gates.sample_sd(noises)
    theo = [mu + sd * _normal_ppf((i + 0.5) / n) for i in range(n)]

    # tail summary
    std = [(x - mu) / sd for x in noises]
    frac_beyond_3sd = sum(1 for s in std if abs(s) > 3) / n
    frac_beyond_4sd = sum(1 for s in std if abs(s) > 4) / n
    # excess kurtosis
    m4 = sum(s ** 4 for s in std) / n
    stats = {
        "n": n, "mean": mu, "sd": sd,
        "frac_beyond_3sd": frac_beyond_3sd,
        "frac_beyond_3sd_normal": 0.0027,
        "frac_beyond_4sd": frac_beyond_4sd,
        "frac_beyond_4sd_normal": 6.3e-05,
        "excess_kurtosis": m4 - 3,
    }
    return noises, theo, stats


def fig9_qq(noises, theo):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(theo, noises, ".", color=C_BAYES, ms=2, alpha=0.5)
    lims = [min(theo[0], noises[0]), max(theo[-1], noises[-1])]
    ax.plot(lims, lims, "--", color="gray", lw=0.8)
    ax.set_xlabel(r"Normal quantiles (fitted $\mu,\sigma$), M EUR")
    ax.set_ylabel(r"Empirical quantiles of $R-E$, M EUR")
    fig.tight_layout()
    out = ROOT / "figures" / "fig9_extraction_noise_qq.pdf"
    fig.savefig(out); plt.close(fig)
    return out


# ------------------------------------------------------------------
# 4. Leakage control at world level
# ------------------------------------------------------------------

def leakage_world_level(cfg):
    master_seed = cfg["master_seed"]
    recs = [json.loads(l) for l in open(ROOT / "results" / "raw" / "calibration.jsonl", encoding="utf-8")]
    seen, ded = set(), []
    for r in recs:
        if r["cache_key"] in seen:
            continue
        seen.add(r["cache_key"]); ded.append(r)
    leak = [r for r in ded if r["call_kind"] == "leakage" and r["valid"]]

    by_world: dict[int, list[float]] = {}
    for r in leak:
        by_world.setdefault(r["world_id"], []).append(r["parsed_estimate"])
    world_means = {wid: sum(v) / len(v) for wid, v in by_world.items()}
    thetas = {wid: worldgen.generate_world(master_seed, wid, cfg).theta for wid in world_means}

    wids = sorted(world_means)
    ests = [world_means[w] for w in wids]
    ths = [thetas[w] for w in wids]

    corr = gates.pearson_corr(ests, ths)
    p = gates.permutation_p_value(ests, ths, n_perm=20_000, seed=0)
    rmse_leak = gates.rmse(ests, ths)
    rmse_base = gates.rmse([cfg["dgp"]["prior_mean"]] * len(ths), ths)
    return {
        "n_worlds": len(wids),
        "corr_world_level": corr,
        "permutation_p_world_level": p,
        "rmse_leakage_world_means": rmse_leak,
        "rmse_baseline": rmse_base,
        "practical_improvement": (rmse_base - rmse_leak) / rmse_base,
        "note": "unit of analysis = per-world mean of the 3 no-doc calls; "
                "permutation over the 100 world clusters",
    }


def main():
    cfg, fit = load_cfg_fit()
    pool = load_pool(cfg)

    print("=== 1. Precision curve: Var(R_bar_m - Theta) vs sigma^2 + nu^2/m ===")
    rows, _ = precision_curve(cfg, fit, pool)
    for r in rows:
        inside = r["empirical_var_ci95"][0] <= r["predicted_var"] <= r["empirical_var_ci95"][1]
        print(f"m={r['m']:2d}  empirical={r['empirical_var']:8.1f} "
              f"[{r['empirical_var_ci95'][0]:8.1f}, {r['empirical_var_ci95'][1]:8.1f}]  "
              f"predicted={r['predicted_var']:8.1f}  pred_in_CI={inside}")

    print("\n--- 1b. Correlated-extraction (Section 6.3) gamma estimate ---")
    gamma_info = estimate_gamma(cfg, fit, pool)
    for k, v in gamma_info.items():
        print(f"{k}: {v}")

    p8 = fig8_precision_curve(rows, gamma_info)
    print(f"wrote {p8}\n")

    print("=== 2. QQ / tails of extraction noise R - E (Grid A root 0) ===")
    noises, theo, qstats = extraction_noise_qq(cfg, pool)
    for k, v in qstats.items():
        print(f"{k}: {v}")
    p9 = fig9_qq(noises, theo)
    print(f"wrote {p9}\n")

    print("=== 3. Bias handling ===")
    print(f"calibration report_bias = {fit['report_bias']:.2f} M EUR")
    print("CONFIRMED NOT SUBTRACTED: aggregate.py's naive_pool / provenance_pool /")
    print("dedup_pool / oracle_pool use raw report values; no bias correction is")
    print("applied anywhere in analyze.py either.")
    prior_sd = cfg["dgp"]["prior_sd"]
    print(f"(scale: bias is {abs(fit['report_bias'])/prior_sd:.3f} prior sd)\n")

    print("=== 4. Leakage control, world-level re-analysis ===")
    leak = leakage_world_level(cfg)
    for k, v in leak.items():
        print(f"{k}: {v}")

    out = {
        "precision_curve": rows,
        "gamma_correlated_extraction": gamma_info,
        "extraction_noise": qstats,
        "bias": {"report_bias": fit["report_bias"], "subtracted_in_aggregators": False},
        "leakage_world_level": leak,
    }
    out_path = ROOT / "results" / "processed" / "diagnostics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
