"""Pure statistics for the Phase 0 pilot gates (EXPERIMENT.md Section 8) and
the calibration-split variance decomposition (Section 4). No network calls;
everything here operates on already-collected records so it can be unit
tested without API access.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


def validity_rate(valid_flags: Sequence[bool]) -> float:
    if not valid_flags:
        return 0.0
    return sum(1 for v in valid_flags if v) / len(valid_flags)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def sample_variance(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def sample_sd(xs: Sequence[float]) -> float:
    return math.sqrt(sample_variance(xs))


def bias(reports: Sequence[float], targets: Sequence[float]) -> float:
    diffs = [r - t for r, t in zip(reports, targets)]
    return mean(diffs)


def rmse(reports: Sequence[float], targets: Sequence[float]) -> float:
    diffs = [(r - t) ** 2 for r, t in zip(reports, targets)]
    return math.sqrt(mean(diffs))


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if denom == 0:
        return 0.0
    return num / denom


def permutation_p_value(xs: Sequence[float], ys: Sequence[float], n_perm: int = 10_000, seed: int = 0) -> float:
    observed = abs(pearson_corr(xs, ys))
    rng = random.Random(seed)
    ys_list = list(ys)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(ys_list)
        if abs(pearson_corr(xs, ys_list)) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)  # add-one smoothing, avoids p=0


@dataclass
class ICCResult:
    rho_hat: float
    n_groups: int
    avg_group_size: float
    ms_between: float
    ms_within: float


def one_way_icc(groups: Sequence[Sequence[float]]) -> ICCResult:
    """Classic one-way random-effects intraclass correlation (ANOVA estimator).

    groups: list of report-value lists, one list per *root* (shared-root block).

    Caution: Corollary 2's rho = sigma^2/(sigma^2+nu^2) is defined conditional
    on Theta -- it is the correlation among reports sharing one root, holding
    the world's true state fixed. This estimator recovers that quantity only
    when the between-group variance in the data reflects root-level noise
    (eps) alone. With k=1 root per world and Theta varying world to world (as
    in the Phase 0 pilot), grouping by world instead conflates cross-world
    Var(Theta) into the between-group term, inflating rho_hat far above the
    true conditional value. It is a valid diagnostic only where multiple
    roots share a common, known Theta (e.g. Grid B/C's k>1 worlds, ideally
    after centering each world's reports on that world's own Theta) -- not a
    substitute for the sigma_hat^2/(sigma_hat^2+nu_hat^2) decomposition below,
    which exploits the known DGP and is what the pilot's gate 3 should use.
    """
    groups = [g for g in groups if len(g) > 0]
    G = len(groups)
    ns = [len(g) for g in groups]
    N = sum(ns)
    grand_mean = sum(sum(g) for g in groups) / N

    ss_between = sum(n * (mean(g) - grand_mean) ** 2 for g, n in zip(groups, ns))
    ss_within = sum(sum((x - mean(g)) ** 2 for x in g) for g in groups)

    df_between = G - 1
    df_within = N - G
    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    n0 = (N - sum(n ** 2 for n in ns) / N) / (G - 1) if G > 1 else mean(ns)
    denom = ms_between + (n0 - 1) * ms_within
    rho_hat = (ms_between - ms_within) / denom if denom != 0 else 0.0
    return ICCResult(
        rho_hat=rho_hat,
        n_groups=G,
        avg_group_size=N / G if G else 0.0,
        ms_between=ms_between,
        ms_within=ms_within,
    )


@dataclass
class VarianceDecomposition:
    sigma_hat2: float  # Var(E - Theta) -- document-level distortion
    nu_hat2: float      # Var(report - E) -- extraction noise
    predicted_rho: float  # sigma_hat2 / (sigma_hat2 + nu_hat2)


def variance_decomposition(eps_values: Sequence[float], extraction_noise: Sequence[float]) -> VarianceDecomposition:
    sigma_hat2 = sample_variance(eps_values)
    nu_hat2 = sample_variance(extraction_noise)
    denom = sigma_hat2 + nu_hat2
    predicted_rho = sigma_hat2 / denom if denom > 0 else 0.0
    return VarianceDecomposition(sigma_hat2=sigma_hat2, nu_hat2=nu_hat2, predicted_rho=predicted_rho)


@dataclass
class LeakageResult:
    rmse_leakage: float
    rmse_baseline: float
    practical_improvement: float  # (baseline - leakage) / baseline; must be < threshold to PASS
    corr: float
    permutation_p: float
    practical_pass: bool
    statistical_pass: bool  # only meaningful at full calibration N (~100 worlds)


def leakage_control(
    leakage_estimates: Sequence[float],
    thetas: Sequence[float],
    prior_mean: float,
    practical_threshold: float,
    statistical_corr_threshold: float,
    n_perm: int = 10_000,
    seed: int = 0,
) -> LeakageResult:
    rmse_leak = rmse(leakage_estimates, thetas)
    baseline_preds = [prior_mean] * len(thetas)
    rmse_base = rmse(baseline_preds, thetas)
    improvement = (rmse_base - rmse_leak) / rmse_base if rmse_base > 0 else 0.0
    corr = pearson_corr(leakage_estimates, thetas)
    p_val = permutation_p_value(leakage_estimates, thetas, n_perm=n_perm, seed=seed)
    return LeakageResult(
        rmse_leakage=rmse_leak,
        rmse_baseline=rmse_base,
        practical_improvement=improvement,
        corr=corr,
        permutation_p=p_val,
        practical_pass=improvement < practical_threshold,
        statistical_pass=abs(corr) < statistical_corr_threshold,
    )
