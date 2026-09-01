"""Aggregators (EXPERIMENT.md Section 5). All receive the true prior over
Theta (Normal(prior_mean, prior_sd^2)) and return a Gaussian posterior
(mean, sd). Parameters (sigma_r2, sigma_hat2, nu_hat2) come from the frozen
calibration fit (calibration_fit.json), never from evaluation worlds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from embed import embed_texts, cosine_similarity


@dataclass
class Posterior:
    mean: float
    sd: float


def _gaussian_update(prior_mean: float, prior_sd: float, obs_mean: float, obs_precision: float) -> Posterior:
    prior_precision = 1.0 / (prior_sd ** 2)
    post_precision = prior_precision + obs_precision
    post_mean = (prior_mean * prior_precision + obs_mean * obs_precision) / post_precision
    post_sd = math.sqrt(1.0 / post_precision)
    return Posterior(mean=post_mean, sd=post_sd)


def naive_pool(reports: Sequence[float], prior_mean: float, prior_sd: float, sigma_r2: float) -> Posterior:
    """Theorem-1-naive: n reports treated as iid observations of variance sigma_r2."""
    n = len(reports)
    if n == 0:
        return Posterior(mean=prior_mean, sd=prior_sd)
    likelihood_precision = n / sigma_r2
    obs_mean = sum(reports) / n
    return _gaussian_update(prior_mean, prior_sd, obs_mean, likelihood_precision)


def provenance_pool(
    reports_by_root: dict, prior_mean: float, prior_sd: float, sigma_hat2: float, nu_hat2: float
) -> Posterior:
    """Equicorrelated Gaussian posterior per shared-root block (Corollary 2),
    blocks combined additively (roots assumed independent given Theta).

    Per root j with m_j reports, block precision J_j = m_j / (nu^2 + m_j*sigma^2)
    (Proposition 2) and the block's sufficient statistic is its own mean.
    """
    total_precision = 0.0
    weighted_sum = 0.0
    for root_id, values in reports_by_root.items():
        m = len(values)
        if m == 0:
            continue
        j = m / (nu_hat2 + m * sigma_hat2)
        rbar = sum(values) / m
        total_precision += j
        weighted_sum += j * rbar
    if total_precision == 0:
        return Posterior(mean=prior_mean, sd=prior_sd)
    obs_mean = weighted_sum / total_precision
    return _gaussian_update(prior_mean, prior_sd, obs_mean, total_precision)


def provenance_pool_gamma(
    reports_by_root: dict, prior_mean: float, prior_sd: float,
    sigma_hat2: float, nu_hat2: float, gamma: float,
) -> Posterior:
    """Correlated-extraction provenance aggregator (paper Section 6.3).

    Within a root, Cov(eta_i, eta_j) = gamma * nu^2 for i != j, so each
    report has conditional variance v = sigma^2 + nu^2 and within-block
    off-diagonal covariance c = sigma^2 + gamma * nu^2. Block precision:
        J_m = m / (v + (m-1) * c),
    which reduces exactly to Proposition 2's m / (nu^2 + m*sigma^2) at
    gamma = 0. Blocks combine additively (roots independent given Theta).
    """
    v = sigma_hat2 + nu_hat2
    c = sigma_hat2 + gamma * nu_hat2
    total_precision = 0.0
    weighted_sum = 0.0
    for root_id, values in reports_by_root.items():
        m = len(values)
        if m == 0:
            continue
        j = m / (v + (m - 1) * c)
        rbar = sum(values) / m
        total_precision += j
        weighted_sum += j * rbar
    if total_precision == 0:
        return Posterior(mean=prior_mean, sd=prior_sd)
    obs_mean = weighted_sum / total_precision
    return _gaussian_update(prior_mean, prior_sd, obs_mean, total_precision)


def oracle_pool(
    e_true_by_root: dict, prior_mean: float, prior_sd: float, sigma_hat2: float
) -> Posterior:
    """Diagnostic upper bound: the exact primitive evidence E_wj is observed
    directly for each root (nu^2 -> 0), only the sigma^2 uncertainty about
    Theta given E remains."""
    total_precision = 0.0
    weighted_sum = 0.0
    for root_id, e_value in e_true_by_root.items():
        j = 1.0 / sigma_hat2
        total_precision += j
        weighted_sum += j * e_value
    if total_precision == 0:
        return Posterior(mean=prior_mean, sd=prior_sd)
    obs_mean = weighted_sum / total_precision
    return _gaussian_update(prior_mean, prior_sd, obs_mean, total_precision)


def _cluster_by_threshold(vectors: Sequence[Sequence[float]], threshold: float) -> list[list[int]]:
    """Connected components: link i,j if cosine_similarity >= threshold."""
    n = len(vectors)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for k in range(i + 1, n):
            if cosine_similarity(vectors[i], vectors[k]) >= threshold:
                union(i, k)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def dedup_pool(
    values: Sequence[float],
    rationales: Sequence[str],
    prior_mean: float,
    prior_sd: float,
    sigma_r2: float,
    threshold: float,
) -> tuple[Posterior, int]:
    """Report-space deduplication baseline (Section 5.2, primary variant:
    embed the rationale only). Collapses each cluster to its mean value, then
    pools the cluster means with the naive method (each treated as variance
    sigma_r2, per the brief's literal "then pool as in 1" -- not re-scaled by
    cluster size)."""
    n = len(values)
    if n == 0:
        return Posterior(mean=prior_mean, sd=prior_sd), 0
    vectors = embed_texts(rationales)
    clusters = _cluster_by_threshold(vectors, threshold)
    cluster_means = [sum(values[i] for i in idxs) / len(idxs) for idxs in clusters]
    post = naive_pool(cluster_means, prior_mean, prior_sd, sigma_r2)
    return post, len(clusters)


def nll(theta: float, post: Posterior) -> float:
    var = post.sd ** 2
    return 0.5 * math.log(2 * math.pi * var) + (theta - post.mean) ** 2 / (2 * var)


def fit_dedup_threshold(
    worlds_data: Sequence[dict], prior_mean: float, prior_sd: float, sigma_r2: float,
    threshold_grid: Sequence[float],
) -> float:
    """Sweep a fixed threshold grid on calibration worlds, pick the one
    minimizing mean NLL of the deduplicated aggregator (Section 5.2), frozen
    before evaluation.

    worlds_data: list of {"theta": float, "values": [...], "rationales": [...]}.
    """
    best_threshold, best_nll = threshold_grid[0], float("inf")
    for t in threshold_grid:
        nlls = []
        for w in worlds_data:
            post, _ = dedup_pool(w["values"], w["rationales"], prior_mean, prior_sd, sigma_r2, t)
            nlls.append(nll(w["theta"], post))
        mean_nll = sum(nlls) / len(nlls)
        if mean_nll < best_nll:
            best_nll = mean_nll
            best_threshold = t
    return best_threshold
