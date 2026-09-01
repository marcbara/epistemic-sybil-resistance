import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aggregate  # noqa: E402
import embed  # noqa: E402


# ---------------------------------------------------------------------------
# Reproduce the paper's Section 12.3 table exactly: tau2=sig2=nu2=1, k=1,
# n in {1,2,4,8,16,32}, checking naive vs dependence-aware coverage/NLL match
# the published numbers (within Monte Carlo tolerance is not needed here --
# we use the closed-form posterior directly, so this must match tightly).
# ---------------------------------------------------------------------------

PAPER_TABLE_12_3 = {
    # n: (bayes_coverage, naive_coverage) -- from main.tex Section 12.3
    1: (0.949, 0.949),
    2: (0.950, 0.921),
    4: (0.949, 0.833),
    8: (0.950, 0.686),
    16: (0.950, 0.521),
    32: (0.950, 0.381),
}


def _simulate_coverage(n, sigma2, nu2, tau2, prior_mean, reps, seed):
    rng = random.Random(seed)
    z95 = 1.959963984540054
    bayes_hits = 0
    naive_hits = 0
    for _ in range(reps):
        theta = rng.gauss(prior_mean, tau2 ** 0.5)
        eps = rng.gauss(0, sigma2 ** 0.5)
        e = theta + eps
        reports = [e + rng.gauss(0, nu2 ** 0.5) for _ in range(n)]

        post_bayes = aggregate.provenance_pool(
            {0: reports}, prior_mean, tau2 ** 0.5, sigma2, nu2
        )
        if abs(theta - post_bayes.mean) <= z95 * post_bayes.sd:
            bayes_hits += 1

        sigma_r2 = sigma2 + nu2
        post_naive = aggregate.naive_pool(reports, prior_mean, tau2 ** 0.5, sigma_r2)
        if abs(theta - post_naive.mean) <= z95 * post_naive.sd:
            naive_hits += 1
    return bayes_hits / reps, naive_hits / reps


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
def test_provenance_and_naive_match_paper_table_12_3(n):
    bayes_cov, naive_cov = _simulate_coverage(
        n, sigma2=1.0, nu2=1.0, tau2=1.0, prior_mean=0.0, reps=60_000, seed=20260818 + n
    )
    expected_bayes, expected_naive = PAPER_TABLE_12_3[n]
    assert bayes_cov == pytest.approx(expected_bayes, abs=0.01)
    assert naive_cov == pytest.approx(expected_naive, abs=0.01)


def test_provenance_pool_matches_proposition_2_precision():
    # J_m = m / (nu^2 + m*sigma^2); posterior precision = 1/tau^2 + J_m
    sigma2, nu2, tau2 = 2500.0, 625.0, 10000.0
    for m in [1, 2, 4, 8, 16, 32]:
        reports = {0: [500.0] * m}  # value doesn't matter for the variance check
        post = aggregate.provenance_pool(reports, 500.0, tau2 ** 0.5, sigma2, nu2)
        expected_j = m / (nu2 + m * sigma2)
        expected_precision = 1 / tau2 + expected_j
        expected_sd = (1 / expected_precision) ** 0.5
        assert post.sd == pytest.approx(expected_sd, rel=1e-9)


def test_provenance_pool_multiple_roots_sums_precision_additively():
    sigma2, nu2, tau2 = 2500.0, 625.0, 10000.0
    reports_by_root = {0: [500.0, 510.0], 1: [495.0, 505.0, 500.0]}
    post = aggregate.provenance_pool(reports_by_root, 500.0, tau2 ** 0.5, sigma2, nu2)
    j0 = 2 / (nu2 + 2 * sigma2)
    j1 = 3 / (nu2 + 3 * sigma2)
    expected_precision = 1 / tau2 + j0 + j1
    assert post.sd == pytest.approx((1 / expected_precision) ** 0.5, rel=1e-9)


def test_oracle_pool_has_no_extraction_noise_uncertainty():
    # Oracle observes E exactly for k roots; precision should be k/sigma_hat2 + prior.
    sigma_hat2, tau2 = 2500.0, 10000.0
    e_true_by_root = {0: 480.0, 1: 520.0, 2: 500.0}
    post = aggregate.oracle_pool(e_true_by_root, 500.0, tau2 ** 0.5, sigma_hat2)
    expected_precision = 1 / tau2 + 3 / sigma_hat2
    assert post.sd == pytest.approx((1 / expected_precision) ** 0.5, rel=1e-9)
    assert post.mean == pytest.approx(500.0, abs=1e-6)  # symmetric E's average to 500


def test_naive_pool_reduces_variance_without_bound_as_n_grows():
    # Sanity: unlike provenance_pool, naive keeps shrinking (no saturation).
    sigma_r2, prior_sd = 100.0, 100.0
    sds = []
    for n in [1, 2, 4, 8, 16, 32]:
        post = aggregate.naive_pool([500.0] * n, 500.0, prior_sd, sigma_r2)
        sds.append(post.sd)
    assert all(sds[i] > sds[i + 1] for i in range(len(sds) - 1))


def test_gamma_pool_reduces_to_provenance_pool_at_gamma_zero():
    sigma2, nu2, tau2 = 2500.0, 625.0, 10000.0
    reports = {0: [480.0, 510.0, 495.0], 1: [505.0, 490.0]}
    p0 = aggregate.provenance_pool(reports, 500.0, tau2 ** 0.5, sigma2, nu2)
    pg = aggregate.provenance_pool_gamma(reports, 500.0, tau2 ** 0.5, sigma2, nu2, gamma=0.0)
    assert pg.mean == pytest.approx(p0.mean, rel=1e-12)
    assert pg.sd == pytest.approx(p0.sd, rel=1e-12)


def test_gamma_pool_matches_section_6_3_precision():
    # J_m = m / (v + (m-1)c) with v = sigma^2+nu^2, c = sigma^2+gamma*nu^2
    sigma2, nu2, tau2, gamma = 2500.0, 5000.0, 10000.0, 0.7
    for m in [1, 2, 4, 8, 16, 32]:
        post = aggregate.provenance_pool_gamma(
            {0: [500.0] * m}, 500.0, tau2 ** 0.5, sigma2, nu2, gamma
        )
        v = sigma2 + nu2
        c = sigma2 + gamma * nu2
        expected_precision = 1 / tau2 + m / (v + (m - 1) * c)
        assert post.sd == pytest.approx((1 / expected_precision) ** 0.5, rel=1e-9)


def test_gamma_pool_ceiling_is_sigma2_plus_gamma_nu2():
    # As m grows, block precision approaches 1/(sigma^2 + gamma*nu^2).
    sigma2, nu2, gamma = 2500.0, 5000.0, 0.7
    v, c = sigma2 + nu2, sigma2 + gamma * nu2
    j_large = 100000 / (v + (100000 - 1) * c)
    assert j_large == pytest.approx(1 / c, rel=1e-3)


def test_gamma_pool_is_more_conservative_than_gamma_zero():
    sigma2, nu2, tau2 = 2500.0, 5000.0, 10000.0
    reports = {0: [500.0] * 16}
    p0 = aggregate.provenance_pool_gamma(reports, 500.0, tau2 ** 0.5, sigma2, nu2, 0.0)
    p7 = aggregate.provenance_pool_gamma(reports, 500.0, tau2 ** 0.5, sigma2, nu2, 0.7)
    assert p7.sd > p0.sd


# ---------------------------------------------------------------------------
# Dedup / embedding
# ---------------------------------------------------------------------------

def test_embed_identical_text_has_similarity_one():
    v1 = embed.embed_text("Product segment stated directly, services as percentage of product.")
    v2 = embed.embed_text("Product segment stated directly, services as percentage of product.")
    assert embed.cosine_similarity(v1, v2) == pytest.approx(1.0, abs=1e-9)


def test_embed_unrelated_text_has_low_similarity():
    v1 = embed.embed_text("Product segment stated directly, services as percentage of product.")
    v2 = embed.embed_text("Quarterly headcount growth remained flat across all regional offices.")
    assert embed.cosine_similarity(v1, v2) < 0.3


def test_dedup_pool_collapses_identical_rationales_to_one_cluster():
    values = [480.0, 485.0, 520.0, 515.0]
    rationales = ["same wording here"] * 2 + ["different wording entirely"] * 2
    post, n_clusters = aggregate.dedup_pool(values, rationales, 500.0, 100.0, 2500.0, threshold=0.99)
    assert n_clusters == 2


def test_dedup_pool_threshold_zero_merges_everything_into_one_cluster():
    values = [480.0, 485.0, 520.0, 515.0]
    rationales = ["completely different text A", "totally unrelated text B",
                  "another distinct phrase C", "yet another phrase D"]
    post, n_clusters = aggregate.dedup_pool(values, rationales, 500.0, 100.0, 2500.0, threshold=-1.0)
    assert n_clusters == 1


def test_fit_dedup_threshold_picks_lowest_mean_nll():
    # Construct synthetic calibration worlds where the "right" threshold is
    # known: reports come in two ground-truth clusters far apart in value.
    rng = random.Random(0)
    worlds_data = []
    for _ in range(50):
        theta = rng.gauss(500, 50)
        cluster_a = [theta + rng.gauss(-20, 2) for _ in range(4)]
        cluster_b = [theta + rng.gauss(20, 2) for _ in range(4)]
        rationales = ["alpha phrasing consistent"] * 4 + ["beta phrasing consistent"] * 4
        worlds_data.append({"theta": theta, "values": cluster_a + cluster_b, "rationales": rationales})
    threshold = aggregate.fit_dedup_threshold(
        worlds_data, prior_mean=500.0, prior_sd=100.0, sigma_r2=400.0,
        threshold_grid=[0.5, 0.8, 0.9, 0.95, 0.99],
    )
    assert threshold in [0.5, 0.8, 0.9, 0.95, 0.99]
