import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gates  # noqa: E402


def test_validity_rate():
    assert gates.validity_rate([True, True, True, False]) == 0.75
    assert gates.validity_rate([]) == 0.0


def test_bias_and_rmse_zero_for_perfect_reports():
    reports = [100.0, 200.0, 300.0]
    targets = [100.0, 200.0, 300.0]
    assert gates.bias(reports, targets) == 0.0
    assert gates.rmse(reports, targets) == 0.0


def test_bias_and_rmse_known_offset():
    reports = [105.0, 210.0, 295.0]
    targets = [100.0, 200.0, 300.0]
    assert gates.bias(reports, targets) == pytest.approx((5 + 10 - 5) / 3)
    assert gates.rmse(reports, targets) == pytest.approx(
        ((5**2 + 10**2 + 5**2) / 3) ** 0.5
    )


def test_pearson_corr_perfect_and_none():
    xs = [1, 2, 3, 4, 5]
    ys = [2, 4, 6, 8, 10]
    assert gates.pearson_corr(xs, ys) == pytest.approx(1.0)
    rng = random.Random(0)
    ys_random = [rng.uniform(-1, 1) for _ in xs]
    assert -1.0 <= gates.pearson_corr(xs, ys_random) <= 1.0


def test_one_way_icc_recovers_known_rho_with_theta_held_fixed():
    # Corollary 2's rho is conditional on Theta. To test the ANOVA estimator
    # against it validly, hold Theta fixed and let many independent roots
    # (clusters) vary only through eps -- the Grid B/C nested-design regime,
    # not the Phase 0 k=1-varying-Theta regime (see docstring in gates.py).
    rng = random.Random(42)
    theta = 500.0
    sigma2, nu2 = 2500.0, 625.0
    n_roots, n_reports = 500, 8
    groups = []
    for _ in range(n_roots):
        e = theta + rng.gauss(0, sigma2 ** 0.5)
        reports = [e + rng.gauss(0, nu2 ** 0.5) for _ in range(n_reports)]
        groups.append(reports)
    result = gates.one_way_icc(groups)
    expected_rho = sigma2 / (sigma2 + nu2)
    assert result.rho_hat == pytest.approx(expected_rho, abs=0.05)


def test_one_way_icc_conflates_theta_variance_when_grouping_by_varying_theta_world():
    # Documents the caution in gates.one_way_icc's docstring: grouping by
    # world when Theta itself varies world-to-world inflates rho_hat well
    # above the true conditional value, since Var(Theta) leaks into the
    # between-group term. This is why Phase 0 gate 3 must use the known-DGP
    # variance decomposition instead of this estimator.
    rng = random.Random(42)
    sigma2, nu2 = 2500.0, 625.0
    n_worlds, n_reports = 500, 8
    groups = []
    for _ in range(n_worlds):
        theta = rng.gauss(500, 100)
        e = theta + rng.gauss(0, sigma2 ** 0.5)
        reports = [e + rng.gauss(0, nu2 ** 0.5) for _ in range(n_reports)]
        groups.append(reports)
    result = gates.one_way_icc(groups)
    true_conditional_rho = sigma2 / (sigma2 + nu2)
    assert result.rho_hat > true_conditional_rho + 0.1  # visibly inflated


def test_one_way_icc_near_zero_when_independent():
    rng = random.Random(1)
    groups = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(500)]
    result = gates.one_way_icc(groups)
    assert abs(result.rho_hat) < 0.05


def test_variance_decomposition_matches_icc_under_dgp():
    rng = random.Random(7)
    sigma2, nu2 = 2500.0, 625.0
    n_worlds, n_reports = 500, 8
    eps_values, noise_values = [], []
    for _ in range(n_worlds):
        theta = rng.gauss(500, 100)
        eps = rng.gauss(0, sigma2 ** 0.5)
        e = theta + eps
        eps_values.append(eps)
        for _ in range(n_reports):
            eta = rng.gauss(0, nu2 ** 0.5)
            noise_values.append(eta)
    decomp = gates.variance_decomposition(eps_values, noise_values)
    assert decomp.predicted_rho == pytest.approx(0.8, abs=0.03)


def test_leakage_control_passes_when_no_signal():
    rng = random.Random(3)
    thetas = [rng.gauss(500, 100) for _ in range(100)]
    estimates = [rng.gauss(500, 100) for _ in range(100)]  # independent of theta
    result = gates.leakage_control(
        estimates, thetas, prior_mean=500.0,
        practical_threshold=0.10, statistical_corr_threshold=0.20,
        n_perm=2000, seed=0,
    )
    assert result.practical_pass
    assert result.statistical_pass


def test_leakage_control_fails_when_strong_signal():
    rng = random.Random(4)
    thetas = [rng.gauss(500, 100) for _ in range(100)]
    estimates = [t + rng.gauss(0, 5) for t in thetas]  # near-perfect leakage
    result = gates.leakage_control(
        estimates, thetas, prior_mean=500.0,
        practical_threshold=0.10, statistical_corr_threshold=0.20,
        n_perm=2000, seed=0,
    )
    assert not result.practical_pass
    assert not result.statistical_pass
    assert result.permutation_p < 0.01
