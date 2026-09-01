# Phase 0 pilot report
Worlds: 30, reports/world: 8, temperature: 0.7
Total report calls: 240, valid: 240
Total leakage calls: 90, valid: 90

## Gate 1: parse + validity rate >= 95%
Validity rate: 1.000 -- PASS

## Gate 2: extraction noise is real (not clone regime)
Mean within-document report sd: 26.02 M EUR -- PASS

## Gate 3: rho_hat in workable band [0.2, 0.95]
sigma_hat^2 = 1871.51 (Var(E-Theta))
nu_hat^2 = 4859.32 (Var(report-E))
rho_hat (predicted, sigma^2/(sigma^2+nu^2)) = 0.278 -- PASS

## Gate 4: leakage control
RMSE(no-doc estimate, Theta) = 456.58
RMSE(constant prior mean, Theta) = 88.32
Practical improvement over baseline: -417.0% (must be < 10%) -- PASS
corr(no-doc estimate, Theta) = -0.030, permutation p = 0.7404 (supplementary; statistical gate requires the full ~100-world calibration set, not this 30-world pilot)

## Gate 5: reports approximately unbiased around E
Bias (mean report - E): 3.69 M EUR -- PASS

## Gate 6: task informative but nontrivial
RMSE(report, E) = 69.66 M EUR, prior sd = 100.0 M EUR, ratio = 0.697 (must be < 0.5) -- FAIL

## Overall: SOME GATES FAILED

Per EXPERIMENT.md Section 8: STOP here for human review. Do not freeze templates or proceed to Grids A-D until a human has read this report.
