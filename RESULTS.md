# Results: Grids A and B

Empirical companion to `paper/latex-src/main.tex` Sections 12.3-12.4 and
predictions 13.1-13.2. Main agent: `claude-haiku-4-5-20251001`, T=0.7, frozen
prompt (see `EXPERIMENT_NOTES.md` for the Phase 0 freeze record). All
parameters below (`sigma_r2`, `sigma_hat2`, `nu_hat2`, dedup threshold) were
fit on the disjoint 100-world calibration split (Section 4) -- none are
fitted on the 300 Grid A/B evaluation worlds reported here.

## Data quality

- Grid A/B collection: 19,200/19,200 calls valid (100.0%), 2 rationale
  truncations (out of 19,200 -- the frozen prompt holds at scale).
- Calibration split: 800/800 report calls valid, 300/300 leakage calls valid.
- Total spend across the whole session (Phase 0 pilot iterations +
  calibration + Grid A/B): ~$18.6, against a combined cap of $10 (pilot) +
  $25 (Grid A/B).

## Calibration fit (frozen, Section 4)

| Parameter | Value |
|---|---|
| sigma_r2 = Var(report - Theta) | 7718.35 |
| sigma_hat2 = Var(E - Theta) | 2518.69 |
| nu_hat2 = Var(report - E) | 5107.78 |
| rho_hat = sigma_hat2/(sigma_hat2+nu_hat2) | 0.330 |
| report bias (report - E) | -11.31 |
| dedup cosine threshold (frozen) | 0.80 |

**rho identity note.** rho_hat = 0.330 sits inside the pilot's prespecified
workable band [0.2, 0.95] and is consistent across pilot (0.25-0.33) and
calibration runs. On its own this is a stability check, not evidence that
the random-effects model describes the reports: the identity
rho = sigma^2/(sigma^2+nu^2) is imposed by the definition of rho_hat, not
tested by computing it. The substantive test is the precision-curve
diagnostic below -- which in fact *rejects* the independent-extraction
model and selects Section 6.3's correlated-extraction variant instead.

**Leakage control** (100 calibration worlds): practical criterion passes
decisively (no-doc RMSE is 5.1x *worse* than the constant-prior baseline,
i.e. no predictive signal). For the statistical criterion, the correct unit
of analysis is the world, not the call: the 3 no-doc calls per world share
Theta, so the initial call-level analysis (corr = 0.114, permutation
p = 0.049 over 300 pseudo-replicated observations) was anti-conservative.
Re-run at world level (per-world means, permutation over the 100 world
clusters): corr = 0.165, permutation p = 0.101. The primary threshold
(|corr| < 0.20) still passes, though with less margin than the call-level
number suggested; both analyses are reported.

**Dedup threshold note.** The frozen threshold (0.80) sits at the low end
of the swept grid; mean NLL was monotonically non-increasing as the
threshold decreased across the full grid tested (0.1 to 0.98), because
every calibration world has k=1 -- there is no calibration-world structure
that would ever reward *not* merging all of a world's reports into one
cluster. This is an expected property of a single-root calibration set, not
a search failure.

## Grid A: report multiplication at k=1 (prediction 13.1)

300 worlds, root 1's report pool nested for n in {1,2,4,8,16,32}.

| n | Naive coverage | Prov. coverage | Naive C | Prov. C | Dedup C | Naive NLL | Prov. NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.940 [0.910, 0.963] | 0.940 [0.910, 0.963] | 0.961 [0.870, 1.047] | 0.965 [0.874, 1.052] | 0.961 | 5.571 | 5.571 |
| 2 | 0.900 [0.867, 0.933] | 0.923 [0.893, 0.950] | 1.246 [1.116, 1.373] | 1.083 [0.975, 1.192] | 1.014 | 5.661 | 5.566 |
| 4 | 0.787 [0.740, 0.833] | 0.897 [0.863, 0.930] | 1.836 [1.644, 2.033] | 1.247 [1.121, 1.373] | 1.063 | 6.300 | 5.656 |
| 8 | 0.570 [0.510, 0.627] | 0.877 [0.837, 0.913] | 2.655 [2.359, 2.945] | 1.351 [1.213, 1.486] | 1.017 | 7.834 | 5.723 |
| 16 | 0.400 [0.343, 0.453] | 0.860 [0.823, 0.897] | 3.848 [3.428, 4.283] | 1.427 [1.283, 1.579] | 0.982 | 11.388 | 5.788 |
| 32 | 0.263 [0.213, 0.310] | 0.850 [0.810, 0.887] | 5.473 [4.867, 6.081] | 1.458 [1.313, 1.616] | 0.948 | 18.626 | 5.810 |

95% CIs are cluster bootstrap by world, 1000 resamples. n=300 worlds at every cell.

**Verdict on 13.1 (multiplicity inflation): confirmed.** Naive coverage
falls monotonically from 0.940 (n=1) to 0.263 (n=32), a collapse the
bootstrap CIs place well outside overlap with the n=1 cell by n=8. Naive C
grows from ~0.96 to 5.47 -- at n=32 the naive posterior's reported
uncertainty is roughly 5.5x too tight. Provenance-aware coverage instead
stays in a much narrower 0.850-0.940 band throughout (not perfectly flat at
the nominal 0.95 the way the idealized Section 12 simulation is; the
post-hoc diagnostics below trace the drift to correlated extraction errors
within a document, gamma_cal = 0.72, and show that a gamma-aware variant
of the same aggregator restores 0.94-0.95 coverage throughout -- but even
without that correction, categorically different from naive's collapse).
Naive NLL more than triples (5.57 to 18.63) while provenance-aware NLL is
essentially flat (5.57 to 5.81). The report-space dedup baseline, tuned on
calibration, tracks provenance-aware closely here (C stays near 1.0
throughout) -- expected, since Grid A is exactly the single-root regime the
dedup threshold was tuned on (see caveat above: the calibration set can't
distinguish "correctly discounting a shared root" from "correctly avoiding
false splits under genuine independence," so this cell is not yet the
baseline's hard case -- that's Grid D/prediction 13.4 territory).

## Grid B: report count vs root count at n=16 fixed (12.4 analogue)

Same 300 worlds; k in {1,2,4,8,16} roots, 16/k reports per root.

| k | Naive RMSE | Prov. RMSE | Naive coverage | Prov. coverage | Naive NLL | Prov. NLL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 82.55 [73.53, 91.88] | 67.10 [60.33, 74.23] | 0.400 | 0.860 | 11.388 | 5.788 |
| 2 | 54.83 [50.08, 60.20] | 50.31 [46.05, 54.80] | 0.563 | 0.843 | 7.252 | 5.456 |
| 4 | 36.90 [34.02, 40.20] | 35.87 [33.06, 38.76] | 0.747 | 0.880 | 5.464 | 5.044 |
| 8 | 28.93 [26.57, 31.48] | 28.81 [26.49, 31.40] | 0.850 | 0.907 | 4.894 | 4.810 |
| 16 | 23.49 [21.63, 25.34] | 23.49 [21.64, 25.33] | 0.927 | 0.927 | 4.584 | 4.585 |

k=1 reproduces Grid A's n=16 cell exactly (same underlying data view) --
useful as an internal consistency check, and it matches to the last digit.

**Verdict on the 12.4 analogue: confirmed, including the k=n convergence
the theory predicts.** Holding report count fixed at n=16, both RMSE and
NLL improve monotonically as k grows for both aggregators -- more
independent roots carry more real information, full stop. The distinctive
prediction is in the *naive/provenance gap*, not either curve alone: at
k=1 naive is badly overconfident (NLL 11.39 vs provenance's 5.79, coverage
0.40 vs 0.86); as k rises toward n the naive independence assumption
becomes literally correct, and by k=16 (every report its own root) naive
and provenance-aware are statistically indistinguishable (NLL 4.584 vs
4.585, coverage 0.927 vs 0.927, RMSE CIs overlapping almost completely).
Report count alone (n=16 constant throughout) does not track this gap --
only the root count does, exactly the paper's point that report count and
evidence-root count are different quantities.

## Post-hoc diagnostics (no additional API calls; `src/diagnostics.py`)

**Precision curve: the independent-extraction model is rejected; the
correlated-extraction model (Section 6.3) is validated out-of-sample.**
The empirical Var(R_bar_m - Theta) across Grid A's 300 evaluation worlds,
against two predictions whose every parameter -- sigma^2, nu^2, and gamma --
is estimated exclusively on the disjoint 100-world calibration split
(gamma_cal = 1 - (mean within-document variance)/nu_hat2 = 1 - 1435/5108
= 0.719, from the calibration split's 8-report pools; frozen into
`calibration_fit.json` before evaluating):

| m | Empirical Var [95% CI] | Independent pred. (sigma^2 + nu^2/m) | Correlated pred. (gamma_cal = 0.719) | Corr. pred. in CI |
|---:|---:|---:|---:|:---:|
| 1 | 9177 [7254, 11313] | 7627 | 7627 | yes |
| 2 | 8258 [6431, 10225] | 5073 | 6909 | yes |
| 4 | 8172 [6337, 10051] | 3796 | 6550 | yes |
| 8 | 7748 [6141, 9584] | 3157 | 6371 | yes |
| 16 | 7671 [6080, 9489] | 2838 | 6281 | yes |
| 32 | 7510 [5931, 9300] | 2678 | 6236 | yes |

The curve barely declines in m: replicate reports from the same model on
the same document mostly repeat the same error rather than drawing fresh
noise. The independent prediction falls outside the bootstrap CI at every
m >= 2; the calibration-frozen correlated prediction
sigma^2 + gamma*nu^2 + (1-gamma)*nu^2/m falls inside the CI at every m.
An in-sample re-estimate of gamma from Grid A's own within-document
variance gives 0.703, closely agreeing with the calibration value (0.719)
-- the parameter is stable across disjoint world sets. The implied
information ceiling is sigma^2 + gamma*nu^2 = 6191, far above the
independent model's sigma^2 = 2519. This is a substantive empirical
finding, not a nuisance: same-model report multiplicity buys almost
nothing beyond small m, exactly the "nominal agent multiplicity cannot
remove error components common to all extractors" regime the paper
describes analytically. It also explains why the provenance-aware
aggregator (which assumes independent eta within a root) drifts to
0.85-0.86 coverage at large n in Grid A instead of holding 0.95: it is
itself still too optimistic about what replicates can average away.
(fig8 draws the gamma_cal, i.e. out-of-sample, curve.)

**Gamma-aware provenance recovers calibration (exploratory).** Adding
Section 6.3's correlation to the provenance aggregator -- block precision
J_m = m/(v + (m-1)c) with v = sigma^2 + nu^2, c = sigma^2 + gamma_cal*nu^2,
all three parameters calibration-only -- and re-running Grids A and B:

| n (Grid A, k=1) | Naive cov. | Prov. cov. | Prov.+gamma cov. | Naive C | Prov. C | Prov.+gamma C |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.940 | 0.940 | 0.940 | 0.961 | 0.965 | 0.965 |
| 2 | 0.900 | 0.923 | 0.943 | 1.246 | 1.083 | 0.952 |
| 4 | 0.787 | 0.897 | 0.947 | 1.836 | 1.247 | 0.963 |
| 8 | 0.570 | 0.877 | 0.953 | 2.655 | 1.351 | 0.960 |
| 16 | 0.400 | 0.860 | 0.943 | 3.848 | 1.427 | 0.961 |
| 32 | 0.263 | 0.850 | 0.943 | 5.473 | 1.458 | 0.954 |

Coverage holds at 0.94-0.95 for every n, with the calibration ratio flat
at 0.95-0.96 -- the gamma-aware aggregator is essentially perfectly
calibrated across the full multiplicity range, where the gamma=0
provenance aggregator drifted to 0.85. In Grid B it dominates or matches
gamma=0 provenance at every k and converges to it exactly at k=16 (m=1
per root, where gamma is vacuous): coverage 0.943/0.967/0.967/0.953/0.927
across k=1..16. The paper's Section 6.3 extension is therefore not just a
diagnosis of the miscalibration but its correction, with no evaluation-set
tuning.

Methodological status, stated precisely: the correlated-extraction
specification was present in the theoretical model prior to the experiment,
but its empirical evaluation was prompted by the observed failure of the
independent-extraction specification. All parameters used in the
subsequent evaluation were estimated exclusively on the disjoint
calibration set. This analysis is exploratory in the selection of the
diagnostic, not in the fitting of its parameters, and is kept out of the
confirmatory figures (fig5-fig7) accordingly -- it appears separately in
fig10.

**Extraction noise is heavy-tailed.** QQ of R - E pooled over Grid A root-0
reports (n = 9600): excess kurtosis 2.35; 1.20% of standardized residuals
beyond 3 sd (Normal: 0.27%) and 0.39% beyond 4 sd (Normal: 0.006%). The
tails are the arithmetic-slip population visible in the pilot transcripts.
Gaussian aggregators are therefore working with a misspecified likelihood
in the tails -- a further contributor to residual miscalibration for both
aggregators. (fig9.)

**Bias is estimated but not subtracted -- and the sensitivity check shows
it is immaterial.** The calibration bias estimate (-11.31 M EUR, 0.113
prior sd) is not applied in any aggregator. A back-of-envelope shift
argument initially suggested it could explain the k = 16 coverage
shortfall (0.927 vs nominal 0.95: a 0.53-posterior-sd shift alone predicts
0.918). The direct sensitivity analysis refutes that attribution: with
bias_cal subtracted from every report and all aggregators re-run, k = 16
coverage moves 0.927 -> 0.920 (no improvement, well within bootstrap
noise), and no Grid A or Grid B cell changes by more than ~0.04. The
calibration-split bias evidently does not transfer as a constant shift to
the evaluation worlds, so leaving it unsubtracted was immaterial either
way -- main conclusions are unchanged under the sensitivity variant. The
residual gap to nominal coverage where independence is true (0.92-0.93 vs
0.95, about 1.8 bootstrap se) is better attributed to the heavy-tailed
extraction noise below and to finite-calibration variance-estimation
error. The realistic benchmark for a well-specified Gaussian aggregator on
this data remains ~0.93-0.95, against which naive's 0.263 at n = 32 stands
unchanged.

## Figures

`figures/fig5_calibration_empirical.pdf` (Grid A calibration ratio vs n),
`figures/fig6_logscore_empirical.pdf` (Grid A NLL vs n),
`figures/fig7_roots_info_empirical.pdf` (Grid B NLL vs k),
`figures/fig8_precision_curve.pdf` (empirical precision curve vs the
independent- and correlated-extraction predictions),
`figures/fig9_extraction_noise_qq.pdf` (QQ of R - E), and
`figures/fig10_gamma_exploratory.pdf` (exploratory: naive / provenance /
provenance+gamma_cal coverage and calibration ratio on Grid A -- kept
separate from the confirmatory fig5/fig6 to preserve the visual boundary
between the prespecified analysis and post-hoc model refinement), in
`make_figures.py`'s style (same rcParams, sizing; one added color for the
dedup baseline, which has no analogue in the paper's Section 12 figures).

## Freeze status

Grids A and B, the calibration fit, the diagnostics, and the exploratory
gamma extension above are FROZEN as of this revision. No further analysis
changes except for demonstrated errors.

## Scope and limitations

- Single task family (fictional quarterly-revenue memos), single main model
  (claude-haiku-4-5-20251001), synthetic documents only -- per Section 10's
  required limitations.
- Gate 6 (Phase 0) was not met on its literal prespecified threshold and is
  retained as a disclosed limitation, not corrected: some of the extraction
  noise nu_hat2 driving these results reflects genuine Haiku arithmetic
  unreliability on 3-term percentage arithmetic, not only interpretive
  ambiguity in the documents. This is treated as a feature of the studied
  regime (Section 2 frames eta as measured, not designed), and rho_hat
  landing inside the workable band on real data is itself informative.
- Grids C (mixed-model-family gamma) and D (propagation chains, prediction
  13.3) and prediction 13.4 (the 2x2 similarity/ancestry design) are not
  run here -- deferred per the user's explicit scope decision, not blocked
  by anything found in A/B.
- The report-space dedup baseline's threshold was tuned entirely on k=1
  calibration data (see caveat above); its behavior in Grid A, where k=1
  throughout, is not yet a real test of the false-merge/false-split
  failure modes Section 5.2 predicts for it -- that requires Grid D or
  prediction 13.4's mixed-ancestry design.
