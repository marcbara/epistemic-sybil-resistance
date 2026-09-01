# Results: 2x2 similarity x ancestry design (prediction 13.4)

Empirical companion to `paper/latex-src/main.tex` prediction 13.4 and
Theorem 1's non-identifiability result. Pre-registration (renderer,
clustering algorithm, balanced-threshold objective, evaluation metrics) is
recorded in `EXPERIMENT_NOTES.md` before any calibration or evaluation data
was collected. Main agent, DGP, and templates are unchanged from Grids A/B;
`sigma_r2`, `sigma_hat2`, `nu_hat2`, and `gamma_cal` are reused from the
A/B calibration fit (properties of the task/model, not re-derived here).

**Framing.** This is a controlled representation stress test, not a claim
about the typical rationale distribution a production system would see. It
demonstrates that a representation-only transformation -- Theta, E, the
extraction process, and the estimate all held fixed -- can invert what
report-only similarity implies about evidential ancestry.

## Design

Paired by world: every world generates 4 roots (root_id 0-3). A single
frozen elicitation (byte-identical to the A/B prompt) produces
estimate + raw_rationale for each of 7 report slots per world (4 on root 0,
1 each on roots 1-3). A deterministic post-hoc renderer
(`src/render.py`) -- which never reads a report's estimate or raw
rationale -- then supplies the S1/S2 text actually embedded, so ancestry
(which roots feed a cell) and representation (which renderer style is
applied) are manipulated independently by construction:

| | Shared root | Independent roots |
|---|---|---|
| **Similar** representation | A | B |
| **Dissimilar** representation | C | D |

A first pilot attempt manipulated representation via elicitation-time
prompt phrasing and found this changed extraction RMSE by 0.80x between
styles -- a genuine confound. The redesign (single elicitation, post-hoc
deterministic rendering) removed it: estimate-quality equivalence across
styles is guaranteed by construction, not tested for.

## Pilot (30 worlds, $0.00 -- served entirely by A/B-prompt cache hits)

- Gate 1 (validity >= 95%): 100%, PASS.
- Gate 2 (renderer template separation, sanity check on the templates
  themselves): gap = 0.351 (within-style cosine 0.759 vs cross-style
  0.408), PASS.
- Gate 3 (Delta_BC inversion, the core mechanism check): Delta_BC =
  E[cos(B)] - E[cos(C)] = 0.242, 95% CI [0.216, 0.270] (cluster bootstrap
  by world), PASS. Independent-root reports rendered in the *same* style
  already look more alike than same-root reports rendered in *different*
  styles -- the intended inversion, with Theta, E, and the estimate
  untouched.

## Balanced dedup threshold (60 calibration worlds, disjoint from
evaluation, $0.33)

Threshold selected by minimizing mean NLL with equal weight across cells
A/B/C/D (every calibration world contributes exactly one instance of each
cell, so plain pooling already weights them equally), swept over
[0.30, 0.95]. Unlike A/B's calibration (single-root-only, threshold pinned
to the grid's lower boundary because merging was always optimal there),
this fit has genuine interior curvature -- cells push in different
directions, so a nontrivial trade-off is being resolved:

| t | Balanced mean NLL |
|---:|---:|
| 0.30 | 5.377 |
| 0.50 | 5.322 |
| **0.70** | **5.310 (minimum)** |
| 0.80 (legacy) | 5.317 |
| 0.95 | 5.336 |

**Frozen balanced threshold: 0.70**, selected before any evaluation data
was collected. The legacy A/B threshold (0.80) is kept unmodified as a
historical-baseline comparator.

## Evaluation (200 worlds, $1.09, 1400/1400 calls valid)

### Mechanism diagnostics (predefined primary metrics)

Using the frozen balanced threshold (0.70):

| Metric | Value |
|---|---:|
| False merge rate, cell B (independent roots, similar representation) | **0.622** |
| False split rate, cell C (shared root, dissimilar representation) | **0.846** |

Mean inferred cluster count vs. true root count:

| Cell | True roots | Mean inferred clusters |
|---|---:|---:|
| A (similar, shared) | 1 | 1.62 |
| B (similar, independent) | 4 | 1.69 |
| C (dissimilar, shared) | 1 | 3.08 |
| D (dissimilar, independent) | 4 | 3.09 |

**Factorial contrast on inferred cluster count (paired bootstrap by world,
n=200, 5000 resamples) is the cleanest statistic in the whole experiment:**

| Contrast | Estimate | 95% CI |
|---|---:|---:|
| Representation effect, (C+D)/2 - (A+B)/2 | **+1.425** | [1.363, 1.485] |
| Ancestry effect, (B+D)/2 - (A+C)/2 | +0.040 | [-0.045, 0.120] |
| Interaction, (D-C) - (B-A) | -0.060 | [-0.185, 0.060] |

The tested report-space deduplication mechanism responds to representation
style with an effect of +1.43 clusters and a CI that excludes zero by a
wide margin; it responds to a 4x change in true root count with an effect
of +0.04 clusters and a CI centered on zero. Inferred cluster count tracks
*representation style*, not *true root count*: A and B (both "similar")
collapse to ~1.6-1.7 clusters regardless of whether the true root count is
1 or 4; C and D (both "dissimilar") inflate to ~3.1 clusters regardless of
whether the true root count is 1 or 4. The tested mechanism -- rationale
embedding plus cosine-threshold clustering -- tracks representation rather
than evidential ancestry; this experiment provides an operational
illustration of the report-only non-identifiability result (Theorem 1),
not a proof that every possible report-space method must fail this way.
(fig12.)

**Confirmatory Delta_BC on the 200 evaluation worlds** (the pilot's check,
repeated on the disjoint evaluation set, changing nothing and selecting
nothing): mean cos(B) = 0.779, mean cos(C) = 0.531, Delta_BC = **0.248**,
95% CI [0.236, 0.259] -- closely reproducing the pilot's 0.242
[0.216, 0.270] on an independent, five-times-larger world set.

### Downstream calibration effect

| Aggregator | A cov. | B cov. | C cov. | D cov. | A C | B C | C C | D C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive | 0.785 | 0.925 | 0.785 | 0.925 | 1.639 | 0.994 | 1.639 | 0.994 |
| Provenance-aware | 0.920 | 0.925 | 0.920 | 0.925 | 1.130 | 0.999 | 1.130 | 0.999 |
| Provenance + gamma_cal | 0.950 | 0.925 | 0.950 | 0.925 | 0.900 | 0.999 | 0.900 | 0.999 |
| Dedup (legacy t=0.80) | 0.880 | 0.970 | 0.810 | 0.945 | 1.165 | 0.885 | 1.463 | 0.930 |
| Dedup (balanced t=0.70) | 0.945 | 0.970 | 0.820 | 0.960 | 0.980 | 0.852 | 1.395 | 0.886 |

Naive and provenance-aware are, correctly, *identical* on A vs. C and on
B vs. D -- neither aggregator looks at representation, only at true
ancestry, so a representation-only manipulation cannot move them (a useful
internal sanity check: A and C share true ancestry structure, k=1; B and D
share true ancestry structure, k=4; both aggregators are blind to which
renderer style was applied, as they should be). Provenance + gamma_cal
again holds the tightest, most stable calibration ratios throughout,
consistent with Grid A/B.

Dedup, at both thresholds, is *not* invariant to representation despite
identical ancestry. Cells A and C share exactly the same evidence and the
same estimates -- root 0's four reports, unchanged -- and differ only in
which renderer style was applied downstream; the same holds for B and D.
This makes the C-A and D-B contrasts closer to a causal read on
representation's effect than a comparison of two separately-measured
numbers, and both are paired-bootstrapped by world (n=200, 5000
resamples) using the balanced-threshold dedup aggregator:

| Contrast (same evidence & estimates, representation only differs) | Delta coverage | Delta NLL | Delta calibration ratio C |
|---|---:|---:|---:|
| C - A (shared root: dissimilar vs. similar) | **-0.125** [-0.175, -0.080] | **+0.223** [+0.099, +0.373] | **+0.416** [+0.333, +0.513] |
| D - B (independent roots: dissimilar vs. similar) | -0.010 [-0.035, +0.015] | -0.191 [-0.243, -0.142] | +0.034 [-0.019, +0.083] |

For the shared-root pair, every metric moves in the harmful direction with
a CI excluding zero: switching only the rendered representation of
identical evidence costs 12.5 points of coverage and pushes calibration
ratio up by 0.42. For the independent-root pair, coverage and calibration
ratio are not distinguishable from no effect (CIs include zero), and NLL
moves in the *favorable* direction for the dissimilar-representation cell
(D) -- because for cell D dissimilar representation happens to align with
the true structure (four genuinely independent roots), so it correctly
resists merging, whereas cell B's similar representation over-merges
genuinely independent evidence. This is the same asymmetry visible in the
raw numbers, now with confidence intervals attached: false-splitting a
shared root (harmful, C-A) is the dangerous direction; false-merging
independent roots (comparatively benign, D-B) discards information -- the
resulting posterior is too wide, not too narrow -- and can even help when
the representation manipulation happens to correlate with the true
structure, as it does in D. (fig13.)

## Oracle threshold sweep: coarse grid, then exact (both diagnostic only,
neither used to select any frozen threshold)

**Coarse grid** [0.30, 0.95], 14 points, asking whether any single t
achieves FMR_B(t) approx 0 and FSR_C(t) approx 0 simultaneously:

| t | FMR_B | FSR_C |
|---:|---:|---:|
| 0.30 | 1.000 | 0.000 |
| 0.50 | 1.000 | 0.543 |
| 0.65 | 0.856 | 0.782 |
| 0.70 (balanced) | 0.622 | 0.846 |
| 0.80 (legacy) | 0.380 | 0.889 |
| 0.95 | 0.250 | 0.912 |

FMR_B and FSR_C move in opposite directions across the swept range; the
closest joint point to the ideal (0,0) on this grid is t=0.85, with
FMR_B=0.250 and FSR_C=0.912. This supports the claim that no threshold *in
the tested grid* resolves both cells -- a weaker claim than "no threshold
can," since the grid has gaps.

**Exact sweep**, closing that gap: every one of the 29 distinct
pairwise cosine-similarity values actually observed in cells B and C
defines a cut point (30 candidate thresholds, the midpoints between
consecutive distinct values plus the two extremes), so this covers every
threshold that could possibly produce a different clustering on this
evaluation set -- there is no threshold value between two evaluated cut
points that would change any pairwise merge/split decision.

- min_t max(FMR_B(t), FSR_C(t)) = **0.846**, at t=0.6936.
- min_t sqrt(FMR_B(t)^2 + FSR_C(t)^2) = **0.945**, at t=0.9139.

Even choosing the threshold that minimizes the *worse* of the two error
rates, that worse rate is 84.6%. **No threshold for this similarity-based
clustering rule simultaneously achieves low false-merge and false-split
error on the observed evaluation set** -- exact over the full space of
distinguishable cut points on this data, not an artifact of grid spacing.
(fig11 shows the coarse-grid curve; the exact sweep's minimax point is
consistent with it.)

## Verdict on prediction 13.4

**Confirmed.** The tested report-space deduplication mechanism -- rationale
embedding plus cosine-threshold clustering, the report-only defense
Section 5.2 specifies as the baseline to test -- tracks representation
rather than evidential ancestry: a representation-only manipulation moves
inferred cluster count by +1.43 (CI excluding zero) while a 4x change in
true root count moves it by +0.04 (CI centered on zero), and no threshold
in the exact, gap-free sweep gets both its false-merge and false-split
error below 0.85. This experiment provides an operational illustration of
the report-only non-identifiability result (Theorem 1) for one concrete
mechanism and one task family -- not a claim that no report-space method
could ever succeed. Provenance-aware aggregation is unaffected by the
representation manipulation by construction (it uses ancestry, not report
content), and provenance + gamma_cal remains the best-calibrated
aggregator throughout, consistent with Grids A/B.

## Figures

`figures/fig11_tradeoff_curve.pdf` (FMR_B vs. FSR_C across the oracle
sweep, legacy and balanced thresholds marked), `figures/fig12_cluster_counts.pdf`
(true root count vs. mean inferred clusters per cell), and
`figures/fig13_coverage_by_cell.pdf` (coverage by aggregator x cell).

## Cost

Pilot: $0.00 (cache). Balanced calibration: $0.33 (420 calls). Evaluation:
$1.09 (1400 calls). Final statistical round (factorial contrasts,
confirmatory Delta_BC, paired downstream contrasts, exact oracle sweep):
$0.00, no new API calls. Total: **$1.42**, against the ~$1.60 estimated and
the existing $10 cap -- no new budget authorization needed.

## Freeze status

FROZEN. No further analysis changes except demonstrated errors. Grid C
(mixed-model-family gamma) and Grid D (propagation chains, prediction
13.3) remain deferred, available as a later extension (e.g. if review
requests cross-model generalization) but not required to close the main
empirical arc: Grid A (multiplicity inflation), Grid B (report count vs.
root count), the correlated-extraction diagnosis and correction (gamma_cal,
Section 6.3), and the 2x2 (similarity vs. ancestry) together confirm
predictions 13.1, the 12.4 analogue, and 13.4.

## Scope and limitations

- This is a controlled stress test of representation, not a survey of
  naturally occurring rationale similarity in deployed systems.
- The deterministic renderer's content-preservation guarantee is
  architectural (it never reads report content) rather than verified per
  report -- there is nothing to verify, by construction, but this also
  means the templates' truth (that every document has exactly this
  three-segment structure) is itself a property of this task family, not
  demonstrated to generalize.
- The lightweight hashed bag-of-words embedding (not a full semantic
  embedding model) is used throughout, as in A/B -- see
  `EXPERIMENT_NOTES.md` and `src/embed.py`'s docstring for that tradeoff.
- Single task family, single main model, synthetic documents only, as in
  A/B.
