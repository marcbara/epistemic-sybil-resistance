# Phase 0 pilot: decisions, deviations, and freeze record

**Status: FROZEN for Grids A and B.** Templates, prompts, thresholds, and the
validity filter below are frozen as of this pilot. Grids C and D are
explicitly deferred pending their own prerequisites (heterogeneous model
access for C; a distinct chain-propagation generative structure for D) and
are not blocked by this freeze -- they were never gated on it.

**Update (post Grid A/B): results and analyses also frozen.** Grid A/B data
collection, the calibration fit (including gamma_cal), the post-hoc
diagnostics, and the exploratory gamma-aware aggregator are frozen as final
per RESULTS.md's freeze note. The next experiment is the 2x2 similarity x
ancestry design (paper prediction 13.4), which gets its own pilot gates and
freeze cycle.

## 2x2 pre-registration (frozen before the 60-world balanced calibration run)

The 2x2 pilot (30 worlds, redesigned after a prompt-level style manipulation
was found to confound extraction quality -- see `src/render.py`'s module
docstring) passed all 3 gates: validity 100%, renderer template separation
gap=0.351, and the core mechanism check Delta_BC = E[cos(B)] - E[cos(C)] =
0.242, 95% CI [0.216, 0.270] (cluster-bootstrapped by world). Before
spending on the 60-world balanced-calibration run or the 200-world
evaluation, four things are frozen in writing, unchanged from the pilot:

1. **The renderer preserves informational content exactly, by construction.**
   `render.render_rationale(style, seed)` never reads a report's actual
   estimate or raw rationale -- it deterministically selects a canned
   sentence from a fixed template pool (`render.S1_TEMPLATES` /
   `render.S2_TEMPLATES`) describing the one true abstract structure every
   document shares (a directly-stated segment, a segment expressed as a
   percentage of it, a segment adjusted for growth over a prior figure).
   Every template in both pools states this same true structure; only
   vocabulary and sentence form differ between S1 and S2. No template adds,
   omits, or alters a number, a relationship, or a conclusion. This was
   true in the pilot and is architecturally guaranteed to remain true --
   there is no code path by which it could vary with a report's content.

2. **Clustering algorithm frozen: connected components at cosine >=
   threshold** (`aggregate._cluster_by_threshold`, single-linkage via
   union-find, unchanged from A/B). Not re-derived or swapped for the 2x2.

3. **Balanced-threshold selection: exclusively on the 60 calibration
   worlds, minimizing mean NLL with equal weight across cells A, B, C, D.**
   No re-optimization against false-merge/false-split rates, before or
   after seeing evaluation data. The selected threshold is frozen the
   moment calibration fitting completes, before the 200-world evaluation
   collection begins. The legacy A/B threshold (0.80) is retained
   unmodified as a separate historical-baseline comparator, not re-tuned.

4. **Evaluation primary metrics, predefined:** false merge rate in cell B,
   false split rate in cell C, mean inferred cluster count vs. true root
   count per cell, coverage, NLL, and calibration ratio C -- reported for
   naive, provenance-aware, provenance+gamma_cal, legacy dedup (t=0.80),
   and balanced-calibrated dedup.

**Oracle threshold sweep (0.30-0.95) is strictly diagnostic**, run after
evaluation, and answers one question: does any single threshold t achieve
FMR_B(t) approx 0 AND FSR_C(t) approx 0 simultaneously? It does not select
or influence any frozen threshold above.

**Presentation framing, carried into RESULTS.md and the paper draft:** the
2x2 is a controlled representation stress test -- it demonstrates that a
representation-only transformation (Theta, E, the extraction process, and
the estimate all held fixed) can invert what report-only similarity implies
about evidential ancestry. It is not a claim about the typical rationale
distribution any production system would see.

No further design changes between this freeze and the evaluation results.

## Deviations from a literal reading of EXPERIMENT.md

1. **Cue mechanism for segment B** (Section 2). The brief specifies B as "a
   percentage of the (unstated) total", i.e. a self-referential
   E = A + p*E + Q*(1+g). The first pilot run showed Haiku 4.5 correctly
   recognizing this structure but making arithmetic slips solving the
   implied division, inflating nu_hat^2 to ~61,000 (report sd ~247M EUR) --
   model computation error, not extraction/interpretation noise. Segment B
   was redefined as a percentage of stated segment A: E = A*(1+p) + Q*(1+g).
   This needs only multiplication and addition (exact for any finite-decimal
   p, A, Q, g, no curated grid needed) and is arithmetic Haiku can reliably
   execute, while still requiring the agent to combine three cues rather
   than read one number. See `src/worldgen.py` module docstring for the
   full derivation.

2. **Growth rate g excludes 0%.** Pilot transcripts showed the model
   sometimes misreads "grew 0% over Q" as "contributes 0" rather than
   "equals Q" -- a template-wording artifact concentrated at that single
   point (one world contributed 5 of the pilot's 10 largest errors before
   this fix). `g_grid_pct_range` now excludes |g| < 2.

3. **Entity name collision avoidance is heuristic, not verified.** No web
   lookup is performed; names are generated from invented syllable pairs
   plus a generic sector suffix, checked only against a small blocklist of
   well-known real company names/fragments. This is a known limitation, not
   a guarantee of zero web-plausible collision. The leakage control is where
   this gets checked empirically: if invented names carried real-world
   signal, no-doc estimates would correlate with Theta, and gate 4 would
   catch it. It doesn't (corr ~ -0.03, n=30; full statistical gate pending
   the 100-world calibration run).

4. **Document length padding.** The literal cue sentences alone run ~50
   words, short of the brief's 150-250 word target (Section 2). Non-numeric
   filler sentences (business-memo boilerplate, no cues, no numbers) are
   assembled around the core paragraph to reach a target sampled in
   [170, 220] words per document.

5. **Elicitation prompt hardening**, all responses to observed failure
   modes, not to steer results:
   - `max_tokens` raised from 300 to 500: the model's verbose rationale was
     truncating the JSON output, producing the pilot's first-run 6.7%
     invalid rate.
   - Explicit instruction not to second-guess or "adjust" the computed sum:
     transcripts showed the model computing the three segments correctly,
     summing them correctly in the rationale text, then appending an
     unmotivated "adjustment" with no basis in the document.
   - Rationale capped at 25 words in the prompt ("a plain description of
     which segments you used, with no arithmetic and no numbers"), with a
     deterministic word-level truncation at 30 words as a structural
     safety net (`elicit._truncate_rationale`, `rationale_truncated` logged
     per record) -- not a repair call, so it costs nothing extra. Motivation:
     a free-running derivation is unneeded chain-of-thought, adds cost and
     variance, and would confound the report-space dedup baseline (Section
     5.2), which embeds the rationale text. At freeze, the frozen prompt
     produces rationales averaging 19 words with zero truncations needed
     across 240 report calls.

## Gate 3 measurement note

Gate 3's rho_hat uses the known-DGP variance decomposition
(sigma_hat^2 = Var(E-Theta), nu_hat^2 = Var(report-E), both computable
because the synthetic DGP's ground truth is known), per Section 4. A
cross-world one-way ANOVA ICC was also implemented (`gates.one_way_icc`) but
is NOT used for gate 3: with k=1 root per world and Theta varying
world-to-world, grouping by world conflates Var(Theta) into the
between-group term and inflates rho_hat well above the true
conditional-on-Theta value (verified empirically: ~0.95 vs a true 0.80 in a
synthetic check). That estimator is only a valid diagnostic in designs where
multiple roots share a common, known Theta (Grid B/C, k>1) -- see the
docstring in `src/gates.py`.

## Model choice

- **claude-haiku-4-5-20251001** (main agent, frozen). Passes gates 1-5
  cleanly on the frozen prompt (validity 100%, within-doc report sd 26.0M,
  rho_hat 0.278, leakage corr -0.03, bias +3.7M).

  **Gate 6: prespecified threshold not met; experiment retained because
  reports remained materially informative relative to the prior baseline.**
  RMSE(report, E) landed at 0.63-0.70x prior_sd across pilot runs (frozen
  run: 0.697), against a prespecified <0.5x threshold. This is not
  redefined as a pass. Inspection of the largest-error cases shows
  unambiguous documents (all three cues clearly stated, correct arithmetic
  shown in the rationale) paired with a wrong final JSON estimate --
  genuine Haiku computational unreliability on 3-term arithmetic with
  percentages, not a fixable document defect, and not corrected or
  postprocessed: it is literally eta, the extraction-noise process the
  experiment measures (Section 2: "extraction noise eta is whatever the
  model produces; it is measured, not injected"). A rough cross-check: with
  sigma=50 and an observed extraction RMSE of ~63-65, the implied
  rho = sigma^2/(sigma^2+nu^2) is approximately 50^2/(50^2+64^2) ~ 0.38 --
  consistent with the fitted rho_hat and squarely inside the workable band
  (neither near-independent nor near-clone). The elevated noise is treated
  as a feature of this regime, not a defect to engineer away.

- **claude-sonnet-5** was tried as an alternative and rejected. It rejects
  an explicit `temperature` parameter (fixed/managed sampling) --
  `src/elicit.py` falls back to omitting `temperature` for any model that
  400s on it. With temperature uncontrollable, all 8 reports per document
  came back near-identical (nu_hat^2 ~ 0, rho_hat ~ 1.0): the clone regime,
  not the noisy T=0.7 extraction regime Section 2 requires as the main
  condition. Higher arithmetic reliability does not compensate for this --
  it would make the main experiment a near-trivial demonstration of the
  clone regime rather than a test of shared-root saturation under genuine
  extraction noise. claude-haiku-4-5-20251001 remains the main agent.

## Cost

Five full pilot iterations plus one Sonnet test, under $3 total spent
against the $10 pilot cap (`results/raw/pilot.jsonl` has the full audit
trail; the sha256 response cache means re-running an unchanged config costs
nothing).

## Status

5/6 Phase 0 gates pass; gate 6 is a disclosed, retained limitation per
above -- not redefined as a pass, not iterated further once traced to
genuine extraction noise rather than a template defect. Frozen for Grids A
and B. C and D remain open (see top of file).
