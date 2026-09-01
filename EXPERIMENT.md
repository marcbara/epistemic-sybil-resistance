# LLM Experiment: Epistemic Sybil Resistance (implementation brief)

**Instructions for Claude Code:** read this file fully, then the paper sections listed below. If you want to deviate from any design decision here, propose the change and the reason before coding. Build Phase 0 first and STOP for human review at the gates (Section 8) before spending on the full grids.

**Inputs provided alongside this file:**
- `main.tex`: the working paper. Read: Section 6 (Gaussian shared-root model, equicorrelation, Corollary 2), Section 7.2 (model parameters as evidence: the leakage problem), Section 12 (simulation design and metrics; this experiment is its empirical companion), Section 13 (predictions).
- `epistemic-sybil-resistance-latex.zip`: contains `make_figures.py` (reference implementation of the metrics and the figure style; reuse its rcParams and colors) and `figures/`.

**Goal:** an empirical section for the paper. Real LLM agents read controlled documents and estimate a latent quantity; three aggregators combine their reports; we test the paper's predictions 13.1 (multiplicity inflation), 13.2 (shared-root saturation), and 13.3 (provenance under propagation). Prediction 13.4 (the 2x2 similarity/ancestry design) is Phase 2 and out of scope here.

**Honesty rule:** a clean negative result is publishable content. Do not tune the setup toward the predictions after the pilot freeze. Report whatever comes out, with uncertainty.

---

## 1. Non-negotiable design constraints

1. **Synthetic worlds only (anti-leakage, paper Section 7.2).** The latent state Theta must be unknowable from pretraining: fictional entities with generated names, invented quantities. If Theta were a real-world fact, model parameters would act as an extra evidence root and the control of k breaks.
2. **Continuous Theta, point estimates.** Agents output a single number (plus a one-sentence rationale). Probability elicitation from LLMs is noisy and truncated; point estimates transfer the paper's Gaussian machinery (RMSE, coverage, NLL) directly.
3. **Documents must require inference, not lookup.** If the document states the target number verbatim, extraction noise nu^2 is ~0 and the shared-root regime degenerates into the clone regime. The value E_j must be implied by several partial cues that the agent has to combine (see Section 2).
4. **Strict information isolation.** One document per elicitation call. Stateless calls: no chat history, no other reports, no true Theta anywhere in the context. The only exceptions are the propagation chains of Grid D, where the input is exactly one upstream report by design.
5. **Everything seeded, cached, logged.** World generation is exactly reproducible from seeds. LLM API sampling is not guaranteed reproducible even with a seed parameter, so the standard is: fully auditable and replayable from cache, with every raw response stored.

## 2. Data-generating process (default spec; pilot may adjust, then freeze)

**World w:**
- Entity: fictional company with a generated name (random syllable composition; assert the name has no exact web-plausible collision pattern, e.g., avoid real suffixes like "Siemens").
- Latent state: Theta_w ~ Normal(500, 100^2), truncated to positive. Interpretation: total quarterly revenue in millions of EUR. The prior is known to the aggregators by construction (they know the DGP), not to the agents.

**Root j (primitive evidence):**
- E_wj = Theta_w + eps_wj, with eps_wj ~ Normal(0, sigma^2), default sigma = 50 (10 percent). This is the document-level distortion, the sigma^2 of the paper's Section 6.
- Document D_wj: a short memo (150-250 words) generated from templates. The cues jointly imply E_wj without stating it. Default cue mechanism: revenue split into three segments; segment A given directly, segment B given as a percentage of the (unstated) total, segment C given as growth over a stated prior-quarter figure. Include 2-3 numeric distractors (headcount, margin) that do not affect the answer.
- Unit test requirement: a deterministic solver must recombine the cues of every generated document to exactly E_wj. If the solver fails, the generator is buggy.
- Exact-arithmetic rule: draw cue parameters from discrete rational grids so that every displayed value implies E_wj exactly at displayed precision. Snap E_wj to the display grid before deriving cues, and use the solver-recovered value as E_wj everywhere in analysis. Never compute a hidden exact value and then round the cues independently: rounding residue would masquerade as extraction noise.
- Surface variation: seeded lexical template variants (ordering, synonyms). Do NOT use an LLM to stylize documents in Phase 0/1; if introduced later, add a verification pass that re-extracts every numeric cue and asserts equality.

**Report i on root j:**
- Input: instructions + D_wj only. Output schema: `{"estimate": <number in M EUR>, "rationale": "<one sentence>"}`. The rationale is required: it gives the report-space baselines something textual to work with.
- Main temperature T = 0.7 (noisy extraction, nu^2 > 0). A T = 0 cell is kept deliberately: it approximates the clone regime nu^2 ~ 0.
- Extraction noise eta is whatever the model produces; it is measured, not injected.

## 3. Experimental grids

Shared worlds where possible; reuse reports across cells by nesting (generate the maximum needed per root once, subsample for smaller cells).

- **Grid A (prediction 13.1; empirical analogue of paper Section 12.3).** k = 1, n in {1, 2, 4, 8, 16, 32}. Generate 32 reports on root 1 per world; nest the smaller n as subsets. Default W = 300 worlds.
- **Grid B (empirical analogue of Section 12.4).** n = 16 fixed, k in {1, 2, 4, 8, 16}, reports allocated evenly. Same worlds as Grid A. Per world this needs 16 roots; report counts per root can be nested (root 1 reuses Grid A's pool; roots 2..16 need at most 8, 4, 4, 2x4, 1x8 reports respectively, about 32 extra calls per world).
- **Grid C (Section 6.3, gamma; exploratory).** k = 1, n = 8, three conditions: (a) same model, T = 0.7; (b) mixed model families (2-3 distinct models, ideally distinct providers), T = 0.7; (c) same model, T = 0. We test whether mixed-family extraction reduces residual intraclass correlation relative to same-model extraction, and whether T = 0 reduces within-root extraction variance. No ordering is imposed as a validity requirement: families can share document-interpretation errors, one family can simply be a worse extractor, and T = 0 does not guarantee deterministic sampling. Because mixed families break homoscedasticity, report per-model bias and variance in this grid. Default W = 200.
- **Grid D (prediction 13.3, propagation).** k = 1. Per world, run 4 parallel chains from the same root: in each chain, agent 1 reads D_w1, and agent l reads only agent l-1's estimate and rationale (no document). Extend every chain to L = 8 and record the chain state at L in {1, 2, 4, 8}, nested by prefix, so the cost is 4 x 8 = 32 calls per world. At each L, aggregate the 4 chain-state reports, all descendants of one root, with and without lineage. Default W = 200.
- **Leakage control (mandatory).** 3 calls per world asking for the same estimate given only the entity name, no document. Run it on the ~100 calibration worlds, not only the 30 pilot worlds: a hard gate on a correlation estimated from 30 points trips far too often under the null. Pass criteria, primary first: (i) practical: no-doc predictions improve RMSE over the constant-prior-mean baseline by less than 10 percent; (ii) statistical: |corr(no-doc estimate, Theta)| below 2/sqrt(W_control), about 0.20 at 100 worlds. Report the CI and a permutation p-value as supplementary information; never freeze the experiment on a borderline p alone.

**Cost formula:** total calls ~ W_A x 64 (Grids A+B nested) + W_C x 24 + W_D x 32 + controls. Defaults give roughly 30-35k short calls. Trim option if budget matters: drop n = 32 and set W = 200 everywhere, which lands near 20k calls; halving W_C and W_D takes it near 15k. Use a small, cheap model as the main agent. Hard cost cap in config; the run must halt gracefully at the cap.

## 4. Calibration split

Reserve a disjoint set of ~100 worlds (never used for evaluation) to estimate, at frozen pilot settings:
- per-report total variance sigma_r^2 = Var(report - Theta),
- intraclass correlation rho_hat within shared-root blocks (this is the rho of Corollary 2),
- variance components separately, exploiting that the synthetic DGP knows E: sigma_hat^2 = Var(E - Theta) and nu_hat^2 = Var(report - E),
- per-report bias (subtract in the observation model if material).

These fitted values parameterize the aggregators. No parameter is fitted on evaluation worlds.

**Required diagnostic (candidate figure for the paper):** test whether rho_hat is approximately sigma_hat^2 / (sigma_hat^2 + nu_hat^2). If the identity holds, real LLM extraction empirically matches the random-effects structure behind Corollary 2, which upgrades the section from a simulation of the theory to a direct empirical connection between shared-root structure and real agent behavior. If it fails badly, that is itself a reportable result: LLM extraction does not follow the simple shared-root model. The provenance aggregator keeps using rho_hat directly either way.

## 5. Aggregators (all receive the true prior over Theta)

1. **Naive independent pooling.** Gaussian conjugate update treating each report as independent with variance sigma_r^2.
2. **Report-space deduplication (the Theorem 1 baseline).** Primary variant: embed the rationale only; including the estimate in the embedding can make independent reports that merely agree numerically look like clones. Cluster by cosine threshold; collapse each cluster to its mean; then pool as in 1. Threshold selection, pre-specified: sweep a fixed grid of thresholds on calibration worlds, pick the one minimizing mean NLL of the deduplicated aggregator, and freeze it before evaluation. Sensitivity variant: rationale + estimate embeddings. This is the report-only defense the paper predicts must fail somewhere.
3. **Provenance-aware.** Knows the true root assignment. Equicorrelated Gaussian posterior per block; equivalently the m_eff discount of Corollary 2 with rho_hat. This is the paper's proposal.
4. **Oracle (diagnostic upper bound, optional).** Full DGP parameters.

## 6. Metrics and analysis

Exactly the paper's Section 12.2: posterior RMSE, nominal 95 percent coverage, mean negative log score, calibration ratio C = RMSE / reported posterior sd. Add, for Grids A and C, the empirical precision curve: Var(block-mean error) versus m, to exhibit the ceiling (compare against 1/sigma^2 and 1/(sigma^2 + gamma nu^2)).

- Uncertainty: cluster bootstrap by world (1000 resamples) for every reported number and difference.
- Validity filter, pre-registered at pilot freeze: non-parseable outputs and estimates beyond mean_prior plus/minus 6 prior sd are invalid; report invalid rates per cell; analyses run on valid reports, with a sensitivity check including winsorized invalids.
- Figures in the exact style of `make_figures.py` (same rcParams, colors, sizes), named `fig5_*.pdf` onward.

## 7. Expected patterns (report against these; they are expectations, not gates)

- **13.1:** at k = 1, naive coverage decreases monotonically in n while provenance-aware coverage stays near 0.95; naive C grows.
- **12.4 analogue:** at fixed n = 16, the ideal aggregator's NLL and RMSE improve with k; naive confidence does not track k.
- **13.2:** block-mean precision flattens toward a ceiling in m. How the ceiling moves across Grid C conditions is exploratory (see Grid C); no ordering is imposed.
- **13.3:** with chains, lineage-aware aggregation is better calibrated than report-only on the same terminal reports.

## 8. Phase 0: pilot, with stop-and-review gates

Run 30 worlds, k = 1, n = 8, main model, T = 0.7, plus the leakage control. Then STOP and present:

1. Parse + validity rate >= 95 percent.
2. Extraction noise is real: within-document report sd meaningfully > 0 (not a clone regime at T = 0.7).
3. rho_hat in a workable band, roughly 0.2 to 0.95.
4. Leakage control passes.
5. Reports approximately unbiased around E_j (small bias is fine if stable; it gets calibrated out).
6. The task is informative but nontrivial: RMSE(report, E) materially below the prior standard deviation (default: below 0.5 x prior sd), while within-root report sd stays materially above zero. Gates 2 and 3 check that noise exists; this one checks that agents actually extract signal instead of guessing.

Iterate the document generator and elicitation prompt until the gates pass, then freeze templates, prompts, thresholds, and the validity filter. Everything after the freeze is confirmatory.

## 9. Engineering requirements

- Python 3.11+, config-driven (`config.yaml` holds every knob named in this brief: sigma, prior, grids, W, temperatures, models, thresholds, cost cap, master seed).
- Modules: `worldgen.py` (with the exact-recombination unit tests), `elicit.py` (async batching, retries with backoff, strict JSON validation with one repair retry, response cache keyed by sha256(model, temperature, prompt, seed)), `aggregate.py`, `analyze.py`, `figures.py`.
- Raw log: one JSONL record per call (world, root, indices, model, T, prompt hash, full response, tokens, timestamp). Runs resumable from cache.
- Seeds: world seed = f(master_seed, world_id); nothing nondeterministic outside model sampling.
- `--dry-run` mode that mocks the model with the Gaussian DGP itself (useful to test the whole pipeline end to end and to reproduce Section 12 as a sanity check).
- Main agent model: small and cheap (Haiku-class); mixed-family condition uses whatever additional API keys are configured, 2-3 distinct models.

## 10. Deliverables (definition of done)

1. `results/`: raw JSONL + processed parquet per grid.
2. `figures/fig5_*.pdf` onward, style-consistent with the paper.
3. `RESULTS.md`: tables mirroring the format of paper Sections 12.3-12.5, one subsection per grid, verdict per expected pattern with bootstrap CIs, invalid rates, and total cost, plus the rho identity diagnostic (predicted versus estimated rho) with its own figure.
4. `draft_empirical_section.tex`: a paper-ready subsection in the voice of `main.tex`. Prose rules: flowing paragraphs, no bullet lists, no em dashes, every quantitative claim tied to a number in `results/`, limitations stated (single task family, synthetic documents, one main model).

## 11. Out of scope for now

- Prediction 13.4 (the 2x2 of semantic similarity versus evidential ancestry). Cell B, similar reports from independent roots, requires worlds engineered so that independent roots converge naturally; defer to Phase 2.
- Strategic or truthful provenance (paper Section 10).
- Any human-subject or real-world-fact condition.
