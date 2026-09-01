# Reproducibility

## Confirmatory vs. exploratory

**Confirmatory** (design, thresholds, and validity filter frozen before
evaluation data were collected):

- Grid A (report multiplicity, Section 8.2) and Grid B (root count,
  Section 8.3).
- The 2x2 similarity-vs-ancestry design (Section 8.5) is confirmatory with
  respect to its own frozen design: its renderer, clustering algorithm,
  balanced-threshold objective, and evaluation metrics were pre-registered
  in `EXPERIMENT_NOTES.md`'s "2x2 pre-registration" section after a pilot
  and before the balanced-calibration and evaluation data were collected.

**Exploratory:**

- The correlated-extraction specification (gamma) existed in the
  theoretical model before the experiment, but its empirical evaluation
  was prompted by the observed failure of the independent-extraction
  specification on Grid A -- an out-of-sample check of an exploratory
  choice, not a pre-registered confirmatory test. All of its parameters
  (`gamma_cal`) were nonetheless estimated exclusively on the calibration
  split, never on Grid A.
- The gamma-aware provenance aggregator (Section 8.4) is exploratory for
  the same reason and is kept visually separate from the confirmatory
  figures (fig10, not fig5/fig6).
- The oracle threshold sweep (both the coarse grid and the exact,
  gap-free version over every distinct observed cosine value) is
  diagnostic and post-hoc. It is never used to select or freeze any
  threshold; the frozen thresholds are the legacy A/B value (0.80) and
  the balanced 2x2 value (0.70), both fit on calibration data alone.

This wording is carried over from, not reinterpreted from, the frozen
methodological record in `EXPERIMENT_NOTES.md`, `RESULTS.md`, and
`RESULTS_2X2.md`.

## Calibration / evaluation separation

Every fitted parameter (`sigma_r2`, `sigma_hat2`, `nu_hat2`, `rho_hat`,
`gamma_cal`, the leakage-control statistics, the legacy dedup threshold)
is fit on the 100-world calibration split (`results/raw/calibration.jsonl`,
`world_id_ranges.calibration` in `config.yaml`). The 2x2 design's balanced
dedup threshold is fit on its own 60-world calibration split
(`results/raw/two_by_two_dedup_calibration.jsonl`,
`world_id_ranges.two_by_two_dedup_calibration`). Neither is refit on, or
selected using, any evaluation-set outcome. World-id ranges are disjoint
by construction (`config.yaml`'s `world_id_ranges` block); no world_id
used for calibration is ever reused for evaluation.

## What reproduces exactly, and what does not

**Reproduces exactly, from frozen data, no API access:** every number and
figure `reproduce_paper.py` regenerates -- the calibration fit, both dedup
thresholds, Grid A/B, the correlated-extraction diagnostics and
out-of-sample correction, the 2x2 evaluation and both oracle sweeps, and
the factorial contrasts. This is a deterministic function of
`results/raw/*.jsonl`: `reproduce_paper.py`'s own integrity check confirms
bit-identical output against the values committed in
`results/processed/*.json`.

**Cannot be guaranteed to reproduce exactly in the future:** anything
requiring a live call to `claude-haiku-4-5-20251001` via the Anthropic
API (`src/pilot.py`, `src/calibration.py`, `src/grid_ab.py`,
`src/two_by_two.py`). Hosted model providers can change weights or serving
behavior behind a fixed model identifier without notice, so rerunning
these scripts is a *replication* of the experiment, not a guaranteed
bit-for-bit *reproduction* of the frozen numbers. This is why the frozen
raw outputs in `results/raw/` are committed rather than treated as
regenerable: they are the primary record.
