# Epistemic Sybil Resistance -- reproducibility repository

This repository contains the theory, code, frozen model outputs, and
analysis pipeline for:

> **Epistemic Sybil Resistance: Multiplying AI Agents Without Multiplying
> Evidence**, Marc Bara, preprint, August 2026 (`paper/latex-src/main.tex`,
> compiled PDF at `paper/latex-src/main.pdf`).

It supports the paper's Section 7 (synthetic model validation) and Section
8 (empirical study with LLM agents: Grid A, Grid B, the correlated-
extraction diagnosis and correction, and the 2x2 similarity-vs-ancestry
design), plus their appendices.

## What this is and is not

This is **not** a general-purpose framework or a second project bolted
onto the paper. It is the exact code that produced the paper's reported
numbers, tables, and figures, plus the frozen model outputs those numbers
were computed from. A reviewer can inspect precisely how every experiment
was run and reproduce every reported table and figure from the frozen
outputs without API credentials.

## Repository structure

```
config.yaml                 All experiment knobs: DGP parameters, model
                             config, seeds, world-id ranges (splits),
                             cost caps, aggregator/threshold-grid settings.
requirements.txt            Pinned Python dependencies.
reproduce_paper.py           One-command reproduction entry point (see below).

src/                        All experiment and analysis code.
  worldgen.py                Synthetic-world generator (entity names,
                              Theta, exact-recombination cue mechanism).
  elicit.py                   Elicitation client: the exact system prompts
                              used (report_system_prompt), sha256 response
                              cache, retries, JSON validation, cost tracking.
  render.py                    Deterministic post-hoc rationale renderer
                              used by the 2x2 design (never calls a model).
  gates.py, aggregate.py, embed.py
                             Pure statistics: Phase-0 gates, the four
                             aggregators (naive / provenance-aware /
                             report-space dedup / oracle), bag-of-words
                             embedding for the dedup baseline.
  pilot.py, calibration.py, grid_ab.py, two_by_two.py
                             Data-collection scripts (make live API calls;
                             need credentials -- see "Optional" below).
  fit_dedup_threshold.py, analyze.py, diagnostics.py, gamma_analysis.py,
  figures.py, two_by_two_analysis.py, two_by_two_figures.py,
  two_by_two_stats.py
                             Analysis scripts: pure functions of the
                             frozen data in results/raw/, no API calls.
                             reproduce_paper.py calls these directly.

tests/                      Unit tests for the pure logic (worldgen exact
                             recombination, gates, aggregators, renderer).

results/
  raw/*.jsonl                Frozen model outputs: one JSON record per API
                              call (prompt, model, temperature, seed, raw
                              response, parsed estimate/rationale, tokens,
                              cost, timestamp), the primary evidentiary
                              artifact this package preserves.
  processed/*.json, *.md      Frozen analysis outputs computed from raw/:
                              calibration fit, Grid A/B results, gamma
                              diagnostics, 2x2 results and statistics.
  cache/                       Internal sha256 response cache used only
                              during live collection to avoid re-billing
                              an already-answered prompt. Not tracked, not
                              needed for reproduction (see .gitignore).

figures/                    All PDF figures referenced in the paper.
paper/                      main.tex, the compiled PDF, and figures/ as
                             embedded in the paper (a copy of the above).

benchmark/                  ESB, the Epistemic Sybil Benchmark: the frozen
                             Grid A / Grid B evaluation data repackaged as
                             a standalone benchmark with a documented
                             aggregator interface (instances + held-out
                             truth + scoring command + baseline adapters).
                             Plug in your own aggregator and measure its
                             coverage-collapse curve. See benchmark/README.md.

config.yaml, EXPERIMENT.md, EXPERIMENT_NOTES.md, RESULTS.md, RESULTS_2X2.md
                             The experimental brief, the frozen design
                             decisions and their rationale, and the full
                             results writeups the paper's numbers are
                             drawn from.
REPRODUCIBILITY.md            What is confirmatory vs. exploratory, what
                             reproduces exactly vs. requires live calls.
```

## Software requirements

- Python 3.11 or later (developed and tested on 3.12.10).
- A LaTeX distribution (e.g. MiKTeX or TeX Live) only if you want to
  recompile `paper/latex-src/main.tex` yourself; the compiled PDF is
  already included.

## Installing dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Reproducing the paper's results (no API credentials needed)

```bash
python reproduce_paper.py
```

This rebuilds every empirical result and figure in the paper from the
frozen outputs in `results/raw/*.jsonl`: the calibration fit, the legacy
and balanced dedup thresholds, Grid A and Grid B, the correlated-
extraction diagnostics and the out-of-sample gamma-aware correction, the
2x2 design's evaluation and both oracle threshold sweeps, the factorial
contrasts, and every figure referenced in Section 8. It ends with an
integrity check comparing the freshly reproduced values against the
values already committed in `results/processed/*.json`, at full stored
precision, and exits non-zero if any check falls outside tolerance. No
network access or API key is required or used anywhere in this path.

The pure Monte Carlo simulation behind Section 7's main table and Figures
1-4 (`paper/latex-src/make_figures.py`) is separately, trivially
reproducible: it is a fixed-seed simulation with no model calls involved
at all, and is not re-run by `reproduce_paper.py`.

## Optionally rerunning the model calls yourself

If you have a valid `ANTHROPIC_API_KEY`, you can regenerate the raw
outputs from scratch instead of using the frozen ones:

```bash
cp .env.example .env   # then add your key
python src/pilot.py                                  # Phase 0 pilot
python src/calibration.py                              # calibration split
python src/grid_ab.py collect --range grid_ab           # Grid A/B
python src/two_by_two.py collect --range two_by_two_dedup_calibration
python src/two_by_two.py collect --range two_by_two_eval
```

Each script prints its estimated and actual spend and halts at the
configured cost cap (`config.yaml`'s `cost` block). **Rerunning hosted LLM
calls is a replication, not a guaranteed bit-for-bit reproduction**: model
providers can update weights or serving behavior behind a fixed model
identifier without notice, so a rerun today is expected to be close to,
but not necessarily numerically identical to, the frozen results this
repository ships with. `reproduce_paper.py` is the only path that
guarantees exact reproduction, because it never makes a network call.

## Tests

```bash
pytest
```

Covers `worldgen.py`'s exact-recombination guarantee, `gates.py` and
`aggregate.py`'s statistics (including reproducing the paper's own
Section 12.3 simulation table as a correctness check), `embed.py`, and
`render.py`'s deterministic 2x2 rendering.

## License

This repository uses two licenses for two different kinds of material:

- **Code** (everything under `src/`, `tests/`, `reproduce_paper.py`, and
  other `*.py` files in this repository) is licensed under the **MIT
  License** -- see [`LICENSE-CODE`](LICENSE-CODE).
- **Data, prompts, frozen model outputs, and other experimental
  material** -- the elicitation prompts embedded in `src/elicit.py`'s
  `report_system_prompt`, `results/raw/*.jsonl` (frozen raw model
  outputs), `results/processed/*.json` (frozen analysis outputs), the
  synthetic world-generation data and calibration/evaluation world-ID
  splits, and the experimental notes/logs (`EXPERIMENT.md`,
  `EXPERIMENT_NOTES.md`, `RESULTS.md`, `RESULTS_2X2.md`) -- is licensed
  under **Creative Commons Attribution 4.0 International (CC BY 4.0)**
  -- see [`LICENSE-DATA`](LICENSE-DATA).

**The compiled paper** (`paper/latex-src/main.pdf`) is posted as an
arXiv preprint under **CC BY 4.0**, consistent with the data license
above. `paper/latex-src/arxiv.sty` is a third-party preprint style
(not written for this paper), from
[kourgeorge/arxiv-style](https://github.com/kourgeorge/arxiv-style),
used here under its own **MIT License** and not relicensed by this
repository.
