# Reproducibility audit

## What was already reproducible before this pass

The analysis code itself was already fully deterministic and already
separated pure statistics (`gates.py`, `aggregate.py`, `embed.py`) from
data collection (`pilot.py`, `calibration.py`, `grid_ab.py`,
`two_by_two.py`). Every analysis script (`analyze.py`, `diagnostics.py`,
`gamma_analysis.py`, `two_by_two_analysis.py`, `two_by_two_stats.py`,
`figures.py`, `two_by_two_figures.py`) already read only from
`results/raw/*.jsonl` and `results/processed/*.json`, never from a live
API call. `config.yaml` already held every experiment knob (DGP
parameters, model name, seeds, world-id ranges/splits, cost caps,
threshold grids) in one place, and `EXPERIMENT_NOTES.md` / `RESULTS.md` /
`RESULTS_2X2.md` already documented, in frozen prose, what was
confirmatory versus exploratory. The unit test suite already covered the
pure logic. What was missing was not correctness or determinism -- it was
that the frozen raw evidence was gitignored, there was no single entry
point that exercised the whole pipeline end to end, and there was no
automated check that the pipeline actually reproduces the numbers the
paper reports.

## What was changed

1. **`.gitignore`**: removed the bare `*.jsonl` rule (which was silently
   excluding `results/raw/*.jsonl`, the frozen model outputs, from
   version control) and replaced it with an explicit, documented
   exclusion of only `results/cache/` (the internal sha256 response
   cache, a during-collection performance optimization with no
   reproducibility value beyond what `results/raw/` already holds).
2. **`results/raw/*.jsonl` (6 files, ~47MB) added to git**: the frozen,
   already-collected model outputs for the pilot, calibration, Grid A/B,
   the 2x2 pilot, the 2x2 balanced-threshold calibration, and the 2x2
   evaluation. These are the primary evidentiary artifact.
3. **`reproduce_paper.py` (new)**: single entry point. Imports and calls
   the existing analysis modules directly, in the same dependency order
   the original build used, reconstructing the one step that originally
   required a live call (the calibration fit) by deserializing
   `ElicitResult` records straight from `calibration.jsonl` and calling
   `calibration.fit_calibration()` -- the exact same pure function the
   original run used, with no statistics reimplemented anywhere. Ends
   with an integrity check against 7 values, compared at full stored
   precision (not the paper's rounded display values) with a stated
   1e-6 tolerance, and exits non-zero on any mismatch. Also syncs the
   regenerated figures into `paper/latex-src/figures/`, the copies the
   compiled paper actually embeds (a real gap the clean-machine test
   surfaced -- see below).
4. **`requirements.txt` (new)**: only the packages actually imported
   anywhere in `src/` or `tests/`, pinned to the exact versions the
   frozen results were produced and re-verified with.
5. **`README.md` (new)**: repository purpose, structure, install, the
   frozen-data reproduction path, and the optional credentialed rerun
   path with the replication-vs-reproduction distinction stated
   explicitly.
6. **`REPRODUCIBILITY.md` (new)**: confirmatory vs. exploratory analyses,
   calibration/evaluation separation, and what reproduces exactly vs.
   requires live calls -- wording carried over from, not reinterpreted
   from, the frozen methodological record.
7. **`REPRODUCIBILITY_TEXT.md` (new)**: draft paragraph for later
   insertion into `main.tex`. Not inserted; `main.tex` was not touched.

## Files added or modified

| File | Status |
|---|---|
| `.gitignore` | modified |
| `README.md` | added |
| `REPRODUCIBILITY.md` | added |
| `REPRODUCIBILITY_TEXT.md` | added |
| `REPRODUCIBILITY_AUDIT.md` | added (this file) |
| `requirements.txt` | added |
| `reproduce_paper.py` | added |
| `results/raw/calibration.jsonl` | added to git (previously gitignored) |
| `results/raw/grid_ab.jsonl` | added to git (previously gitignored) |
| `results/raw/pilot.jsonl` | added to git (previously gitignored) |
| `results/raw/two_by_two_dedup_calibration.jsonl` | added to git (previously gitignored) |
| `results/raw/two_by_two_eval.jsonl` | added to git (previously gitignored) |
| `results/raw/two_by_two_pilot.jsonl` | added to git (previously gitignored) |

No existing analysis code, configuration, theorem, numerical result,
table, or figure content was changed. `main.tex` was not touched. No new
experiments were run; all six raw JSONL files are exactly the outputs
already used to produce the paper's committed `results/processed/*.json`
and figures.

## Clean-machine test: PASSED

Performed twice (once before, once after the figure-sync fix below), each
time from a fresh `git clone` into an isolated temporary directory with
no `.env` and no `ANTHROPIC_API_KEY` in the environment:

1. `python -m venv .venv_clean && pip install -r requirements.txt` --
   succeeded, exact pinned versions installed.
2. `python reproduce_paper.py` -- exit code 0, no network access or API
   key used anywhere in the run.
3. Integrity check: **7/7 checks passed** (Grid A coverage at n=1 and
   n=32, `gamma_cal`, the representation and ancestry factorial-contrast
   effects, the confirmatory `Delta_BC`, and the exact oracle sweep's
   `min_t max(FMR_B, FSR_C)`), all reproduced values bit-identical to the
   frozen `results/processed/*.json` values at full stored precision.
4. `pytest` -- 53/53 tests passed in the same clean environment.
5. All 9 empirical figures (fig5-fig13) regenerated in `figures/` and
   synced into `paper/latex-src/figures/`.

**One real issue was found and fixed during this test**, not before it:
the first version of `reproduce_paper.py` regenerated `figures/*.pdf` but
not `paper/latex-src/figures/*.pdf`, the copies the compiled paper
actually embeds -- a reviewer could have run the script, seen the
integrity check pass, and still been looking at stale figures if they
opened the PDF. Fixed by adding an explicit sync step (a file copy, not a
byte comparison, since matplotlib embeds a fresh creation-date in every
PDF render, so even a fully correct regeneration differs from the
committed copy at the byte level without being wrong in content). Re-ran
the full clean-machine test after the fix; it still passes.

## Known, harmless non-determinism

Regenerated PDF figures are byte-identical in size but not in content
hash to the committed versions: `git diff --stat` shows them as modified
with 0 insertions/0 deletions, consistent with only an embedded
`CreationDate`/`ModDate` metadata field changing. This does not affect any
reported number (all 7 integrity-check values are exact) and was not
"fixed" by pinning a static date in `savefig`, since that would touch
plotting code across four scripts for a cosmetic property with no bearing
on reproducibility of the results themselves; it is recorded here instead
so a future contributor does not mistake the `git diff` after running
`reproduce_paper.py` for a real change.

## Remaining issues that would block making the repository public

None identified. Specifically checked and clear:

- No API keys, `.env`, or credentials in any tracked file (`.env` is
  gitignored; only the placeholder `.env.example` is tracked).
- No PII: `results/raw/*.jsonl` contains only synthetic fictional
  company names, fabricated financial figures, and model-generated
  rationale text; spot-checked and grepped for credential-shaped strings
  (`sk-ant`, `ANTHROPIC_API_KEY`, `Bearer `) with no matches.
- No absolute local filesystem paths or private directory references in
  any tracked file (grepped for `C:\Users`, `/Users/`, `/home/`).
- No irrelevant temporary or huge cache files staged; `results/cache/`
  (~23,000 small files, a pure performance optimization) is deliberately
  excluded and the `.gitignore` comment explains why rather than leaving
  a bare, unexplained rule.
- `.gitignore` is otherwise sensible: `.venv/`, `__pycache__/`,
  `.pytest_cache/`, `*.parquet`, `.DS_Store`.

One item worth the user's explicit attention rather than a silent
decision on my part: the six raw JSONL files add **~47MB** to the git
history (`grid_ab.jsonl` alone is ~41MB). This is well within GitHub's
100MB single-file limit and is not large in absolute terms, but it is a
one-way door once pushed to a public remote -- if a different distribution
mechanism is preferred for the largest file (e.g. a release asset or an
archival DOI repository such as Zenodo/OSF instead of the git history
itself), that decision should be made before the first public push, not
after.
