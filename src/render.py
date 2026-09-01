"""Deterministic, post-hoc rationale rendering for the 2x2 design.

Not an elicitation prompt variant -- a pure function from (style, seed) to
text, applied AFTER the single frozen elicitation has already produced
estimate + raw_rationale. It never reads the model's estimate or raw
rationale, so it cannot move the estimate: representation is manipulated
with Theta, E, the extraction process, and the estimate all held fixed.

The pilot's first attempt (prompt-level S1/S2 phrasing at elicitation time)
showed this matters: RMSE differed 0.80x between styles, meaning the
"representation" manipulation was leaking into extraction quality. This
module replaces that approach entirely.

Every document in this design has the same abstract structure (segment
stated directly, segment as % of it, segment as growth over a prior
figure), so a style's rendering can be a fixed template pool with light
seeded lexical variation, independent of any particular report's content.
"""
from __future__ import annotations

import random

S1_TEMPLATES = [
    "I combined the direct segment, the percentage-derived segment, and "
    "the growth-adjusted prior-quarter segment.",
    "The total draws on the segment stated directly, the segment derived "
    "as a percentage of it, and the segment adjusted for quarter-over-"
    "quarter growth.",
    "I added the directly reported segment to the percentage-based "
    "segment and the growth-adjusted segment from the prior period.",
    "This figure sums the segment given outright, the segment computed "
    "as a share of it, and the segment carried forward with growth "
    "applied.",
]

S2_TEMPLATES = [
    "The estimate reconciles booked revenue, proportional contribution, "
    "and the growth-adjusted comparative-period amount.",
    "This figure aggregates the baseline reported amount, its "
    "proportional allocation, and the period-over-period adjusted "
    "comparative figure.",
    "The reconciliation combines the booked baseline, its proportional "
    "derivative, and the comparative-period growth adjustment.",
    "This valuation consolidates the disclosed baseline, its "
    "proportional share, and the trailing-period growth-adjusted "
    "reconciliation amount.",
]

_TEMPLATES = {"S1": S1_TEMPLATES, "S2": S2_TEMPLATES}


def render_rationale(style: str, seed: int) -> str:
    rng = random.Random(f"render|{style}|{seed}")
    return rng.choice(_TEMPLATES[style])
