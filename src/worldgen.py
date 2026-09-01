"""World and document generation (EXPERIMENT.md Section 2, 9).

Design note (revised after the Phase 0 pilot -- see EXPERIMENT_NOTES.md):
Section 2 specifies segment B as "a percentage of the (unstated) total", i.e.
E = A + p*E + Q*(1+g), solved as E = (A + Q*(1+g)) / (1-p). The pilot showed
this self-referential form has two costs. First, it is arithmetic-exactness-
hostile: a naive runtime division by (1-p) only terminates as a finite
decimal when (1-p)'s reduced fraction has solely 2 and 5 as prime factors,
so an exact solver needs p restricted to a narrow curated grid. Second, and
decisively, the pilot's raw transcripts show the main agent (Haiku 4.5)
correctly recognizing the algebraic structure but then making basic
arithmetic slips solving the division (e.g. computing 554.4 instead of the
correct 565.12 for (138.6+143.96)/0.5) -- inflating nu_hat^2 to ~61,406
(implied report sd ~247M EUR) versus the intended sigma=50 noise regime.
That is model arithmetic error, not extraction/interpretation noise, and it
defeats gates 3 and 6.

Segment B is therefore redefined as a percentage of stated segment A rather
than of the unstated total: E = A + p*A + Q*(1+g) = A*(1+p) + Q*(1+g). This
still requires combining three cues (locate A, compute B from A and p,
compute C from Q and g, sum) -- genuine multi-cue inference, not lookup --
but needs only multiplication and addition, which (a) is exact for any
finite-decimal p, A, Q, g with zero curation of the p grid, and (b) is
arithmetic Haiku 4.5 can reliably execute, so residual noise should better
reflect the paper's intended extraction/interpretation noise nu^2 rather
than computational mistakes.

Entity names are synthetic (invented syllables + a generic sector suffix) with
a blocklist of real company names as a safety net. This is a heuristic, not a
verified absence of web-plausible collision -- flagged as a known limitation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional
import hashlib
import random

REAL_COMPANY_BLOCKLIST = {
    "siemens", "google", "apple", "microsoft", "amazon", "meta", "tesla",
    "samsung", "sony", "philips", "bosch", "nestle", "unilever", "toyota",
    "shell", "bp", "exxon", "chevron", "ibm", "intel", "oracle", "sap",
    "nokia", "ericsson", "volkswagen", "bmw", "mercedes", "airbus", "boeing",
}

SYLLABLES_1 = [
    "zal", "krin", "ovex", "than", "mir", "sela", "yuno", "brix", "delt",
    "novo", "quen", "tarl", "esk", "vior", "plun", "ambr", "cort", "fien",
    "gorn", "hulm", "izo", "jask", "klov", "lorn", "myx", "narr",
]
SYLLABLES_2 = [
    "dor", "mera", "tix", "onic", "vale", "rus", "ken", "sil", "tona",
    "brek", "lum", "ander", "isk", "eron", "ova", "yth", "acel", "indro",
    "urst", "avic",
]
SECTOR_SUFFIXES = [
    "Materials", "Dynamics", "Holdings", "Systems", "Works", "Robotics",
    "Analytics", "Logistics", "Instruments", "Components", "Ventures",
    "Industries", "Technologies", "Solutions", "Networks", "Fabrication",
]


def _contains_blocklisted_fragment(name: str) -> bool:
    low = name.lower()
    return any(bad in low for bad in REAL_COMPANY_BLOCKLIST)


def generate_entity_name(rng: random.Random, max_attempts: int = 50) -> str:
    for _ in range(max_attempts):
        stem = rng.choice(SYLLABLES_1).capitalize() + rng.choice(SYLLABLES_2)
        suffix = rng.choice(SECTOR_SUFFIXES)
        name = f"{stem} {suffix}"
        if not _contains_blocklisted_fragment(name):
            return name
    raise RuntimeError("failed to generate a non-blocklisted entity name")


def world_seed(master_seed: int, world_id: int) -> int:
    h = hashlib.sha256(f"{master_seed}:{world_id}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def _truncated_normal_positive(rng: random.Random, mean: float, sd: float) -> float:
    while True:
        x = rng.gauss(mean, sd)
        if x > 0:
            return x


def _round_to_grid(x: float, grid: float) -> Fraction:
    """Round a float to the nearest multiple of `grid`, returned as an exact Fraction."""
    steps = round(x / grid)
    # represent grid exactly assuming it has a finite decimal representation
    grid_frac = Fraction(grid).limit_denominator(10_000)
    return steps * grid_frac


def format_money(value: Fraction, decimals: int = 1) -> str:
    """Render an exact Fraction as a fixed-decimal string (never as 'n/d')."""
    scale = Fraction(10) ** decimals
    scaled = value * scale
    if scaled.denominator != 1:
        raise ValueError(f"{value} is not exactly representable at {decimals} decimals")
    return f"{int(scaled) / (10 ** decimals):.{decimals}f}"


@dataclass
class RootDocument:
    world_id: int
    root_id: int
    entity_name: str
    theta: float
    eps: float
    e_true: Fraction          # exact primitive evidence value (M EUR)
    a_value: Fraction         # segment A, stated directly
    p_pct: int                # segment B, % of segment A, stated
    q_value: Fraction         # segment C base, prior-quarter figure, stated
    g_pct: int                # segment C growth rate, %, stated
    distractors: dict = field(default_factory=dict)
    text: str = ""

    def e_true_float(self) -> float:
        return float(self.e_true)


def solve_e_from_cues(a_value: Fraction, p_pct: int, q_value: Fraction, g_pct: int) -> Fraction:
    """Deterministic solver: recombine displayed cues to the exact primitive value.

    E = A*(1+p) + Q*(1+g) = A + (segment B, p% of A) + (segment C, Q grown by g%).
    Pure multiplication and addition of exact Fractions -- no division anywhere,
    so the result is exact by construction for any finite-decimal p, A, Q, g.
    """
    p = Fraction(p_pct, 100)
    g = Fraction(g_pct, 100)
    return a_value * (1 + p) + q_value * (1 + g)


def generate_root_document(
    rng: random.Random,
    world_id: int,
    root_id: int,
    entity_name: str,
    theta: float,
    cfg: dict,
) -> RootDocument:
    dgp = cfg["dgp"]
    sigma = dgp["sigma"]
    eps = rng.gauss(0.0, sigma)
    e_target = theta + eps  # guides plausible cue magnitudes; not used after this point

    p_pct = rng.randint(dgp["p_grid_pct_range"][0], dgp["p_grid_pct_range"][1])
    # Exclude g=0: pilot transcripts showed the model sometimes misreads
    # "grew 0% over Q" as "contributes 0" instead of "equals Q" -- a genuine
    # template ambiguity at that single point, not representative extraction
    # noise, and it dominated the pooled noise estimate. |g| >= 2% sidesteps it.
    g_pct = rng.choice([
        g for g in range(dgp["g_grid_pct_range"][0], dgp["g_grid_pct_range"][1] + 1)
        if abs(g) >= 2
    ])
    grid = dgp["display_grid"]

    # E = A*(1+p) + Q*(1+g). a_share splits E's magnitude between the two terms.
    a_share = rng.uniform(*dgp["a_share_range"])
    a_term_target = max(e_target * a_share, grid)
    c_term_target = max(e_target * (1 - a_share), grid)

    a_raw = a_term_target / (1 + p_pct / 100.0)
    q_raw = c_term_target / (1 + g_pct / 100.0)

    a_value = _round_to_grid(max(a_raw, grid), grid)
    q_value = _round_to_grid(max(q_raw, grid), grid)

    e_true = solve_e_from_cues(a_value, p_pct, q_value, g_pct)

    n_distractors = rng.randint(*dgp["num_distractors"])
    distractor_pool = {
        "headcount": rng.randint(80, 4000),
        "operating_margin_pct": rng.randint(4, 28),
        "employee_growth_pct": rng.randint(-5, 15),
        "office_count": rng.randint(1, 40),
        "rd_spend_m": round(rng.uniform(2.0, 60.0), 1),
    }
    keys = rng.sample(list(distractor_pool.keys()), n_distractors)
    distractors = {k: distractor_pool[k] for k in keys}

    doc = RootDocument(
        world_id=world_id,
        root_id=root_id,
        entity_name=entity_name,
        theta=theta,
        eps=eps,
        e_true=e_true,
        a_value=a_value,
        p_pct=p_pct,
        q_value=q_value,
        g_pct=g_pct,
        distractors=distractors,
    )
    doc.text = render_document(doc, rng)
    return doc


_TEMPLATE_VARIANTS = [
    (
        "Quarterly memo -- {entity}\n\n"
        "Product revenue for the quarter reached {a} million EUR. "
        "Services revenue ran at {p}% of the product line's revenue this quarter. "
        "Licensing revenue grew {g}% over the prior quarter's figure of {q} million EUR.\n"
        "{distractor_lines}\n"
        "Prepared for internal circulation only."
    ),
    (
        "Internal briefing: {entity}, current quarter\n\n"
        "This quarter, the product line generated {a} million EUR in revenue. "
        "Services revenue came in at {p} percent of that product-line figure. "
        "The licensing line grew {g}% relative to last quarter's {q} million EUR base.\n"
        "{distractor_lines}\n"
        "For internal use."
    ),
    (
        "{entity} -- quarterly summary\n\n"
        "Product segment revenue: {a}M EUR, reported directly. "
        "Services segment: {p}% of the product segment's revenue (not separately disclosed in absolute terms). "
        "Licensing segment: up {g}% versus the prior quarter, when it stood at {q}M EUR.\n"
        "{distractor_lines}\n"
        "Distribution restricted to internal readers."
    ),
]

_DISTRACTOR_LABELS = {
    "headcount": "Headcount stood at {v} employees.",
    "operating_margin_pct": "Operating margin was {v}%.",
    "employee_growth_pct": "Headcount growth was {v}% year over year.",
    "office_count": "The company operates {v} offices.",
    "rd_spend_m": "R&D spend was {v} million EUR.",
}

# Non-numeric filler sentences padding documents to the brief's target length
# (150-250 words, Section 2). These carry no cues and must never introduce a
# number, so they cannot be mistaken for extraction signal or a distractor.
_INTRO_FILLERS = [
    "The finance team compiled this note ahead of the internal quarterly review.",
    "Market conditions in the sector remained broadly stable over the period.",
    "Leadership has continued to emphasize disciplined execution against the annual plan.",
    "This summary reflects preliminary figures pending the standard month-end close process.",
    "The business continues to operate across its established set of core markets.",
    "Management commentary below draws on inputs from the regional finance leads.",
    "No material changes to the reporting structure occurred during the quarter.",
    "The following notes are intended to support the upcoming leadership discussion.",
    "Commercial activity followed the seasonal pattern typical for this part of the year.",
    "This memo consolidates inputs gathered from the relevant business unit controllers.",
    "The competitive landscape did not shift materially relative to the prior quarter.",
    "Currency effects were modest and are not separately broken out in this note.",
    "The team notes that macro conditions remained a background consideration throughout the period.",
    "This briefing is intended as a working document ahead of formal reporting.",
    "Internal stakeholders should treat the figures below as directional pending final review.",
    "The quarter proceeded largely in line with internal expectations set at the outset.",
]

_OUTRO_FILLERS = [
    "Further detail is available on request from the finance business partner team.",
    "A fuller breakdown will accompany the formal quarterly reporting package.",
    "Questions on the figures above should be directed to the relevant segment lead.",
    "This note will be superseded by the audited figures once finalized.",
    "The team welcomes feedback ahead of the next planning cycle.",
    "Additional context on segment performance can be found in the supporting slide deck.",
    "This summary does not constitute a public disclosure or investor communication.",
    "Later revisions to these figures, if any, will be circulated separately.",
    "The commentary above reflects the views of the finance team as of this writing.",
    "Recipients should not forward this memo outside the immediate working group.",
    "A follow-up discussion is planned once the month-end close is finalized.",
    "This document supersedes any earlier informal estimates shared for the same period.",
]


def render_document(doc: RootDocument, rng: random.Random, target_words: Optional[int] = None) -> str:
    template = rng.choice(_TEMPLATE_VARIANTS)
    distractor_lines = " ".join(
        _DISTRACTOR_LABELS[k].format(v=v) for k, v in doc.distractors.items()
    )
    body = template.format(
        entity=doc.entity_name,
        a=format_money(doc.a_value),
        p=doc.p_pct,
        g=doc.g_pct,
        q=format_money(doc.q_value),
        distractor_lines=distractor_lines,
    )

    if target_words is None:
        target_words = rng.randint(170, 220)

    header, _, rest = body.partition("\n\n")
    intro_pool = rng.sample(_INTRO_FILLERS, len(_INTRO_FILLERS))
    outro_pool = rng.sample(_OUTRO_FILLERS, len(_OUTRO_FILLERS))

    def word_count(s: str) -> int:
        return len(s.split())

    intro_sentences: list[str] = []
    outro_sentences: list[str] = []
    i = o = 0
    while word_count(f"{header}\n\n{' '.join(intro_sentences)}\n{rest}\n{' '.join(outro_sentences)}") < target_words:
        if i < len(intro_pool) and (i <= o or o >= len(outro_pool)):
            intro_sentences.append(intro_pool[i]); i += 1
        elif o < len(outro_pool):
            outro_sentences.append(outro_pool[o]); o += 1
        else:
            break  # pools exhausted; accept whatever length results

    intro_block = " ".join(intro_sentences)
    outro_block = " ".join(outro_sentences)
    parts = [header, "", intro_block, rest]
    if outro_block:
        parts.append(outro_block)
    return "\n".join(p for p in parts if p != "")


@dataclass
class World:
    world_id: int
    entity_name: str
    theta: float


def generate_world(master_seed: int, world_id: int, cfg: dict) -> World:
    seed = world_seed(master_seed, world_id)
    rng = random.Random(seed)
    dgp = cfg["dgp"]
    theta = _truncated_normal_positive(rng, dgp["prior_mean"], dgp["prior_sd"])
    entity_name = generate_entity_name(rng)
    return World(world_id=world_id, entity_name=entity_name, theta=theta)


def generate_root_for_world(world: World, root_id: int, master_seed: int, cfg: dict) -> RootDocument:
    seed = world_seed(master_seed, world.world_id) ^ (root_id * 0x9E3779B97F4A7C15)
    rng = random.Random(seed & ((1 << 63) - 1))
    return generate_root_document(rng, world.world_id, root_id, world.entity_name, world.theta, cfg)
