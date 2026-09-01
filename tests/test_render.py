import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import render  # noqa: E402


def test_render_is_deterministic():
    a = render.render_rationale("S1", "world1-root0")
    b = render.render_rationale("S1", "world1-root0")
    assert a == b


def test_render_styles_draw_from_disjoint_pools():
    s1_texts = {render.render_rationale("S1", i) for i in range(50)}
    s2_texts = {render.render_rationale("S2", i) for i in range(50)}
    assert s1_texts.issubset(set(render.S1_TEMPLATES))
    assert s2_texts.issubset(set(render.S2_TEMPLATES))
    assert s1_texts.isdisjoint(s2_texts)


def test_render_varies_across_seeds():
    texts = {render.render_rationale("S1", i) for i in range(20)}
    assert len(texts) > 1  # not degenerate to a single template
