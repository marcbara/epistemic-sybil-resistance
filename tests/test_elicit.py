import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import elicit  # noqa: E402


@pytest.mark.parametrize(
    "text,expected_estimate",
    [
        ('{"estimate": 482.1, "rationale": "combined segments"}', 482.1),
        ('```json\n{"estimate": 500, "rationale": "x"}\n```', 500.0),
        ('Sure, here it is: {"estimate": 490.5, "rationale": "y"} hope that helps', 490.5),
        ("not json at all", None),
        ('{"estimate": "about 500", "rationale": "z"}', None),
        ('{"rationale": "missing estimate key"}', None),
    ],
)
def test_parse_json_response(text, expected_estimate):
    estimate, _ = elicit._parse_json_response(text)
    assert estimate == expected_estimate


def test_truncate_rationale_leaves_short_text_untouched():
    text = "combined the three stated segments"
    out, truncated = elicit._truncate_rationale(text)
    assert out == text
    assert truncated is False


def test_truncate_rationale_caps_long_text():
    words = [f"w{i}" for i in range(60)]
    out, truncated = elicit._truncate_rationale(" ".join(words))
    assert truncated is True
    assert len(out.split()) == elicit.MAX_RATIONALE_WORDS
    assert out == " ".join(words[: elicit.MAX_RATIONALE_WORDS])


def test_truncate_rationale_handles_none():
    out, truncated = elicit._truncate_rationale(None)
    assert out is None
    assert truncated is False


def test_cache_key_deterministic_and_sensitive_to_inputs():
    k1 = elicit.cache_key_for("model-a", 0.7, "prompt text", 1)
    k2 = elicit.cache_key_for("model-a", 0.7, "prompt text", 1)
    assert k1 == k2
    assert k1 != elicit.cache_key_for("model-b", 0.7, "prompt text", 1)
    assert k1 != elicit.cache_key_for("model-a", 0.0, "prompt text", 1)
    assert k1 != elicit.cache_key_for("model-a", 0.7, "different prompt", 1)
    assert k1 != elicit.cache_key_for("model-a", 0.7, "prompt text", 2)


@pytest.mark.asyncio
async def test_cost_tracker_halts_at_cap():
    tracker = elicit.CostTracker(hard_cap_usd=0.01, price_in=0.000001, price_out=0.000005)
    # first charge stays under cap
    await tracker.charge(input_tokens=1000, output_tokens=500)  # cost = 0.001 + 0.0025 = 0.0035
    assert tracker.spent_usd == pytest.approx(0.0035)
    # second charge would push spent to 0.007, still under 0.01
    await tracker.charge(input_tokens=1000, output_tokens=500)
    assert tracker.spent_usd == pytest.approx(0.007)
    # third charge would exceed the cap -> must raise and not mutate spent_usd
    with pytest.raises(elicit.BudgetExceededError):
        await tracker.charge(input_tokens=1000, output_tokens=500)
    assert tracker.spent_usd == pytest.approx(0.007)
