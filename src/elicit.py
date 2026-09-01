"""LLM elicitation: async batching, caching, retries, strict JSON validation.

Implements EXPERIMENT.md Section 9: response cache keyed by
sha256(model, temperature, prompt, seed), retries with backoff, one repair
retry on invalid JSON, and a hard cost cap the run halts at gracefully.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

load_dotenv()

try:
    from anthropic import AsyncAnthropic, APIError, APIStatusError, APIConnectionError, RateLimitError, InternalServerError, BadRequestError
except ImportError:  # pragma: no cover
    AsyncAnthropic = None
    APIError = Exception
    APIStatusError = Exception
    APIConnectionError = Exception
    RateLimitError = Exception
    InternalServerError = Exception
    BadRequestError = Exception

# Models that reject an explicit `temperature` (some Claude 5-family models
# use fixed/managed sampling). Learned at runtime from a 400 response and
# cached here so we stop resending the invalid parameter after the first hit.
_NO_TEMPERATURE_MODELS: set[str] = set()


MAX_RATIONALE_WORDS = 30

# The frozen A/B prompt's rationale clause -- kept byte-identical as style S1,
# so A/B's reports remain exactly reproducible from cache.
_RATIONALE_CLAUSE_S1 = (
    '"<at most 25 words, a plain description of which segments you used, with '
    'no arithmetic and no numbers>"'
)
# Style S2 (2x2 design only, prediction 13.4): same task, same information
# content requested, deliberately different vocabulary/register -- an
# accounting-memo phrasing instead of a plain-English one. This is a
# controlled representation stress test (see EXPERIMENT_NOTES.md's 2x2
# section): a surface-form change with no bearing on the evidence itself.
# If it changes an aggregator's independence decision, similarity does not
# identify ancestry -- that IS the point being tested, not a confound to
# hide.
_RATIONALE_CLAUSE_S2 = (
    '"<at most 25 words, written as a terse internal accounting note using '
    'financial-reporting terminology (e.g. \'segment allocation\', '
    '\'period-over-period adjustment\', \'baseline recomputation\'), with no '
    'arithmetic and no numbers>"'
)
_RATIONALE_CLAUSES = {"S1": _RATIONALE_CLAUSE_S1, "S2": _RATIONALE_CLAUSE_S2}


def report_system_prompt(rationale_style: str = "S1") -> str:
    """rationale_style selects only the rationale-phrasing clause; the
    document, the arithmetic instructions, and the estimate-format
    requirement are byte-identical across styles by construction, so any
    difference in the numeric estimate between styles is a genuine
    extraction effect to gate-check, not a prompt confound."""
    clause = _RATIONALE_CLAUSES[rationale_style]
    return (
        "You are analyzing an internal company memo to estimate the company's total "
        "quarterly revenue in millions of EUR. The memo describes exactly three "
        "revenue segments: one stated directly, one stated as a percentage of the "
        "first, and one stated as growth over a prior-quarter figure. Compute each "
        "segment and add them: that sum is your final answer. Do not second-guess, "
        "adjust, or reinterpret the sum once you have computed it -- there is no "
        "hidden complication and no additional segment. "
        "Work out the arithmetic silently; do not show your work, and do not write "
        "out a step-by-step derivation anywhere in your reply. "
        "Respond with a single JSON object and nothing else, of the form: "
        '{"estimate": <number>, "rationale": ' + clause + "}. "
        "The estimate must be a plain number in millions of EUR (e.g. 482.1), not a "
        "range or a string."
    )


REPORT_SYSTEM_PROMPT = report_system_prompt("S1")  # frozen A/B prompt, unchanged

LEAKAGE_SYSTEM_PROMPT = (
    "You are asked to estimate a company's total quarterly revenue in millions "
    "of EUR, given only its name. You have no document and no other information. "
    "Respond with a single JSON object and nothing else, of the form: "
    '{"estimate": <number>, "rationale": "<one sentence>"}.'
)

REPAIR_SUFFIX = (
    "\n\nYour previous reply could not be parsed as JSON. Reply again with "
    'ONLY a JSON object of the form {"estimate": <number>, "rationale": "<one '
    'sentence>"} and nothing else -- no markdown, no code fences, no extra text.'
)


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class ElicitResult:
    cache_key: str
    world_id: int
    root_id: Optional[int]
    call_kind: str  # "report" | "leakage"
    model: str
    temperature: float
    prompt: str
    raw_text: Optional[str]
    parsed_estimate: Optional[float]
    parsed_rationale: Optional[str]
    valid: bool
    repaired: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str
    error: Optional[str] = None
    rationale_truncated: bool = False


def cache_key_for(model: str, temperature: float, prompt: str, seed: int) -> str:
    payload = f"{model}|{temperature}|{seed}|{prompt}"
    return hashlib.sha256(payload.encode()).hexdigest()


class CostTracker:
    def __init__(self, hard_cap_usd: float, price_in: float, price_out: float):
        self.hard_cap_usd = hard_cap_usd
        self.price_in = price_in
        self.price_out = price_out
        self.spent_usd = 0.0
        self._lock = asyncio.Lock()

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.price_in + output_tokens * self.price_out

    async def charge(self, input_tokens: int, output_tokens: int) -> float:
        cost = self.estimate_cost(input_tokens, output_tokens)
        async with self._lock:
            if self.spent_usd + cost > self.hard_cap_usd:
                raise BudgetExceededError(
                    f"cost cap reached: spent={self.spent_usd:.4f} + "
                    f"next={cost:.4f} > cap={self.hard_cap_usd}"
                )
            self.spent_usd += cost
        return cost

    async def would_exceed(self) -> bool:
        async with self._lock:
            return self.spent_usd >= self.hard_cap_usd


def _parse_json_response(text: str) -> tuple[Optional[float], Optional[str]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None, None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None, None
    if not isinstance(obj, dict) or "estimate" not in obj:
        return None, None
    try:
        estimate = float(obj["estimate"])
    except (TypeError, ValueError):
        return None, None
    rationale = obj.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        rationale = str(rationale)
    return estimate, rationale


def _truncate_rationale(rationale: Optional[str]) -> tuple[Optional[str], bool]:
    """Hard cap on rationale length (Section 5.2's dedup baseline embeds the
    rationale; a runaway step-by-step derivation would confound it, inflate
    cost, and add variability the confirmatory grids don't need). Deterministic
    word-level truncation, not a repair call -- the estimate is unaffected."""
    if rationale is None:
        return None, False
    words = rationale.split()
    if len(words) <= MAX_RATIONALE_WORDS:
        return rationale, False
    return " ".join(words[:MAX_RATIONALE_WORDS]), True


class Elicitor:
    def __init__(
        self,
        cfg: dict,
        cache_dir: Path,
        cost_tracker: CostTracker,
        max_concurrency: int = 8,
    ):
        if AsyncAnthropic is None:
            raise RuntimeError("anthropic package not installed in this environment")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to a .env file in the project root."
            )
        self.client = AsyncAnthropic(api_key=api_key)
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cost_tracker = cost_tracker
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> Optional[dict]:
        p = self._cache_path(key)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def _save_cache(self, key: str, record: dict) -> None:
        self._cache_path(key).write_text(json.dumps(record, indent=2))

    @retry(
        retry=retry_if_exception_type((APIConnectionError, RateLimitError, InternalServerError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    async def _call_api(self, system: str, prompt: str, model: str, temperature: float):
        kwargs = dict(
            model=model,
            max_tokens=self.cfg["elicitation"]["max_tokens"],
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if model not in _NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = temperature
        try:
            return await self.client.messages.create(**kwargs)
        except BadRequestError as e:
            if "temperature" in str(e).lower() and "deprecated" in str(e).lower():
                _NO_TEMPERATURE_MODELS.add(model)
                kwargs.pop("temperature", None)
                return await self.client.messages.create(**kwargs)
            raise

    async def elicit(
        self,
        *,
        world_id: int,
        root_id: Optional[int],
        call_kind: str,
        document_text: Optional[str],
        model: str,
        temperature: float,
        seed: int,
        jsonl_path: Optional[Path] = None,
        rationale_style: str = "S1",
    ) -> ElicitResult:
        """jsonl_path: if given, appends the raw record here -- but only for a
        fresh API call, never for a cache hit, so re-running an already-
        collected batch (e.g. a small test batch later folded into a bigger
        run) can't duplicate a call's log entry under its own cache_key.

        rationale_style: "S1" (default, byte-identical to the frozen A/B
        prompt) or "S2" (2x2 design only -- same task, different rationale
        phrasing register). Only affects call_kind="report"."""
        system = report_system_prompt(rationale_style) if call_kind == "report" else LEAKAGE_SYSTEM_PROMPT
        if call_kind == "report":
            prompt = f"Memo:\n\n{document_text}"
        else:
            prompt = f"Company name: {document_text}"

        key = cache_key_for(model, temperature, f"{system}\n{prompt}", seed)
        cached = self._load_cache(key)
        if cached is not None:
            return ElicitResult(**cached)

        if await self.cost_tracker.would_exceed():
            raise BudgetExceededError("cost cap already reached")

        async with self.semaphore:
            resp = await self._call_api(system, prompt, model, temperature)
        raw_text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        repaired = False

        estimate, rationale = _parse_json_response(raw_text)
        if estimate is None:
            repaired = True
            async with self.semaphore:
                resp2 = await self._call_api(
                    system, prompt + REPAIR_SUFFIX, model, temperature
                )
            raw_text2 = "".join(
                block.text for block in resp2.content if getattr(block, "type", None) == "text"
            )
            in_tok += resp2.usage.input_tokens
            out_tok += resp2.usage.output_tokens
            estimate, rationale = _parse_json_response(raw_text2)
            raw_text = raw_text2 if estimate is not None else raw_text + "\n---REPAIR---\n" + raw_text2

        rationale, rationale_truncated = _truncate_rationale(rationale)
        cost = await self.cost_tracker.charge(in_tok, out_tok)
        valid = estimate is not None

        result = ElicitResult(
            cache_key=key,
            world_id=world_id,
            root_id=root_id,
            call_kind=call_kind,
            model=model,
            temperature=temperature,
            prompt=prompt,
            raw_text=raw_text,
            parsed_estimate=estimate,
            parsed_rationale=rationale,
            valid=valid,
            repaired=repaired,
            rationale_truncated=rationale_truncated,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            timestamp=datetime.now(timezone.utc).isoformat(),
            error=None if valid else "unparseable_json_after_repair",
        )
        self._save_cache(key, asdict(result))
        if jsonl_path is not None:
            append_jsonl(jsonl_path, asdict(result))
        return result


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
