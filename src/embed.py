"""Lightweight, dependency-free text embedding for the report-space dedup
baseline (EXPERIMENT.md Section 5.2: "embed the rationale only").

Deviation note: rather than an ML embedding model (e.g. sentence-transformers,
which pulls in torch and a model download), this uses hashed bag-of-words
term-frequency vectors with cosine similarity. This is a deliberate scope
choice, not a claim it matches a semantic embedding's quality -- our
rationales are short (<=30 words, frozen at pilot) and fairly formulaic given
the structured elicitation prompt, so lexical overlap is a reasonable proxy
for "these reports describe the same extraction." Documented here so it is
easy to swap in a real embedding model later without touching aggregate.py's
call sites (embed_texts is the only entry point).
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

VECTOR_DIM = 512

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _hash_index(token: str, dim: int) -> int:
    h = hashlib.sha256(token.encode()).digest()
    return int.from_bytes(h[:4], "big") % dim


def embed_text(text: str, dim: int = VECTOR_DIM) -> list[float]:
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        vec[_hash_index(tok, dim)] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def embed_texts(texts: Sequence[str], dim: int = VECTOR_DIM) -> list[list[float]]:
    return [embed_text(t, dim) for t in texts]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # both already L2-normalized
