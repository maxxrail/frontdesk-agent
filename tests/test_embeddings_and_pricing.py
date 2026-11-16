"""The embedder and the cost table. Offline: WordLlama ships its weights."""

import math
from decimal import Decimal

from frontdesk_agent.llm.pricing import cost_usd
from frontdesk_agent.retrieval.embeddings import EMBEDDING_DIM, default_embedder


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_vectors_are_unit_length_deterministic_and_sized() -> None:
    e = default_embedder()
    a, b = e.embed(["late cancellation fee", "late cancellation fee"])
    assert len(a) == EMBEDDING_DIM == e.dim
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-4)
    assert a == b
    assert e.embed([]) == []


def test_empty_text_embeds_to_zero_not_nan() -> None:
    [v] = default_embedder().embed([""])
    assert not any(v) and not any(math.isnan(x) for x in v)


def test_meaning_beats_unrelated_text() -> None:
    q, fee, insurance = default_embedder().embed(
        [
            "is there a charge for cancelling last minute",
            "Cancellations with less than 24 hours notice are charged a fee",
            "We direct-bill most insurance providers",
        ]
    )
    assert _cos(q, fee) > _cos(q, insurance)


def test_costs() -> None:
    assert cost_usd("claude-haiku-4-5-20251001", 1_000_000, 0) == Decimal("1.000000")
    assert cost_usd("claude-haiku-5-5", 1_000, 1_000) == Decimal("0.000600")
    assert cost_usd("claude-haiku-5-5", 200_000, 0) == Decimal("0.100000")  # long-prompt tier
    assert cost_usd("claude-sonnet-5-5", 0, 1_000_000) == Decimal("10.000000")
    assert cost_usd("some-other-model", 10, 10) is None
