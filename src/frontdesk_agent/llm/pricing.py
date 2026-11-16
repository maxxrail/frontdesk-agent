"""Rough cost of a model call, recorded with every call in llm_calls.

Base rates in USD per million tokens (input, output), from
https://platform.claude.com/docs/en/about-claude/pricing as of 2026-11-14.
Caching and batch discounts are not modelled. Unknown models cost None rather
than a guess, so a dashboard shows a gap instead of a wrong number.
"""

from decimal import Decimal

# Longest prefix first: "claude-sonnet-5" must not swallow "claude-sonnet-5-5".
_PER_MILLION: list[tuple[str, Decimal, Decimal]] = [
    ("claude-haiku-5-5", Decimal("0.10"), Decimal("0.50")),  # prompts <= 100K tokens
    ("claude-haiku-4-5", Decimal("1"), Decimal("5")),
    ("claude-sonnet-5-5", Decimal("2"), Decimal("10")),
    ("claude-sonnet-5", Decimal("2"), Decimal("10")),
]

# Haiku 5.5 charges more once a prompt passes 100K input tokens.
_HAIKU_55_LONG = (Decimal("0.50"), Decimal("2.50"))


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> Decimal | None:
    for prefix, rate_in, rate_out in _PER_MILLION:
        if model.startswith(prefix):
            if prefix == "claude-haiku-5-5" and input_tokens > 100_000:
                rate_in, rate_out = _HAIKU_55_LONG
            total = rate_in * input_tokens + rate_out * output_tokens
            return (total / Decimal(1_000_000)).quantize(Decimal("0.000001"))
    return None
