"""Rough cost of a model call, recorded with every call in llm_calls.

Base rates in USD per million tokens (input, output), from
https://platform.claude.com/docs/en/about-claude/pricing as of 2026-11-23.
Prompt-cache tokens are priced from the input rate: a 5-minute write costs
1.25x, a read 0.1x (0.05x on Sonnet 5.5). Batch discounts are not modelled
(nothing here uses the Batch API). Unknown models cost None rather than a
guess, so a dashboard shows a gap instead of a wrong number.
"""

from decimal import Decimal
from typing import Protocol

# (prefix, input, output, cache-read multiplier). Longest prefix first:
# "claude-sonnet-5" must not swallow "claude-sonnet-5-5".
_PER_MILLION: list[tuple[str, Decimal, Decimal, Decimal]] = [
    ("claude-haiku-5-5", Decimal("0.10"), Decimal("0.50"), Decimal("0.1")),  # prompts <= 100K
    ("claude-haiku-4-5", Decimal("1"), Decimal("5"), Decimal("0.1")),
    ("claude-sonnet-5-5", Decimal("2"), Decimal("10"), Decimal("0.05")),
    ("claude-sonnet-5", Decimal("2"), Decimal("10"), Decimal("0.1")),
]

# Haiku 5.5 charges more once a prompt passes 100K input tokens, counting
# cache reads and writes in the prompt length.
_HAIKU_55_LONG = (Decimal("0.50"), Decimal("2.50"))
_CACHE_WRITE = Decimal("1.25")  # 5-minute TTL, the only one this app uses


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Decimal | None:
    for prefix, rate_in, rate_out, read_mult in _PER_MILLION:
        if model.startswith(prefix):
            prompt = input_tokens + cache_read_tokens + cache_write_tokens
            if prefix == "claude-haiku-5-5" and prompt > 100_000:
                rate_in, rate_out = _HAIKU_55_LONG
            total = (
                rate_in * input_tokens
                + rate_in * _CACHE_WRITE * cache_write_tokens
                + rate_in * read_mult * cache_read_tokens
                + rate_out * output_tokens
            )
            return (total / Decimal(1_000_000)).quantize(Decimal("0.000001"))
    return None


class Usage(Protocol):
    @property
    def model(self) -> str: ...
    @property
    def input_tokens(self) -> int: ...
    @property
    def output_tokens(self) -> int: ...
    @property
    def cache_read_tokens(self) -> int: ...
    @property
    def cache_write_tokens(self) -> int: ...


def cost_of(call: Usage) -> Decimal | None:
    """The cost of one completion or agent turn, cache tokens included."""
    return cost_usd(
        call.model,
        call.input_tokens,
        call.output_tokens,
        cache_read_tokens=call.cache_read_tokens,
        cache_write_tokens=call.cache_write_tokens,
    )
