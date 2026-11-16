"""Chunking: no database, no model."""

import pytest

from frontdesk_agent.retrieval.chunking import (
    ChunkingConfig,
    chunk_document,
    parse_markdown,
)

DOC = """# Cancellation Policy

## Notice required

We ask for at least 24 hours' notice.

## Late cancellation fee

A $50 fee applies.
"""


def test_parse_takes_title_and_sections() -> None:
    doc = parse_markdown(DOC, fallback_title="x")
    assert doc.title == "Cancellation Policy"
    assert [h for h, _ in doc.sections] == ["Notice required", "Late cancellation fee"]


def test_fallback_title_when_no_h1() -> None:
    assert parse_markdown("## A\n\nbody", fallback_title="Fallback").title == "Fallback"


def test_section_chunks_never_mix_topics_and_carry_headings() -> None:
    chunks = chunk_document(parse_markdown(DOC, "x"), ChunkingConfig("section", 50, 5))
    assert [c.heading for c in chunks] == [
        "Cancellation Policy > Notice required",
        "Cancellation Policy > Late cancellation fee",
    ]
    assert "$50" not in chunks[0].content
    assert chunks[1].embedding_text().startswith("Cancellation Policy > Late cancellation fee")


def test_long_sections_are_windowed_with_overlap() -> None:
    body = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_document(
        parse_markdown(f"## S\n\n{body}", "t"), ChunkingConfig("section", 10, 3)
    )
    words = [c.content.split() for c in chunks]
    assert all(len(w) <= 10 for w in words)
    assert words[0][-3:] == words[1][:3]  # the overlap
    assert words[-1][-1] == "w24"  # nothing dropped at the end
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_fixed_chunks_ignore_headings() -> None:
    chunks = chunk_document(parse_markdown(DOC, "x"), ChunkingConfig("fixed", 100, 10))
    assert len(chunks) == 1
    assert "24 hours" in chunks[0].content and "$50" in chunks[0].content


@pytest.mark.parametrize(
    ("strategy", "size", "overlap"), [("semantic", 10, 1), ("section", 10, 10), ("fixed", 5, -1)]
)
def test_bad_configs_are_rejected(strategy: str, size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(strategy, size, overlap)
