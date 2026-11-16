"""Splitting policy documents into retrievable chunks.

Two strategies, so the choice between them is made from measurements
(``evals/retrieval.py``) rather than taste:

- ``section``: split at ``##`` headings, then window long sections by words
  with overlap. A chunk never mixes two topics.
- ``fixed``: word windows over the whole document, ignoring headings. The
  baseline most tutorials use.

Chunk text is built from the document's own words joined by single spaces, so
a phrase from the document can be found in a chunk after whitespace
normalisation. The golden set relies on that.
"""

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    # Defaults chosen from evals/reports/retrieval-20261114.md (ADR 0008).
    strategy: str = "section"  # "section" or "fixed"
    max_words: int = 160
    overlap_words: int = 30

    def __post_init__(self) -> None:
        if self.strategy not in {"section", "fixed"}:
            raise ValueError(f"unknown chunking strategy: {self.strategy}")
        if not 0 <= self.overlap_words < self.max_words:
            raise ValueError("overlap_words must be >= 0 and smaller than max_words")

    @property
    def label(self) -> str:
        return f"{self.strategy}:{self.max_words}/{self.overlap_words}"


@dataclass(frozen=True, slots=True)
class Chunk:
    ordinal: int
    heading: str  # "Document title > Section"; empty section for fixed chunks
    content: str

    def embedding_text(self) -> str:
        # The heading travels with the text: "Late cancellation fee" is often
        # the only place the topic is named.
        return f"{self.heading}\n{self.content}" if self.heading else self.content


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    title: str
    sections: list[tuple[str, str]]  # (heading, body)


def parse_markdown(markdown: str, fallback_title: str) -> ParsedDocument:
    title = fallback_title
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append((heading, text))

    for line in markdown.splitlines():
        match = _HEADING.match(line.strip())
        if match and len(match.group(1)) == 1 and title == fallback_title and not sections:
            title = match.group(2).strip()
            continue
        if match:
            flush()
            heading, body = match.group(2).strip(), []
        else:
            body.append(line)
    flush()
    return ParsedDocument(title=title, sections=sections)


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(words[start : start + size])
        if start + size >= len(words):
            break
    return out


def chunk_document(doc: ParsedDocument, config: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    if config.strategy == "section":
        for heading, body in doc.sections:
            label = f"{doc.title} > {heading}" if heading else doc.title
            for window in _windows(body.split(), config.max_words, config.overlap_words):
                chunks.append(Chunk(len(chunks), label, " ".join(window)))
    else:
        words = " ".join(f"{h}. {b}" if h else b for h, b in doc.sections).split()
        for window in _windows(words, config.max_words, config.overlap_words):
            chunks.append(Chunk(len(chunks), doc.title, " ".join(window)))
    return chunks
