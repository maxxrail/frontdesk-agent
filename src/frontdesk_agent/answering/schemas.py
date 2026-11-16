"""Request and response bodies for policy questions."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field

RefusalReason = Literal["no_relevant_documents", "not_in_documents", "ungrounded_answer"]


class PolicyQuestion(BaseModel):
    # Stripped before the length check, so "   " is rejected, not embedded.
    model_config = {"str_strip_whitespace": True}

    question: str = Field(min_length=1, max_length=500)


class ModelAnswer(BaseModel):
    """The JSON the model must return. Extra keys are a validation failure."""

    model_config = {"extra": "forbid"}

    answerable: bool
    answer: str = Field(max_length=600)
    citations: list[str] = Field(max_length=10)


class Citation(BaseModel):
    chunk_id: uuid.UUID
    document_slug: str
    heading: str
    excerpt: str


class PolicyAnswer(BaseModel):
    answerable: bool
    answer: str
    needs_human: bool = Field(
        description="True when the client should get a person, not this answer."
    )
    refusal_reason: RefusalReason | None = None
    citations: list[Citation]
    retrieval_mode: str
    top_similarity: float | None
    model: str | None = None
    prompt_version: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


class SearchResult(BaseModel):
    chunk_id: uuid.UUID
    document_slug: str
    heading: str
    content: str
    score: float
    vector_similarity: float | None


class PolicyDocumentIn(BaseModel):
    slug: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    markdown: str = Field(min_length=1, max_length=100_000)


class IngestOut(BaseModel):
    slug: str
    chunks: int
    skipped: bool
