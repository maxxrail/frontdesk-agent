"""Ingest a folder of markdown policies for one tenant."""

from pathlib import Path

from frontdesk_agent.retrieval.chunking import ChunkingConfig
from frontdesk_agent.retrieval.store import IngestResult, PolicyStore


def read_policy_folder(directory: Path) -> dict[str, str]:
    """``{slug: markdown}`` for every ``*.md`` in the folder; the stem is the slug."""
    files = sorted(directory.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no markdown files in {directory}")
    return {f.stem: f.read_text(encoding="utf-8") for f in files}


async def ingest_documents(
    store: PolicyStore,
    documents: dict[str, str],
    config: ChunkingConfig | None = None,
    *,
    prune: bool = True,
) -> list[IngestResult]:
    """Ingest ``{slug: markdown}``. With ``prune`` (the default), documents not
    in the mapping are deleted, so the store mirrors its source folder."""
    results = [
        await store.ingest(slug=slug, markdown=markdown, config=config)
        for slug, markdown in documents.items()
    ]
    if prune:
        await store.remove_missing(set(documents))
    return results


async def ingest_directory(
    store: PolicyStore, directory: Path, config: ChunkingConfig | None = None, *, prune: bool = True
) -> list[IngestResult]:
    # A few small local files: reading them synchronously is cheaper than a thread hop.
    documents = read_policy_folder(directory)  # noqa: ASYNC240
    return await ingest_documents(store, documents, config, prune=prune)
