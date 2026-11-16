"""Text embeddings behind a small interface.

The default is WordLlama, a static embedding model whose weights ship inside
its pip package: no API key, no network at runtime, CPU only, ~1,000 texts in
tens of milliseconds. It is much weaker than a large hosted model, which is
exactly why retrieval is hybrid and measured (ADR 0008). Swapping in a hosted
model means writing one more class with this interface and re-ingesting.
"""

from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

EMBEDDING_DIM = 256


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Stored on every chunk, so stale embeddings are detectable."""
        ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one unit-length vector per text."""
        ...


class WordLlamaEmbedder:
    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        import wordllama
        from wordllama import WordLlama

        # The wheel bundles its weights and tokenizer under the package folder,
        # but WordLlama 0.4 looks for the tokenizer in a misnamed directory.
        # Pointing cache_dir at the package folder finds both files, and
        # disable_download guarantees we never fetch anything at runtime.
        package_dir = Path(wordllama.__file__).parent
        self._model = WordLlama.load(dim=dim, cache_dir=package_dir, disable_download=True)
        self._dim = dim

    @property
    def name(self) -> str:
        return f"wordllama-l2_supercat-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = np.asarray(self._model.embed(texts), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # An all-stopword text can embed to zero; leave it as zero, never NaN.
        vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
        return [row.tolist() for row in vectors]


@lru_cache
def default_embedder() -> Embedder:
    return WordLlamaEmbedder()
