# ─────────────────────────────────────────────────────────────────────────────
# app/services/embeddings.py
#
# Text embedding service used by the vector store pipeline.
#
# WHY A SEPARATE EMBEDDINGS MODULE?
#   The vector store (Qdrant) needs to convert book titles + author names into
#   fixed-dimension float vectors before it can index or search them.
#   Keeping the embedding logic here (not inside QdrantVectorStoreService)
#   means:
#     1. The embedding model can be swapped independently of the vector DB
#        (e.g. switch from sentence-transformers to OpenAI embeddings without
#        touching any Qdrant code).
#     2. The embedding logic is independently testable — no Qdrant needed.
#     3. Batching, caching, and normalisation concerns live in one place.
#
# MODEL CHOICE: all-MiniLM-L6-v2
#   - 384-dimensional embeddings (compact, fast, low memory)
#   - Excellent semantic similarity for short English phrases (titles, names)
#   - ~80 MB model download; available on HuggingFace
#   - Cosine similarity works well with its output vectors
#
# SOLID note (Dependency Inversion):
#   QdrantVectorStoreService depends on BaseEmbeddingService (abstraction),
#   not on SentenceTransformerEmbeddingService (concrete). Tests inject a
#   DeterministicEmbeddingService that returns fixed vectors.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import hashlib
import struct
from abc import ABC, abstractmethod
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# Dimension of embeddings produced by all-MiniLM-L6-v2
EMBEDDING_DIM = 384


# ── Abstract contract ──────────────────────────────────────────────────────────

class BaseEmbeddingService(ABC):
    """
    Contract for converting text into fixed-dimension float vectors.

    All implementations must produce L2-normalised vectors so that 
    Cosine Similarity = dot product, which Qdrant can compute efficiently.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the number of dimensions in each embedding vector."""
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string into a float vector.

        Args:
            text: Any string — typically "{title} {author}" for book lookups.

        Returns:
            List of `self.dimension` floats, L2-normalised.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of text strings efficiently.

        Batch embedding is significantly faster than calling embed() in a loop
        because the model processes all inputs in a single forward pass.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, same order as input.
        """
        ...


# ── Production implementation ──────────────────────────────────────────────────

class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """
    Embeddings via the sentence-transformers library.

    Uses the all-MiniLM-L6-v2 model by default — a fast, compact model
    well-suited to short semantic similarity tasks like book title matching.

    The model is loaded lazily (on first call to embed/embed_batch) so
    service construction is fast and tests that never call embed() don't
    pay the download cost.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, 
                 model: Any = None,
                 model_name: str = DEFAULT_MODEL) -> None:
        """
        Args:
            model_name: HuggingFace model name or local path.
                        Defaults to "all-MiniLM-L6-v2".
        """
        self._model_name = model_name
        # Lazy-loaded; None until first call to embed() or embed_batch() 
        # NOTE: We are providing reference to same embedding model which we loaded for
        #       Gemma LLM and Embedding Model based recommendation task.
        self._model = model

    @property
    def dimension(self) -> int:
        """Returns EMBEDDING_DIM (384 for all-MiniLM-L6-v2)."""
        return EMBEDDING_DIM

    def embed(self, text: str) -> list[float]:
        """Embed a single string. Internally calls embed_batch([text])."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of strings in a single model forward pass.

        Normalise=True ensures vectors are L2-normalised so cosine similarity
        equals the dot product — this is required for Qdrant's cosine metric.
        """
        if self._model is None:
            self._load_model()

        logger.debug("embedding_batch", count=len(texts))

        # encode() returns a numpy array of shape (N, EMBEDDING_DIM)
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,   # L2 normalisation for cosine similarity
            show_progress_bar=False,     # No tqdm spam in production logs
            batch_size=32,               # Process up to 32 at a time
        )

        # Convert numpy array to plain Python list of lists for JSON-safety
        return [vec.tolist() for vec in vectors]

    def _load_model(self) -> None:
        """
        Lazy-load the SentenceTransformer model.

        Downloads from HuggingFace on first run; cached locally by the
        sentence-transformers library at ~/.cache/torch/sentence_transformers/.
        """
        from sentence_transformers import SentenceTransformer

        logger.info("embedding_model_loading", model=self._model_name)
        self._model = SentenceTransformer(self._model_name)
        logger.info("embedding_model_ready", model=self._model_name, dim=self.dimension)


# ── Test / stub implementation ─────────────────────────────────────────────────

class DeterministicEmbeddingService(BaseEmbeddingService):
    """
    Deterministic embedding service for tests.

    Produces 384-dimensional vectors from text by hashing — no model loading,
    no network calls. The same text always produces the same vector, enabling
    predictable similarity search tests.

    NOT suitable for production (hash-based vectors have no semantic meaning).
    """

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    def embed(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    @staticmethod
    def _hash_to_vector(text: str) -> list[float]:
        """
        Convert a string to a deterministic 384-float vector via MD5 hashing.

        The MD5 digest (16 bytes) is used as a seed to generate 384 floats.
        The vector is then L2-normalised so it behaves like a real embedding.
        """
        # MD5 for speed (not security; this is only for test determinism)
        digest = hashlib.md5(text.encode("utf-8")).digest()

        # Use the 16-byte digest as a repeating seed to generate 384 floats
        floats: list[float] = []
        seed = digest
        while len(floats) < EMBEDDING_DIM:
            # Unpack 4 bytes as an unsigned int, normalise to [-1, 1]
            for i in range(0, len(seed) - 3, 4):
                val = struct.unpack_from(">I", seed, i)[0]
                floats.append((val / 2_147_483_648.0) - 1.0)
            # Re-hash to generate more bytes if needed
            seed = hashlib.md5(seed).digest()

        raw = floats[:EMBEDDING_DIM]

        # L2 normalise
        magnitude = sum(x * x for x in raw) ** 0.5
        if magnitude == 0:
            return [0.0] * EMBEDDING_DIM
        return [x / magnitude for x in raw]
