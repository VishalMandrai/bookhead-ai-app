# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_embeddings.py
#
# Unit tests for app/services/embeddings.py
#
# SentenceTransformerEmbeddingService is NEVER loaded here.
# All tests use DeterministicEmbeddingService (hash-based, no model).
# One test class verifies the contract that any implementation must satisfy.
# ─────────────────────────────────────────────────────────────────────────────

import pytest

from app.services.embeddings import (
    EMBEDDING_DIM,
    DeterministicEmbeddingService,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _l2_norm(vec: list[float]) -> float:
    """Compute the L2 norm of a float vector."""
    return sum(x * x for x in vec) ** 0.5


class TestDeterministicEmbeddingService:
    """
    Tests for DeterministicEmbeddingService AND, by proxy, the
    BaseEmbeddingService contract that all implementations must honour.
    """

    def setup_method(self):
        self.svc = DeterministicEmbeddingService()

    # ── dimension property ────────────────────────────────────────────────────

    def test_dimension_equals_embedding_dim_constant(self):
        """dimension property must equal the module-level EMBEDDING_DIM constant."""
        assert self.svc.dimension == EMBEDDING_DIM

    def test_dimension_is_384(self):
        """For all-MiniLM-L6-v2 compatibility the dimension must be 384."""
        assert self.svc.dimension == 384

    # ── embed() single string ─────────────────────────────────────────────────

    def test_embed_returns_list(self):
        vec = self.svc.embed("Dune Frank Herbert")
        assert isinstance(vec, list)

    def test_embed_returns_correct_length(self):
        vec = self.svc.embed("The Great Gatsby F. Scott Fitzgerald")
        assert len(vec) == EMBEDDING_DIM

    def test_embed_returns_floats(self):
        vec = self.svc.embed("1984 George Orwell")
        assert all(isinstance(x, float) for x in vec)

    def test_embed_is_l2_normalised(self):
        """All embedding vectors must have unit L2 norm (within float tolerance)."""
        vec = self.svc.embed("To Kill a Mockingbird Harper Lee")
        norm = _l2_norm(vec)
        assert abs(norm - 1.0) < 1e-6, f"Expected unit norm, got {norm}"

    def test_embed_is_deterministic(self):
        """Same input text must always produce the same vector."""
        text = "Harry Potter J.K. Rowling"
        vec1 = self.svc.embed(text)
        vec2 = self.svc.embed(text)
        assert vec1 == vec2

    def test_different_texts_produce_different_vectors(self):
        """Different input texts must produce different vectors."""
        vec1 = self.svc.embed("Dune Frank Herbert")
        vec2 = self.svc.embed("Foundation Isaac Asimov")
        assert vec1 != vec2

    def test_empty_string_does_not_crash(self):
        """Embedding an empty string must not raise an exception."""
        vec = self.svc.embed("")
        assert len(vec) == EMBEDDING_DIM

    def test_embed_single_word(self):
        vec = self.svc.embed("Dune")
        assert len(vec) == EMBEDDING_DIM

    def test_embed_long_text(self):
        """Long text (full titles + subtitles) should embed without error."""
        long_text = "The Hitchhiker's Guide to the Galaxy: A Trilogy in Five Parts Douglas Adams"
        vec = self.svc.embed(long_text)
        assert len(vec) == EMBEDDING_DIM

    # ── embed_batch() ─────────────────────────────────────────────────────────

    def test_embed_batch_returns_list_of_lists(self):
        texts = ["Dune", "Foundation", "1984"]
        result = self.svc.embed_batch(texts)
        assert isinstance(result, list)
        assert all(isinstance(v, list) for v in result)

    def test_embed_batch_preserves_order(self):
        """Output must be in the same order as input."""
        texts = ["Dune Frank Herbert", "Foundation Isaac Asimov", "1984 George Orwell"]
        batch_result = self.svc.embed_batch(texts)
        single_results = [self.svc.embed(t) for t in texts]
        assert batch_result == single_results

    def test_embed_batch_each_vector_correct_length(self):
        texts = ["Book A", "Book B", "Book C", "Book D"]
        result = self.svc.embed_batch(texts)
        assert all(len(v) == EMBEDDING_DIM for v in result)

    def test_embed_batch_each_vector_normalised(self):
        texts = ["Dune Frank Herbert", "Foundation Isaac Asimov"]
        result = self.svc.embed_batch(texts)
        for vec in result:
            assert abs(_l2_norm(vec) - 1.0) < 1e-6

    def test_embed_batch_empty_list(self):
        """An empty batch must return an empty list without error."""
        result = self.svc.embed_batch([])
        assert result == []

    def test_embed_batch_single_item(self):
        """A batch of one item should equal embed() on that item."""
        text = "Brave New World Aldous Huxley"
        assert self.svc.embed_batch([text]) == [self.svc.embed(text)]

    # ── Consistency between embed() and embed_batch() ─────────────────────────

    def test_embed_and_embed_batch_agree(self):
        """
        embed(text) must produce the same result as embed_batch([text])[0].
        This ensures both methods share the same underlying computation.
        """
        texts = [
            "The Lord of the Rings J.R.R. Tolkien",
            "Crime and Punishment Fyodor Dostoevsky",
        ]
        for text in texts:
            single = self.svc.embed(text)
            from_batch = self.svc.embed_batch([text])[0]
            assert single == from_batch
