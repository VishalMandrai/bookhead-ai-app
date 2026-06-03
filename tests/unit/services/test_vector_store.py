# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_vector_store.py
#
# Unit tests for QdrantVectorStoreService.
#
# Qdrant is NEVER contacted. A MagicMock replaces the QdrantClient so every
# test runs without a running Qdrant instance. DeterministicEmbeddingService
# replaces sentence-transformers so no model is downloaded.
#
# What we test:
#   - search() returns None on cache miss (no results / below threshold)
#   - search() returns a reconstructed BookRecord on cache hit
#   - upsert() calls the Qdrant client with the correct point structure
#   - upsert_batch() embeds all books and sends a single Qdrant call
#   - _ensure_collection() creates the collection when it doesn't exist
#   - _ensure_collection() skips creation when collection already exists
#   - Deterministic point IDs: same title+author → same UUID
#   - _build_query_text normalises case and whitespace
# ─────────────────────────────────────────────────────────────────────────────

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch, call

import pytest

from app.models.book import BookRecord
from app.services.embeddings import DeterministicEmbeddingService
from app.services.vector_store import (
    QdrantVectorStoreService,
    _build_query_text,
    _deterministic_point_id,
    _book_record_to_payload,
    _payload_to_book_record,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_service(similarity_threshold: float = 0.92) -> QdrantVectorStoreService:
    """
    Build a QdrantVectorStoreService with:
      - A pre-injected mock Qdrant client (no network)
      - DeterministicEmbeddingService (no model download)
      - _collection_ready=True to skip _ensure_collection() in most tests
    """
    svc = QdrantVectorStoreService(
        host="localhost",
        port=6333,
        collection_name="books_test",
        similarity_threshold=similarity_threshold,
        embedding_service=DeterministicEmbeddingService(),
    )
    # Inject mock client and skip collection bootstrap
    svc._client = MagicMock()
    svc._collection_ready = True
    return svc


def _make_book(
    title: str = "Dune",
    author: str = "Frank Herbert",
    source: str = "ocr_auto",
) -> BookRecord:
    return BookRecord(
        book_id=str(uuid.uuid4()),
        title=title,
        author=author,
        crop_image_path="crops/job/book_0.jpg",
        ocr_confidence=0.91,
        source=source,  # type: ignore[arg-type]
        genre="Science Fiction",
        genre_code="SCI",
        summary="A sweeping sci-fi epic.",
    )


def _make_qdrant_result(payload: dict, score: float) -> MagicMock:
    """Build a fake Qdrant ScoredPoint-like object."""
    result = MagicMock()
    result.score = score
    result.payload = payload
    return result


# ── Tests: search() ────────────────────────────────────────────────────────────

class TestVectorStoreSearch:

    def test_returns_none_on_empty_results(self):
        """When Qdrant returns no results, search() must return None."""
        svc = _make_service()
        svc._client.search.return_value = []

        result = svc.search("Unknown Book", "Unknown Author")

        assert result is None

    def test_returns_none_when_below_threshold(self):
        """
        If all results are below the similarity threshold (Qdrant filters them
        via score_threshold), search() receives an empty list and returns None.
        """
        svc = _make_service(similarity_threshold=0.92)
        # Qdrant applies score_threshold server-side and returns [] for misses
        svc._client.search.return_value = []

        result = svc.search("Dune", "Frank Herbert")

        assert result is None

    def test_returns_book_record_on_cache_hit(self):
        """A result above threshold must be reconstructed into a BookRecord."""
        svc = _make_service()
        book = _make_book()
        payload = _book_record_to_payload(book)

        svc._client.search.return_value = [_make_qdrant_result(payload, score=0.97)]

        result = svc.search("Dune", "Frank Herbert")

        assert result is not None
        assert isinstance(result, BookRecord)

    def test_cache_hit_preserves_title(self):
        svc = _make_service()
        book = _make_book(title="Foundation", author="Isaac Asimov")
        payload = _book_record_to_payload(book)
        svc._client.search.return_value = [_make_qdrant_result(payload, score=0.95)]

        result = svc.search("Foundation", "Isaac Asimov")

        assert result.title == "Foundation"

    def test_cache_hit_preserves_author(self):
        svc = _make_service()
        book = _make_book(title="Foundation", author="Isaac Asimov")
        payload = _book_record_to_payload(book)
        svc._client.search.return_value = [_make_qdrant_result(payload, score=0.95)]

        result = svc.search("Foundation", "Isaac Asimov")

        assert result.author == "Isaac Asimov"

    def test_cache_hit_preserves_genre(self):
        """All enrichment fields (genre, summary) must survive the round-trip."""
        svc = _make_service()
        book = _make_book()
        book.genre = "Science Fiction"
        book.genre_code = "SCI"
        payload = _book_record_to_payload(book)
        svc._client.search.return_value = [_make_qdrant_result(payload, score=0.96)]

        result = svc.search("Dune", "Frank Herbert")

        assert result.genre == "Science Fiction"
        assert result.genre_code == "SCI"

    def test_search_calls_qdrant_with_correct_collection(self):
        """search() must query the configured collection name."""
        svc = _make_service()
        svc._client.search.return_value = []

        svc.search("Dune", "Frank Herbert")

        call_kwargs = svc._client.search.call_args[1]
        assert call_kwargs["collection_name"] == "books_test"

    def test_search_passes_score_threshold(self):
        """search() must pass the similarity_threshold to Qdrant."""
        svc = _make_service(similarity_threshold=0.95)
        svc._client.search.return_value = []

        svc.search("Dune", "Frank Herbert")

        call_kwargs = svc._client.search.call_args[1]
        assert call_kwargs["score_threshold"] == 0.95

    def test_search_requests_payload(self):
        """search() must set with_payload=True to get stored metadata."""
        svc = _make_service()
        svc._client.search.return_value = []

        svc.search("Dune", "Frank Herbert")

        call_kwargs = svc._client.search.call_args[1]
        assert call_kwargs["with_payload"] is True


# ── Tests: upsert() ────────────────────────────────────────────────────────────

class TestVectorStoreUpsert:

    def test_upsert_calls_qdrant_client(self):
        """upsert() must call the Qdrant client's upsert method exactly once."""
        svc = _make_service()
        book = _make_book()

        svc.upsert(book)

        svc._client.upsert.assert_called_once()

    def test_upsert_uses_correct_collection(self):
        svc = _make_service()
        svc.upsert(_make_book())

        call_kwargs = svc._client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "books_test"

    def test_upsert_sends_single_point(self):
        """upsert() for one book must send exactly one PointStruct."""
        svc = _make_service()
        svc.upsert(_make_book())

        points = svc._client.upsert.call_args[1]["points"]
        assert len(points) == 1

    def test_upsert_point_id_is_deterministic(self):
        """
        Two upserts for the same title+author must produce the same point ID,
        making upsert idempotent.
        """
        svc = _make_service()
        book = _make_book(title="Dune", author="Frank Herbert")

        svc.upsert(book)
        first_id = svc._client.upsert.call_args[1]["points"][0].id

        svc._client.reset_mock()
        svc.upsert(book)
        second_id = svc._client.upsert.call_args[1]["points"][0].id

        assert first_id == second_id

    def test_upsert_vector_is_correct_dimension(self):
        """The vector stored in Qdrant must have EMBEDDING_DIM dimensions."""
        from app.services.embeddings import EMBEDDING_DIM

        svc = _make_service()
        svc.upsert(_make_book())

        point = svc._client.upsert.call_args[1]["points"][0]
        assert len(point.vector) == EMBEDDING_DIM

    def test_upsert_payload_contains_title(self):
        svc = _make_service()
        book = _make_book(title="Neuromancer")
        svc.upsert(book)

        payload = svc._client.upsert.call_args[1]["points"][0].payload
        assert payload["title"] == "Neuromancer"

    def test_upsert_payload_contains_source(self):
        """The source field must be preserved in the Qdrant payload."""
        svc = _make_service()
        book = _make_book(source="human_corrected")
        svc.upsert(book)

        payload = svc._client.upsert.call_args[1]["points"][0].payload
        assert payload["source"] == "human_corrected"


# ── Tests: upsert_batch() ──────────────────────────────────────────────────────

class TestVectorStoreBatchUpsert:

    def test_batch_upsert_sends_correct_number_of_points(self):
        svc = _make_service()
        books = [_make_book(title=f"Book {i}", author=f"Author {i}") for i in range(5)]

        svc.upsert_batch(books)

        points = svc._client.upsert.call_args[1]["points"]
        assert len(points) == 5

    def test_batch_upsert_empty_list_does_not_call_qdrant(self):
        """An empty batch must not trigger any Qdrant call."""
        svc = _make_service()

        svc.upsert_batch([])

        svc._client.upsert.assert_not_called()

    def test_batch_upsert_each_point_has_payload(self):
        svc = _make_service()
        books = [_make_book(title=f"Book {i}", author="Author") for i in range(3)]

        svc.upsert_batch(books)

        points = svc._client.upsert.call_args[1]["points"]
        for point in points:
            assert "title" in point.payload


# ── Tests: _ensure_collection() ───────────────────────────────────────────────

class TestEnsureCollection:

    def test_creates_collection_when_missing(self):
        """If the collection does not exist, it must be created."""
        svc = QdrantVectorStoreService(
            collection_name="books_test",
            embedding_service=DeterministicEmbeddingService(),
        )
        mock_client = MagicMock()
        # Simulate collection not existing
        mock_client.get_collections.return_value.collections = []
        svc._client = mock_client

        svc._ensure_collection()

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "books_test"

    def test_skips_creation_when_collection_exists(self):
        """If the collection already exists, create_collection must NOT be called."""
        svc = QdrantVectorStoreService(
            collection_name="books_test",
            embedding_service=DeterministicEmbeddingService(),
        )
        mock_client = MagicMock()
        existing = MagicMock()
        existing.name = "books_test"
        mock_client.get_collections.return_value.collections = [existing]
        svc._client = mock_client

        svc._ensure_collection()

        mock_client.create_collection.assert_not_called()

    def test_sets_collection_ready_flag(self):
        svc = QdrantVectorStoreService(
            collection_name="books_test",
            embedding_service=DeterministicEmbeddingService(),
        )
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        svc._client = mock_client

        assert svc._collection_ready is False
        svc._ensure_collection()
        assert svc._collection_ready is True

    def test_fast_path_when_ready(self):
        """When _collection_ready=True, get_collections must NOT be called."""
        svc = QdrantVectorStoreService(
            collection_name="books_test",
            embedding_service=DeterministicEmbeddingService(),
        )
        mock_client = MagicMock()
        svc._client = mock_client
        svc._collection_ready = True  # Already ready

        svc._ensure_collection()

        mock_client.get_collections.assert_not_called()


# ── Tests: helper functions ────────────────────────────────────────────────────

class TestHelperFunctions:

    def test_build_query_text_combines_title_and_author(self):
        assert _build_query_text("Dune", "Frank Herbert") == "dune frank herbert"

    def test_build_query_text_lowercases(self):
        assert _build_query_text("DUNE", "FRANK HERBERT") == "dune frank herbert"

    def test_build_query_text_strips_whitespace(self):
        result = _build_query_text("  Dune  ", "  Frank Herbert  ")
        assert result == "dune frank herbert"

    def test_build_query_text_collapses_spaces(self):
        result = _build_query_text("Dune", "Frank  Herbert")
        assert "  " not in result

    def test_deterministic_point_id_is_uuid_string(self):
        point_id = _deterministic_point_id("Dune", "Frank Herbert")
        # Must be a valid UUID string
        uuid.UUID(point_id)

    def test_deterministic_point_id_same_for_same_input(self):
        id1 = _deterministic_point_id("Dune", "Frank Herbert")
        id2 = _deterministic_point_id("Dune", "Frank Herbert")
        assert id1 == id2

    def test_deterministic_point_id_different_for_different_input(self):
        id1 = _deterministic_point_id("Dune", "Frank Herbert")
        id2 = _deterministic_point_id("Foundation", "Isaac Asimov")
        assert id1 != id2

    def test_deterministic_point_id_case_insensitive(self):
        """IDs for "Dune" and "dune" must be the same (uses query text normalisation)."""
        id1 = _deterministic_point_id("Dune", "Frank Herbert")
        id2 = _deterministic_point_id("DUNE", "FRANK HERBERT")
        assert id1 == id2

    def test_book_record_round_trip(self):
        """Serialising and deserialising a BookRecord must preserve all fields."""
        book = _make_book()
        payload = _book_record_to_payload(book)
        recovered = _payload_to_book_record(payload)

        assert recovered.title == book.title
        assert recovered.author == book.author
        assert recovered.source == book.source
        assert recovered.genre == book.genre
        assert recovered.genre_code == book.genre_code

    def test_payload_datetime_is_string(self):
        """created_at must be serialised as an ISO string for JSON compatibility."""
        book = _make_book()
        payload = _book_record_to_payload(book)
        assert isinstance(payload["created_at"], str)
