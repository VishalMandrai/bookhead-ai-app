# ─────────────────────────────────────────────────────────────────────────────
# app/services/vector_store.py
#
# QdrantVectorStoreService – concrete implementation of 'BaseVectorStoreService' (from base.py).
#
# WHAT IT DOES:
#   Every BookRecord that flows through the pipeline is stored here as a
#   vector + JSON payload in Qdrant.  On subsequent jobs, before calling the
#   expensive LLM, we first query Qdrant with the OCR-extracted title+author.
#   A high-similarity hit (cache hit) means we already know this book and can
#   skip the LLM entirely.  A miss means we proceed to the LLM and then write
#   the enriched record back.
#
# QDRANT DATA MODEL:
#   Collection : "books"  (one collection for the whole app)
#   Point ID   : deterministic UUID derived from title+author (so the same
#                book never gets duplicate entries)
#   Vector     : 384-float L2-normalised embedding of "{title} {author}"
#   Payload    : all BookRecord fields serialised as JSON
#
# SIMILARITY SEARCH STRATEGY:
#   We embed the query text and do a nearest-neighbour search with cosine
#   similarity.  Only the top-1 result is returned, and only if its score
#   exceeds SIMILARITY_THRESHOLD (default 0.92).  

#   NOTE: This threshold was chosen to match near-identical titles (same book,
#   slightly different OCR reading) while rejecting genuinely different books
#   that happen to share words.
#
# WRITE-BACK QUALITY TAGGING:
#   Payload includes a "verified" boolean.  Human-corrected records are tagged
#   verified=True; OCR-auto records are tagged verified=False.  Future work
#   can use this to re-rank search results or trigger model fine-tuning.
#
# SOLID (application):
#   Single Responsibility  – only talks to Qdrant. Transformation to Embeddings is delegated to
#                            BaseEmbeddingService's concrete implementation 
#                            SentenceTransformerEmbeddingService.
#   Dependency Inversion   – depends on BaseEmbeddingService (abstract), not
#                            SentenceTransformerEmbeddingService (concrete).
#   Open/Closed            – a PineconeVectorStoreService could replace this
#                            without changing any pipeline or route code.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid
from typing import Any

from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.models.book import BookRecord
from app.services.base import BaseVectorStoreService
from app.services.embeddings import (
    EMBEDDING_DIM,
    BaseEmbeddingService,
    SentenceTransformerEmbeddingService,
)

logger = get_logger(__name__)

# ── Tuning constants ────────────────────────────────────────────────────────────

# Cosine similarity threshold for a cache hit.
# Scores are in [-1, 1] for raw cosine; Qdrant returns scores in [0, 1] for
# cosine metric (maps -1→0, 0→0.5, 1→1).
# 0.92 means the query and stored vector share ~92 % of their directional
# information — tight enough to match "Dune" vs "Dune " (OCR trailing space)
# but loose enough to reject "Dune" vs "Dune Messiah".
SIMILARITY_THRESHOLD = 0.92

# Maximum number of Qdrant search results to retrieve; we only need the top-1
# but fetching slightly more gives a safety margin in case the top result is
# just below threshold.
SEARCH_TOP_K = 3


class QdrantVectorStoreService(BaseVectorStoreService):
    """
    Qdrant-backed vector store for book metadata.

    Construction is cheap — no network calls happen until the first
    search() or upsert() call triggers _ensure_collection().
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "books",
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        embedding_service: BaseEmbeddingService | None = None,
    ) -> None:
        """
        Args:
            host:                 Qdrant server hostname.
            port:                 Qdrant REST/gRPC port.
            collection_name:      Name of the Qdrant collection to use.
            similarity_threshold: Minimum cosine score for a cache hit (0–1).
            embedding_service:    Injectable embedding backend. Defaults to
                                  SentenceTransformerEmbeddingService.
                                  Pass DeterministicEmbeddingService in tests.
        """
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._similarity_threshold = similarity_threshold
        self._embedder: BaseEmbeddingService = (
            embedding_service or SentenceTransformerEmbeddingService()
        )

        # Lazy-loaded Qdrant client; None until first real operation
        self._client: Any | None = None
        # Tracks whether we've confirmed the collection exists this session
        self._collection_ready = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def search(self, title: str, author: str) -> BookRecord | None:
        """
        Search for a book by semantic similarity of title + author.

        Returns the best-matching BookRecord if its similarity score exceeds
        SIMILARITY_THRESHOLD, otherwise returns None (cache miss).

        The combination string "{title} {author}" is used as the query text
        so that both fields contribute to the similarity score.
        """
        self._ensure_collection()

        # Build the query text in the same format used during upsert
        query_text = _build_query_text(title, author)
        query_vector = self._embedder.embed(query_text)

        logger.debug(
            "vector_store_search",
            title=title,
            author=author,
            query_text=query_text,
        )
        
        while True:
            try:
                results = self._client.search(
                    collection_name=self._collection_name,
                    query_vector=query_vector,
                    limit=SEARCH_TOP_K,
                    with_payload=True,   # Return the stored JSON payload
                    score_threshold=self._similarity_threshold,
                )
                break
            
            except Exception as exc:
                logger.error(
                    f"Qdrant search failed: {exc}. Retrying again....",
                    detail={"title": title, "author": author},
                )

        if not results:
            logger.info(
                "vector_store_cache_miss",
                title=title,
                author=author,
            )
            return None

        # Take the top result (highest similarity score)
        top_result = results[0]
        logger.info(
            "vector_store_cache_hit",
            title=title,
            author=author,
            score=top_result.score,
        )

        # Reconstruct BookRecord from the stored payload
        return _payload_to_book_record(top_result.payload)

    def upsert(self, book: BookRecord) -> None:
        """
        Insert or update a BookRecord in the vector store.

        Uses a deterministic point ID derived from the book's title+author
        so the same book is never duplicated — upsert is idempotent.

        Human-corrected records (source="human_corrected") are tagged
        verified=True in the payload, marking them as higher-quality data.
        """
        self._ensure_collection()

        # Build the text that gets embedded (same format as search)
        text = _build_query_text(book.title, book.author)
        vector = self._embedder.embed(text)

        # Deterministic point ID: same title+author always produces the same ID
        # This makes upsert idempotent — re-processing the same book updates
        # the existing record instead of creating a duplicate.
        point_id = _deterministic_point_id(book.title, book.author)

        # Serialise BookRecord to a plain dict payload
        # We store ALL fields so we can reconstruct a full BookRecord on cache hit
        payload = _book_record_to_payload(book)

        while True:
            try:
                from qdrant_client.models import PointStruct

                self._client.upsert(
                    collection_name=self._collection_name,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload,
                        )
                    ],
                )
                break
            
            except Exception as exc:
                logger.error(
                    f"Qdrant upsert failed for '{book.title}': {exc}",
                    detail={"book_id": book.book_id, "title": book.title},
                )
    

        logger.info(
            "vector_store_upsert_ok",
            title=book.title,
            author=book.author,
            point_id=point_id,
            verified=(book.source == "human_corrected"),
        )

    def upsert_batch(self, books: list[BookRecord]) -> None:
        """
        Upsert multiple BookRecords in a single Qdrant batch call.

        More efficient than calling upsert() in a loop for large catalogs
        (a librarian may process 50+ books at once).

        Args:
            books: List of BookRecord objects to upsert.
        """
        if not books:
            return

        self._ensure_collection()

        # Embed all titles+authors in a single batch forward pass
        texts = [_build_query_text(b.title, b.author) for b in books]
        vectors = self._embedder.embed_batch(texts)

        while True:
            try:
                from qdrant_client.models import PointStruct

                points = [
                    PointStruct(
                        id=_deterministic_point_id(book.title, book.author),
                        vector=vector,
                        payload=_book_record_to_payload(book),
                    )
                    for book, vector in zip(books, vectors)
                ]

                self._client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                )
                break
            except Exception as exc:
                logger.error(
                    f"Qdrant batch upsert failed: {exc}. Retrying again....",
                    detail={"count": len(books)},
                )

        logger.info(
            "vector_store_batch_upsert_ok",
            count=len(books),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """
        Lazily connect to Qdrant and ensure the books collection exists.

        Called at the start of every search() and upsert() call.
        After the first successful call, _collection_ready is set to True
        and this method becomes a fast no-op.

        The collection uses cosine distance — appropriate for L2-normalised
        embeddings from sentence-transformers.
        """
        if self._collection_ready:
            return  # Fast path: already verified

        # Lazy-load the Qdrant client on first operation
        if self._client is None:
            self._client = self._build_client()

        while True:
            try:
                from qdrant_client.models import Distance, VectorParams

                # Check if the collection already exists
                existing = [
                    c.name
                    for c in self._client.get_collections().collections
                ]

                if self._collection_name not in existing:
                    logger.info(
                        "vector_store_creating_collection",
                        collection=self._collection_name,
                        dim=self._embedder.dimension,
                    )
                    
                    ## Creating a new collection with name "books":
                    self._client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=VectorParams(
                            size=self._embedder.dimension,
                            distance=Distance.COSINE,
                        ),
                    )
                else:
                    logger.debug(
                        "vector_store_collection_exists",
                        collection=self._collection_name,
                    )
                    

                self._collection_ready = True
                break

            except Exception as exc:
                logger.error(
                    f"Failed to initialise Qdrant collection '{self._collection_name}': {exc}",
                    retry="Trying again"
                )



    def _build_client(self):
        """
        Instantiate the Qdrant HTTP client.

        Lazy import keeps startup fast when Qdrant is not needed (e.g. tests).
        """
        from qdrant_client import QdrantClient

        logger.info(
            "vector_store_connecting",
            host=self._host,
            port=self._port,
        )
        return QdrantClient(host=self._host, port=self._port, timeout=100)



# ── Module-level helpers (pure functions, no state) ────────────────────────────

def _build_query_text(title: str, author: str) -> str:
    """
    Concatenate title and author into a single query string.

    Normalise to lowercase and strip extra whitespace so that minor OCR
    variations ("Dune" vs "dune", "Frank Herbert " vs "Frank Herbert")
    produce vectors that are very close in embedding space.
    """
    combined = f"{title} {author}".strip()
    # Lowercase and collapse whitespace for embedding consistency
    return " ".join(combined.lower().split())


def _deterministic_point_id(title: str, author: str) -> str:
    """
    Generate a deterministic Qdrant point ID from title + author.

    Uses UUID v5 (namespace SHA-1 hash) so the same book always maps to
    the same ID. This makes upsert() idempotent — safe to call multiple
    times for the same book.

    Returns:
        A UUID string (e.g. "550e8400-e29b-41d4-a716-446655440000").
    """
    # UUID v5 with the DNS namespace as a stable base
    return str(
        uuid.uuid5(uuid.NAMESPACE_DNS, _build_query_text(title, author))
    )


def _book_record_to_payload(book: BookRecord) -> dict:
    """
    Serialise a BookRecord to a plain dict for Qdrant payload storage.

    We store all fields including enrichment data (genre, summary, etc.)
    so a cache hit returns a fully-hydrated BookRecord without any extra
    LLM calls.

    datetime fields are converted to ISO strings for JSON compatibility.
    """
    data = book.model_dump()
    # Convert datetime to ISO string (Qdrant payload must be JSON-serialisable)
    if "created_at" in data and data["created_at"] is not None:
        data["created_at"] = data["created_at"].isoformat()
    return data


def _payload_to_book_record(payload: dict) -> BookRecord:
    """
    Reconstruct a BookRecord from a Qdrant payload dict.

    Pydantic handles type coercion (e.g. ISO string → datetime) automatically.
    """
    return BookRecord(**payload)
