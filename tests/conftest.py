# ─────────────────────────────────────────────────────────────────────────────
# tests/conftest.py
#
# Shared pytest fixtures available to ALL test modules.
#
# conftest.py is auto-loaded by pytest before any test file runs. Fixtures
# defined here are injected by name into test functions without explicit imports.
#
# Key fixtures provided:
#   settings        – A test-safe Settings instance with overridden values
#   app             – FastAPI app with all stub service overrides applied
#   client          – Synchronous TestClient for making HTTP requests in tests
#   sample_image    – An in-memory PNG image as bytes (no real file needed)
#   stub_*          – Lightweight in-memory stubs for every service ABC
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import io
import uuid
from typing import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.core.config import Settings, get_settings
from main import create_app
from app.api.dependencies import (
    get_detection_service,
    get_ocr_service,
    get_confidence_router,
    get_review_queue_service,
    get_result_merger,
    get_vector_store_service,
    get_catalog_service,
    get_recommendation_service,
)
from app.models.book import BoundingBox, BookCorrection, BookRecord, OCRResult
from app.models.request import ReaderPreferences
from app.services.base import (
    BaseDetectionService,
    BaseOCRService,
    BaseConfidenceRouter,
    BaseReviewQueueService,
    BaseResultMerger,
    BaseVectorStoreService,
    BaseCatalogService,
    BaseRecommendationService,
)


# ─────────────────────────────────────────────────────────────────────────────
# Settings override
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def settings() -> Settings:
    """
    Returns a Settings instance configured for testing.
    Uses a temp upload dir and a low confidence threshold for predictable routing.
    Clears the lru_cache so tests always get a fresh instance.
    """
    get_settings.cache_clear()
    return Settings(
        env="test",
        anthropic_api_key="test-key",
        redis_url="redis://localhost:6379/1",    # DB 1: separate from dev
        upload_dir="/tmp/booklens_test_uploads",
        ocr_confidence_threshold=0.75,
        log_level="WARNING",                     # Suppress noise in test output
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stub service implementations
# These are lightweight in-memory fakes that implement the ABCs.
# They never touch models, databases, or external APIs.
# ─────────────────────────────────────────────────────────────────────────────

class StubDetectionService(BaseDetectionService):
    """Returns two pre-canned bounding boxes for any input image."""

    def detect(self, image, job_id: str):
        boxes = [
            BoundingBox(x_min=0, y_min=0, x_max=100, y_max=300, confidence=0.95),
            BoundingBox(x_min=110, y_min=0, x_max=210, y_max=300, confidence=0.87),
        ]
        # Return fake crop paths alongside the boxes
        crop_paths = [
            f"{job_id}_book_0.jpg",
            f"{job_id}_book_1.jpg",
        ]
        return boxes, crop_paths


class StubOCRService(BaseOCRService):
    """Returns one high-confidence and one low-confidence OCR result."""

    def extract(self, crop_path: str, book_id: str) -> OCRResult:
        # Simulate low confidence for the second book (index embedded in path)
        confidence = 0.90 if "book_0" in crop_path else 0.50
        return OCRResult(
            book_id=book_id,
            crop_image_path=crop_path,
            raw_title="The Great Gatsby" if "book_0" in crop_path else "Blurry Book",
            raw_author="F. Scott Fitzgerald" if "book_0" in crop_path else "Unknown",
            confidence=confidence,
            flagged_for_review=False,   # Router sets this; not OCR service
        )


class StubConfidenceRouter(BaseConfidenceRouter):
    """Routes books below 0.75 confidence to the review queue."""

    def route(self, ocr_results):
        auto_accepted = [r for r in ocr_results if r.confidence >= 0.75]
        flagged = [r for r in ocr_results if r.confidence < 0.75]
        for r in flagged:
            r.flagged_for_review = True
        return auto_accepted, flagged


class StubReviewQueueService(BaseReviewQueueService):
    """In-memory review queue – no Redis needed in tests."""

    def __init__(self):
        # session_id → {"flagged": [...], "corrections": [...], "complete": bool}
        self._store: dict = {}

    def create_session(self, job_id: str, flagged_books: list[OCRResult]) -> None:
        self._store[job_id] = {
            "flagged": flagged_books,
            "corrections": [],
            "complete": False,
        }

    def get_pending(self, job_id: str) -> list[OCRResult]:
        self._assert_exists(job_id)
        return self._store[job_id]["flagged"]

    def submit_corrections(self, job_id: str, corrections: list[BookCorrection]) -> None:
        self._assert_exists(job_id)
        self._store[job_id]["corrections"] = corrections
        self._store[job_id]["complete"] = True

    def get_corrections(self, job_id: str) -> list[BookCorrection]:
        self._assert_exists(job_id)
        return self._store[job_id]["corrections"]

    def is_complete(self, job_id: str) -> bool:
        return self._store.get(job_id, {}).get("complete", False)

    def _assert_exists(self, job_id: str):
        from app.core.exceptions import ReviewSessionNotFoundError
        if job_id not in self._store:
            raise ReviewSessionNotFoundError(f"No review session for job {job_id}")


class StubResultMerger(BaseResultMerger):
    """Merges OCR results and corrections into BookRecords without any IO."""

    def merge(self, auto_accepted, corrections, flagged_originals):
        records = []

        # Auto-accepted books → source = "ocr_auto"
        for ocr in auto_accepted:
            records.append(BookRecord(
                book_id=ocr.book_id,
                title=ocr.raw_title,
                author=ocr.raw_author,
                crop_image_path=ocr.crop_image_path,
                ocr_confidence=ocr.confidence,
                source="ocr_auto",
            ))

        # Corrected books → source = "human_corrected"
        flagged_map = {o.book_id: o for o in flagged_originals}
        for correction in corrections:
            original = flagged_map.get(correction.book_id)
            records.append(BookRecord(
                book_id=correction.book_id,
                title=correction.corrected_title,
                author=correction.corrected_author,
                crop_image_path=original.crop_image_path if original else "",
                ocr_confidence=original.confidence if original else 0.0,
                source="human_corrected",
            ))

        return records


class StubVectorStoreService(BaseVectorStoreService):
    """In-memory vector store – always returns a cache miss by default."""

    def __init__(self):
        self._store: dict[str, BookRecord] = {}

    def search(self, title: str, author: str) -> BookRecord | None:
        # Simple exact-match lookup for test predictability
        key = f"{title.lower()}|{author.lower()}"
        return self._store.get(key)

    def upsert(self, book: BookRecord) -> None:
        key = f"{book.title.lower()}|{book.author.lower()}"
        self._store[key] = book


class StubCatalogService(BaseCatalogService):
    """Returns books with stub genre codes without calling the LLM."""

    def generate_catalog(self, books: list[BookRecord]) -> list[BookRecord]:
        for book in books:
            book.genre = "Fiction"
            book.genre_code = "FIC"
            book.summary = f"A great book called {book.title}."
        return books

    def to_csv(self, books: list[BookRecord]) -> str:
        lines = ["title,author,genre,genre_code"]
        for b in books:
            lines.append(f"{b.title},{b.author},{b.genre},{b.genre_code}")
        return "\n".join(lines)


class StubRecommendationService(BaseRecommendationService):
    """Returns stub recommendations without calling the LLM."""

    def recommend(self, books, preferences) -> list:
        from app.models.book import BookRecommendation
        return [
            BookRecommendation(
                book_id=b.book_id,
                title=b.title,
                author=b.author,
                crop_image_path=b.crop_image_path,
                summary=f"You would enjoy {b.title}.",
                rank=i + 1,
                match_reason="Matches your preferences.",
            )
            for i, b in enumerate(books[:preferences.max_recommendations])
        ]


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI test app with all stubs injected
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def stub_review_queue() -> StubReviewQueueService:
    """Shared stub review queue – allows tests to inspect internal state."""
    return StubReviewQueueService()


@pytest.fixture(scope="session")
def app(settings, stub_review_queue):
    """
    Returns a FastAPI app instance with all service dependencies overridden
    by lightweight stubs. No ML models, no Redis, no Qdrant needed.
    """
    import os
    # Ensure upload dir and frontend dirs exist for static file serving
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs("frontend/static/css", exist_ok=True)
    os.makedirs("frontend/static/js", exist_ok=True)
    os.makedirs("frontend/templates", exist_ok=True)

    # Override get_settings so the app uses test settings
    get_settings.cache_clear()

    fastapi_app = create_app()

    # FastAPI's dependency_overrides dict maps dependency callables to
    # replacement callables. Called by FastAPI's DI system on each request.
    fastapi_app.dependency_overrides[get_settings] = lambda: settings
    fastapi_app.dependency_overrides[get_detection_service] = lambda: StubDetectionService()
    fastapi_app.dependency_overrides[get_ocr_service] = lambda: StubOCRService()
    fastapi_app.dependency_overrides[get_confidence_router] = lambda: StubConfidenceRouter()
    fastapi_app.dependency_overrides[get_review_queue_service] = lambda: stub_review_queue
    fastapi_app.dependency_overrides[get_result_merger] = lambda: StubResultMerger()
    fastapi_app.dependency_overrides[get_vector_store_service] = lambda: StubVectorStoreService()
    fastapi_app.dependency_overrides[get_catalog_service] = lambda: StubCatalogService()
    fastapi_app.dependency_overrides[get_recommendation_service] = lambda: StubRecommendationService()

    return fastapi_app


@pytest.fixture(scope="session")
def client(app) -> Generator:
    """Synchronous HTTP test client wrapping the FastAPI app."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# Sample data fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_image_bytes() -> bytes:
    """
    Returns a tiny valid PNG image as bytes.
    Used wherever an UploadFile is required in tests – avoids needing
    real image files on disk.
    """
    img = PILImage.new("RGB", (200, 400), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_ocr_result() -> OCRResult:
    """A high-confidence OCR result for use in unit tests."""
    return OCRResult(
        book_id=str(uuid.uuid4()),
        crop_image_path="test_crop.jpg",
        raw_title="The Great Gatsby",
        raw_author="F. Scott Fitzgerald",
        confidence=0.92,
        flagged_for_review=False,
    )


@pytest.fixture
def low_confidence_ocr_result() -> OCRResult:
    """A low-confidence OCR result (below the 0.75 threshold)."""
    return OCRResult(
        book_id=str(uuid.uuid4()),
        crop_image_path="test_crop_blurry.jpg",
        raw_title="S me Bk Titl",
        raw_author="Unkwn Authur",
        confidence=0.42,
        flagged_for_review=False,  # Router sets this; not the fixture
    )


@pytest.fixture
def sample_book_record() -> BookRecord:
    """A fully populated BookRecord for use in catalog / vector store tests."""
    return BookRecord(
        book_id=str(uuid.uuid4()),
        title="The Great Gatsby",
        author="F. Scott Fitzgerald",
        crop_image_path="test_crop.jpg",
        ocr_confidence=0.92,
        source="ocr_auto",
        genre="Fiction",
        genre_code="FIC",
    )
