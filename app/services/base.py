# ─────────────────────────────────────────────────────────────────────────────
# app/services/base.py
#
# Abstract Base Classes (ABCs) for every service in the application.
#
# WHY ABCs?
#   SOLID – Dependency Inversion Principle:
#     High-level modules (pipelines, routes) depend on THESE abstractions,
#     never on concrete implementations. This means:
#       - Swapping EasyOCR for a cloud OCR API → only OCRService changes.
#       - Swapping Qdrant for Pinecone → only VectorStoreService changes.
#       - Pipelines, routes, and tests are completely unaffected.
#
#   SOLID – Open/Closed Principle:
#     New implementations (e.g. a GPU-accelerated detector) can be added
#     WITHOUT modifying any existing code – just subclass and register.
#
#   Testability:
#     Tests can inject lightweight stub implementations of these ABCs without
#     spinning up real models or databases.
#
# CONVENTION: Every concrete service class must inherit from one of these ABCs
# and implement ALL abstract methods. Python will raise TypeError at import
# time if any abstract method is missing – catching mistakes early.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL.Image import Image as PILImage

from app.models.book import (
    BoundingBox,
    BookCorrection,
    BookRecord,
    OCRResult,
    BookRecommendation,
)
from app.models.request import ReaderPreferences


# ── Detection ─────────────────────────────────────────────────────────────────

class BaseDetectionService(ABC):
    """
    Contract for book spine detection.

    Implementations receive a PIL image and return one BoundingBox per
    detected book spine along with a list of cropped PIL images.
    """

    @abstractmethod
    def detect(
        self,
        image: PILImage,
        job_id: str,
    ) -> tuple[list[BoundingBox], list[str]]:
        """
        Detect book spines in `image`.

        Args:
            image:   The full shelf image as a PIL Image object.
            job_id:  Used to namespace the saved crop file paths.

        Returns:
            A tuple of:
              - List of BoundingBox objects (one per detected book).
              - List of file paths to the saved crop images (same order).

        Raises:
            DetectionError:       If model inference fails.
            NoBooksDetectedError: If the model finds no books at all.
        """
        ...


# ── OCR ───────────────────────────────────────────────────────────────────────

class BaseOCRService(ABC):
    """
    Contract for extracting book title and author text from a crop image.
    """

    @abstractmethod
    def extract(self, crop_path: str, book_id: str) -> OCRResult:
        """
        Run OCR on the cropped book spine image at `crop_path`.

        Args:
            crop_path: File path to the cropped image.
            book_id:   UUID that ties this result back to the detection stage.

        Returns:
            OCRResult with raw_title, raw_author, and a composite confidence score.
            Does NOT set flagged_for_review – that is the ConfidenceRouter's job.

        Raises:
            OCRError: If the OCR engine throws an unexpected exception.
        """
        ...


# ── Confidence routing ────────────────────────────────────────────────────────

class BaseConfidenceRouter(ABC):
    """
    Contract for splitting OCR results into auto-accepted vs flagged buckets.

    Keeping this as a separate service (rather than inlining the threshold
    check) makes the routing logic independently testable and replaceable
    (e.g. a future ML-based router that considers book spine orientation).
    """

    @abstractmethod
    def route(
        self,
        ocr_results: list[OCRResult],
    ) -> tuple[list[OCRResult], list[OCRResult]]:
        """
        Partition `ocr_results` by confidence.

        Returns:
            A tuple of (auto_accepted, flagged_for_review).
            Both lists preserve the original order of `ocr_results`.
        """
        ...


# ── Review queue ──────────────────────────────────────────────────────────────

class BaseReviewQueueService(ABC):
    """
    Contract for managing the human-in-the-loop correction session.

    The review queue bridges the async gap between:
      - The moment the pipeline flags low-confidence books, and
      - The moment the librarian submits their corrections.
    """

    @abstractmethod
    def create_session(
        self,
        job_id: str,
        flagged_books: list[OCRResult],
    ) -> None:
        """Persist the flagged books for the given job_id in the review store."""
        ...

    @abstractmethod
    def get_pending(self, job_id: str) -> list[OCRResult]:
        """Return all unresolved flagged books for a job. Raises ReviewSessionNotFoundError."""
        ...

    @abstractmethod
    def submit_corrections(
        self,
        job_id: str,
        corrections: list[BookCorrection],
    ) -> None:
        """
        Persist librarian corrections. Marks the session as complete.
        Raises ReviewSessionNotFoundError or ReviewAlreadyCompleteError.
        """
        ...

    @abstractmethod
    def get_corrections(self, job_id: str) -> list[BookCorrection]:
        """Return submitted corrections. Raises ReviewSessionNotFoundError."""
        ...

    @abstractmethod
    def is_complete(self, job_id: str) -> bool:
        """Return True if all corrections have been submitted for this job."""
        ...


# ── Result merger ─────────────────────────────────────────────────────────────

class BaseResultMerger(ABC):
    """
    Contract for merging auto-accepted OCR results with human corrections
    into a unified list of BookRecord objects.
    """

    @abstractmethod
    def merge(
        self,
        auto_accepted: list[OCRResult],
        corrections: list[BookCorrection],
        flagged_originals: list[OCRResult],
    ) -> list[BookRecord]:
        """
        Combine auto-accepted books and human-corrected books into BookRecords.

        Args:
            auto_accepted:      OCR results above the confidence threshold.
            corrections:        Librarian-supplied corrections.
            flagged_originals:  The original OCR results that were flagged
                                (needed for crop_image_path and confidence).

        Returns:
            Merged list of BookRecord objects, preserving original order.
        """
        ...


# ── Vector store ──────────────────────────────────────────────────────────────

class BaseVectorStoreService(ABC):
    """
    Contract for storing and retrieving book metadata via vector similarity.
    """

    @abstractmethod
    def search(self, title: str, author: str) -> BookRecord | None:
        """
        Search for an existing book record by semantic similarity.

        Returns:
            A BookRecord if a close match is found above the similarity
            threshold, otherwise None (cache miss → proceed to LLM).
        """
        ...

    @abstractmethod
    def upsert(self, book: BookRecord) -> None:
        """
        Insert or update a BookRecord in the vector store.
        Human-verified records should be tagged with `verified=True`
        in the payload metadata.
        """
        ...


# ── Catalog generation ────────────────────────────────────────────────────────

class BaseCatalogService(ABC):
    """
    Contract for generating a structured library catalog from a list of books.
    """

    @abstractmethod
    def generate_catalog(self, books: list[BookRecord]) -> list[BookRecord]:
        """
        Enrich books with genre codes and return the final catalog list.

        The LLM is called here to genre-code each book. Results are returned
        as an updated list of BookRecord objects with `genre` and `genre_code`
        populated.
        """
        ...

    @abstractmethod
    def to_csv(self, books: list[BookRecord]) -> str:
        """Serialise the catalog to a CSV string for download."""
        ...


# ── Recommendation ────────────────────────────────────────────────────────────

class BaseRecommendationService(ABC):
    """
    Contract for ranking and summarising books for a reader.
    """

    @abstractmethod
    def recommend(
        self,
        books: list[BookRecord],
        preferences: ReaderPreferences,
    ) -> list[BookRecommendation]:
        """
        Use the LLM (with web search) to rank `books` by suitability for the
        given `preferences` and return a ranked list with summaries.
        """
        ...
