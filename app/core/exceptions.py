# ─────────────────────────────────────────────────────────────────────────────
# app/core/exceptions.py
#
# Custom exception hierarchy for BookLens AI.
#
# Why custom exceptions instead of raw Python built-ins?
#   - They carry semantic meaning (BookLensError vs ValueError)
#   - The FastAPI exception handlers (in main.py) can catch them by type
#     and return the correct HTTP status codes automatically
#   - Each exception can carry structured metadata (job_id, book_id, etc.)
#     that gets serialised into the error response body
#
# SOLID note (Open/Closed):
#   Adding new error types never modifies existing handlers – just subclass
#   BookLensError and register a new handler if needed.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations


class BookLensError(Exception):
    """
    Base class for all BookLens application errors.
    Carries a human-readable message and an optional detail dict for
    structured logging / API responses.
    """
    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


# ── File handling errors ──────────────────────────────────────────────────────

class FileTooLargeError(BookLensError):
    """Raised when an uploaded file exceeds MAX_UPLOAD_SIZE_BYTES."""


class UnsupportedFileTypeError(BookLensError):
    """Raised when an uploaded file is not a supported image format."""


# ── Detection errors ──────────────────────────────────────────────────────────

class DetectionError(BookLensError):
    """Raised when the Faster-RCNN model fails to process an image."""


class NoBooksDetectedError(BookLensError):
    """Raised when the detector finds zero books in the input image."""


# ── OCR errors ────────────────────────────────────────────────────────────────

class OCRError(BookLensError):
    """Raised when the OCR engine throws an unexpected exception."""


# ── Pipeline / job errors ─────────────────────────────────────────────────────

class JobNotFoundError(BookLensError):
    """Raised when a job_id does not exist in the result backend."""


class JobNotReadyError(BookLensError):
    """Raised when a job exists but has not yet completed processing."""


class InvalidJobStateError(BookLensError):
    """Raised when an operation is attempted on a job in an invalid state."""


# ── Vector store errors ───────────────────────────────────────────────────────

class VectorStoreError(BookLensError):
    """Raised when Qdrant operations (upsert, search) fail."""


# ── LLM errors ───────────────────────────────────────────────────────────────

class LLMError(BookLensError):
    """Raised when the Anthropic API call fails or returns unexpected content."""


# ── Review queue errors ────────────────────────────────────────────────────────

class ReviewSessionNotFoundError(BookLensError):
    """Raised when a review session ID does not exist."""


class ReviewAlreadyCompleteError(BookLensError):
    """Raised when corrections are submitted for an already-completed session."""
