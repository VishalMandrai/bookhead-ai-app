# ─────────────────────────────────────────────────────────────────────────────
# app/models/response.py
#
# Pydantic models for all API responses.
# Having explicit response models (rather than returning raw dicts) ensures:
#   - FastAPI validates and serialises output – no accidental data leaks
#   - OpenAPI docs show clients exactly what shape to expect
#   - Type checkers catch mismatches between service return types and routes
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

from app.models.book import BookRecord, BookRecommendation, OCRResult


# ── Generic job response (returned immediately after upload) ──────────────────

class JobAcceptedResponse(BaseModel):
    """
    Returned with HTTP 202 when a pipeline job has been queued.
    The client uses `job_id` to poll for status and results.
    """
    job_id: str
    status: Literal["queued"] = "queued"
    message: str = "Job accepted and queued for processing."


# ── Job status poll response ──────────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    """
    Returned by GET /jobs/{job_id}.
    `state` mirrors Celery task states: PENDING → STARTED → SUCCESS | FAILURE.
    An extra "AWAITING_REVIEW" state is injected by the librarian pipeline
    when flagged books are waiting for human correction.
    """
    job_id: str
    state: Literal["PENDING", "STARTED", "AWAITING_REVIEW", "SUCCESS", "FAILURE"]
    progress_message: str | None = None    # Human-readable progress hint
    result: Any | None = None              # Populated only when state = SUCCESS
    error: str | None = None               # Populated only when state = FAILURE


# ── Librarian: review queue response ──────────────────────────────────────────

class ReviewItem(BaseModel):
    """
    A single book presented to the librarian for correction.
    Includes both the crop image URL and the OCR's best-guess text.
    """
    book_id: str
    crop_image_url: str = Field(
        ..., description="URL path to the crop image served as static asset"
    )
    ocr_title: str = Field(..., description="What the OCR read – pre-filled in the UI")
    ocr_author: str = Field(..., description="What the OCR read – pre-filled in the UI")
    confidence: float = Field(..., description="OCR confidence score (0–1)")


class ReviewQueueResponse(BaseModel):
    """
    Returned by GET /librarian/review/{job_id}.
    Lists all books requiring human review for the given job.
    """
    job_id: str
    total_flagged: int
    items: list[ReviewItem]
    instructions: str = (
        "Please verify or correct the title and author for each book below, "
        "then submit all corrections together."
    )


# ── Librarian: catalog response ───────────────────────────────────────────────

class CatalogResponse(BaseModel):
    """
    Final catalog output returned by GET /librarian/catalog/{job_id}
    once all corrections have been submitted and the pipeline completes.
    """
    job_id: str
    total_books: int
    auto_accepted_count: int    # Books whose OCR was confident enough
    human_corrected_count: int  # Books that went through review
    books: list[BookRecord]
    csv_download_url: str = Field(
        ..., description="URL to download the catalog as a CSV file"
    )


# ── Reader: recommendation response ──────────────────────────────────────────

class ReaderResponse(BaseModel):
    """
    Final response returned to a reader after the full pipeline completes.
    """
    job_id: str
    total_books_detected: int
    recommendations: list[BookRecommendation]
    all_books: list[BookRecord] = Field(
        default_factory=list,
        description="All detected books, not just the top recommendations",
    )


# ── Error response ────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error envelope returned for all 4xx / 5xx responses."""
    error: str          # Machine-readable error code, e.g. "job_not_found"
    message: str        # Human-readable explanation
    detail: dict = Field(default_factory=dict)  # Optional structured context
