# ─────────────────────────────────────────────────────────────────────────────
# app/models/book.py
#
# Pydantic models (data schemas) for everything related to a book.
#
# These models serve three purposes:
#   1. Runtime validation – Pydantic rejects malformed data at the boundary
#   2. Serialisation – FastAPI uses them to render JSON responses
#   3. Documentation – FastAPI auto-generates OpenAPI schemas from them
#
# SOLID note (Single Responsibility):
#   Each model represents exactly one concept. Shared fields are composed via
#   inheritance rather than copy-pasted across models.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, field_validator


# ── Bounding box ──────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """
    Pixel coordinates of a detected book spine in the original image.
    Origin (0, 0) is the top-left corner of the image.
    """
    x_min: int = Field(..., ge=0, description="Left edge of the bounding box")
    y_min: int = Field(..., ge=0, description="Top edge of the bounding box")
    x_max: int = Field(..., ge=0, description="Right edge of the bounding box")
    y_max: int = Field(..., ge=0, description="Bottom edge of the bounding box")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Detection confidence score from Faster-RCNN (0–1)"
    )

    @field_validator("x_max")
    @classmethod
    def x_max_must_exceed_x_min(cls, v: int, info) -> int:
        """Ensure the box has positive width."""
        if "x_min" in info.data and v <= info.data["x_min"]:
            raise ValueError("x_max must be greater than x_min")
        return v

    @field_validator("y_max")
    @classmethod
    def y_max_must_exceed_y_min(cls, v: int, info) -> int:
        """Ensure the box has positive height."""
        if "y_min" in info.data and v <= info.data["y_min"]:
            raise ValueError("y_max must be greater than y_min")
        return v


# ── Raw OCR result (per cropped book image) ────────────────────────────────────

class OCRResult(BaseModel):
    """
    Raw output from the OCR engine for a single cropped book spine image.

    The `flagged_for_review` field is set by the ConfidenceRouter based on
    whether `confidence` falls below the configured threshold. It is NOT set
    by the OCR engine itself – keeping that logic out of the OCR service
    honours the Single Responsibility Principle.
    """
    book_id: str = Field(
        ..., description="UUID linking this result to its crop image file"
    )
    crop_image_path: str = Field(
        ..., description="Relative path to the saved crop image (served as static)"
    )
    raw_title: str = Field(
        default="", description="Book title as read by OCR (may be empty or noisy)"
    )
    raw_author: str = Field(
        default="", description="Author name as read by OCR (may be empty or noisy)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Composite OCR confidence score (average of title + author scores)"
    )
    flagged_for_review: bool = Field(
        default=False,
        description="True if confidence < OCR_CONFIDENCE_THRESHOLD"
    )


class OCRValidatedResult(BaseModel):
    """
    OCR result Validated book information using books records from Google Books API.
    USED: Raw output from the OCR engine for a single cropped book spine image.

    The `flagged_for_review` field is set by the ConfidenceRouter based on
    whether `confidence` falls below the configured threshold. It is NOT set
    by the OCR engine itself – keeping that logic out of the OCR service
    honours the Single Responsibility Principle.
    """
    book_id: str = Field(
        ..., description="UUID linking this result to its crop image file"
    )
    crop_image_path: str = Field(
        ..., description="Relative path to the saved crop image (served as static)"
    )
    ori_ocr_ext_spine_txt: str = Field(
        ..., description="Original Spine Text read after complete OCR"
    )
    title: str = Field(
        default="", description="Book title validated from Google Book records"
    )
    subtitle: str = Field(
        default="", description="Book sub-title validated from Google Book records"
    )
    author: str = Field(
        default="", description="Author name validated from Google Book records"
    )
    description: str = Field(
        default="", description="Book description from Google Book records"
    )
    ocr_confidence: float = Field(
        default=0, ge=0.0, le=1.0,
        description="Composite OCR confidence score (average of title + author scores)"
    )
    flagged_for_review: bool = Field(
        default=False,
        description="True if confidence < OCR_CONFIDENCE_THRESHOLD"
    )
    about_book: dict = Field(
        default={}, description="Other relevant book information later to be added to BookRecord"
    )
    
    
    
# ── Human correction (submitted by librarian via review UI) ────────────────────

class BookCorrection(BaseModel):
    """
    A single librarian correction for one flagged book.
    Submitted via POST /librarian/review/{job_id}.
    """
    book_id: str = Field(..., description="Must match an OCRResult.book_id")
    corrected_title: str = Field(..., min_length=1)
    corrected_author: str = Field(default="", description="Optional; empty if unknown")


# ── Verified book (final merged result after correction stage) ─────────────────

class BookRecord(BaseModel):
    """
    The canonical book record that flows into the catalog generator and
    vector database. Produced by the ResultMerger from either:
      - An auto-accepted OCRResult (source = "ocr_auto"), or
      - A human-corrected BookCorrection (source = "human_corrected").

    The `source` field creates an audit trail and allows the vector DB
    to weight human-verified records higher in similarity searches.
    """
    book_id: str
    title: str
    subtitle: str | None = None
    author: str
    crop_image_path: str
    ocr_confidence: float
    source: Literal["ocr_auto", "human_corrected"] = "ocr_auto"
    ori_ocr_ext_spine_txt: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # ── Optional fields populated after LLM enrichment ────────────────────────
    genre: str | None = None
    genre_code: str | None = None        # e.g. "FIC", "SCI", "HIS"
    pagecount: int | None = None
    maturityRating: str | None = None
    ratings: float = Field(default=1, ge=1, le=5)
    description: str = ""
    summary: str = ""
    isbn: str | None = None
    publisher: str | None = None
    year_published: str | None = None



# ── Reader-facing recommendation ─────────────────────────────────────────────

class BookRecommendation(BaseModel):
    """
    A single book recommendation returned to a reader, including the LLM's
    short summary and ranking rationale.
    """
    book_id: str
    title: str
    subtitle: str = ""
    author: str
    genre: str = ""
    crop_image_path: str
    summary: str = Field(..., description="2-3 sentence summary from the LLM")
    rank: int = Field(..., ge=1, description="1 = best match for user preferences")
    match_reason: str = Field(
        default="", description="One sentence: why this book suits the user's filters"
    )
