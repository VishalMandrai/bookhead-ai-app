# ─────────────────────────────────────────────────────────────────────────────
# app/models/request.py
#
# Pydantic models for incoming API request bodies.
# File upload fields are handled by FastAPI's UploadFile (not Pydantic),
# so these models cover the non-file parameters sent alongside uploads.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


# ── Reader preferences (sent with the shelf image upload) ─────────────────────

class ReaderPreferences(BaseModel):
    """
    Filters and preferences that the LLM uses to rank and summarise books
    for the reader. All fields are optional – the LLM handles missing values
    gracefully by returning a general recommendation.
    """
    preferred_genres: list[str] = Field(
        default_factory=list,
        description="e.g. ['science fiction', 'biography']",
        max_length=10,
    )
    mood: str | None = Field(
        default=None,
        description="Free-text mood descriptor, e.g. 'something uplifting' or 'dark thriller'",
        max_length=200,
    )
    preferred_length: Literal["short", "medium", "long", "any"] = "any"
    exclude_genres: list[str] = Field(
        default_factory=list,
        description="Genres to exclude from recommendations",
        max_length=10,
    )
    max_recommendations: int = Field(
        default=5, ge=1, le=20,
        description="How many ranked recommendations to return",
    )


# ── Librarian correction submission ───────────────────────────────────────────

class ReviewSubmission(BaseModel):
    """
    Payload for POST /librarian/review/{job_id}.
    The librarian submits corrections for ALL flagged books in one shot.
    Partial submissions are not accepted – the UI must confirm every flagged
    item (even if the librarian just confirms the OCR text was correct).
    """
    corrections: list[dict] = Field(
        ...,
        description=(
            "List of {book_id, corrected_title, corrected_author} objects. "
            "Must include an entry for every book_id that was flagged."
        ),
        min_length=1,
    )
