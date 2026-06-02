# ─────────────────────────────────────────────────────────────────────────────
# app/api/routes/reader.py
#
# FastAPI router for the Reader flow.
#
# Endpoints:
#   POST /reader/analyze  – Upload shelf image(s) + preferences → queues job
#   GET  /reader/{job_id} – Poll for recommendation results
#
# SOLID note (Single Responsibility):
#   Routes only handle HTTP concerns: parsing requests, validating uploads,
#   dispatching to Celery, and formatting responses. All business logic lives
#   in the pipeline and services layers.
# ─────────────────────────────────────────────────────────────────────────────

import uuid
import os
from pydantic import BaseModel
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

# from app.api.dependencies import get_detection_service, get_ocr_service 
from app.core.config import Settings, get_settings
from app.core.exceptions import FileTooLargeError, JobNotFoundError, JobNotReadyError
from app.core.logging import get_logger
from app.models.request import ReaderPreferences

from app.services.image_saver import save_uploads
from app.services.job_store import JobStore

from app.models.response import (
    ErrorResponse,
    JobAcceptedResponse,
    JobStatusResponse,
    ReaderResponse,
)

logger = get_logger(__name__)

# All routes in this file will be mounted under the `/reader` prefix
router = APIRouter(prefix="/reader", tags=["Reader"])


@router.post(
    "/analyze",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload shelf images and queue a recommendation job",
    description=(
        "Accepts one or more bookshelf images and optional reading preferences. "
        "Returns a job_id immediately. Poll GET /reader/{job_id} for results."
    ),
)
async def analyze_shelf(
    # FastAPI handles multipart form data: files + form fields together
    images: list[UploadFile] = File(..., description="One or more shelf images"),
    preferred_genres: str = Form(default=""),   # Comma-separated; parsed below
    mood: str = Form(default=""),
    preferred_length: str = Form(default="any"),
    max_recommendations: int = Form(default=3),
    settings: Settings = Depends(get_settings),
) -> JobAcceptedResponse:
    """
    Phase 1 stub: validates the upload and returns a job_id.
    Actual Celery dispatch will be wired in Phase 7.
    """
    # ── Validate each uploaded file ────────────────────────────────────────
    for image in images:
        _validate_upload(image, settings)

    # ── Parse preferences from flat form fields into the Pydantic model ────
    preferences = ReaderPreferences(
        preferred_genres=[g.strip() for g in preferred_genres.split(",") if g.strip()],
        mood=mood or None,
        preferred_length=preferred_length,  # type: ignore[arg-type]
        max_recommendations=max_recommendations,
    )

    # ── Generate a unique job ID ───────────────────────────────────────────
    job_id = str(uuid.uuid4())   # In production, consider a more robust ID scheme (e.g. ULID) for better sorting and uniqueness guarantees

    logger.info(
        "reader_job_queued",
        job_id=job_id,
        num_images=len(images),
        preferences=preferences.model_dump(),
    )

    # Persist to disk — Celery workers cannot share in-memory UploadFile objects
    saved_paths = await save_uploads(images, settings.upload_dir, job_id)

    logger.info("reader_job_dispatching", job_id=job_id, num_images=len(saved_paths))

    # task_id=job_id lets AsyncResult(job_id) find this task on poll
    from app.pipelines.reader_pipeline import run_reader_pipeline
    run_reader_pipeline.apply_async(
        args=[job_id, saved_paths, preferences.model_dump(mode="json")],
        task_id=job_id,
    )
    
    return JobAcceptedResponse(job_id=job_id)



@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll for reader job status and results",
)
async def get_reader_job(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> JobStatusResponse:
    """
    Poll Celery's result backend.
    On SUCCESS, `result` contains a serialised ReaderResponse with ranked
    recommendations and all detected books.
    """
    
    return JobStore(redis_url=settings.redis_url).get_status(job_id)
    


# ── Shared helpers ────────────────────────────────────────────────────────────

def _validate_upload(image: UploadFile, settings: Settings) -> None:
    """
    Validate a single uploaded file:
      - Must be a supported image MIME type
      - Must not exceed the configured size limit

    Raises HTTPException with 400 on validation failure.
    """
    ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{image.content_type}'. "
                   f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    # UploadFile.size is available after FastAPI reads the multipart body
    if image.size and image.size > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File '{image.filename}' is too large. "
                   f"Maximum allowed size is {settings.max_upload_size_bytes // (1024*1024)} MB.",
        )


# ────────────────────────────────────────────────────────────────────────────────────────────────────
# ───────────────────────── Endpoint for loading Portfolio Web Link from Env Var ─────────────────────

# /reader/portlink Endpoint — BookHead AI

# A GET /reader/portlink endpoint that returns portfolio web link
# value sourced from environment variables.

# The frontend (about.js) fetches this endpoint once to wire up any links or
# settings that can't be baked into the static HTML at build time.

class FrontendConfig(BaseModel):
    portfolio_url: str = ""


@router.get("/config/portlink", 
            response_model=FrontendConfig)
async def get_config() -> FrontendConfig:
    """
    Return portfolio web link sourced from environment variables.

    Environment variables read:
        PORTFOLIO_URL   — Full URL to the developer's portfolio website.
                          Example: https://yourname.dev
                          Falls back to "" (empty string) if not set.
    """
    print("Inside endpoint")
    return FrontendConfig(
        portfolio_url=os.getenv("PORTFOLIO_URL", ""),
    )
    
# ────────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────────
