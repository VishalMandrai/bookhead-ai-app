# ─────────────────────────────────────────────────────────────────────────────
# app/api/routes/librarian.py
#
# FastAPI router for the Librarian flow.
#
# Endpoints:
#   POST /librarian/catalog              – Upload shelf images → queues job
#   GET  /librarian/catalog/{job_id}     – Poll job state / final catalog
#   GET  /librarian/review/{job_id}      – Fetch flagged books for human correction
#   POST /librarian/review/{job_id}      – Submit librarian corrections → resume pipeline 
#   GET  /librarian/download/{job_id}    – Download final catalog as CSV

#
# The HITL review flow:
#   1. Client uploads images → job is queued → 202 returned.
#   2. Pipeline runs detection + OCR, flags low-confidence books.
#   3. Job state transitions to "AWAITING_REVIEW".
#   4. Client polls GET /librarian/catalog/{job_id} → sees AWAITING_REVIEW.
#   5. Client fetches GET /librarian/review/{job_id} → gets flagged books + crops.
#   6. Librarian reviews the crop images and corrects text in the UI.
#   7. Client submits POST /librarian/review/{job_id} with all corrections.
#   8. Pipeline resumes → generates final catalog → job state → SUCCESS.
# ─────────────────────────────────────────────────────────────────────────────

import uuid
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse

# from app.api.dependencies import get_review_queue_service
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.book import BookCorrection, BookRecord, OCRValidatedResult
from app.models.request import ReviewSubmission
from app.models.response import (
    CatalogResponse,
    JobAcceptedResponse,
    JobStatusResponse,
    ReviewQueueResponse,
    ReviewItem,
)
from app.services.image_saver import save_uploads
from app.services.job_store import JobStore
from app.services.review_queue import RedisReviewQueueService
from app.services.catalog import PlainCatalogService


logger = get_logger(__name__)

router = APIRouter(prefix="/librarian", tags=["Librarian"])


@router.post(
    "/catalog",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload shelf images and queue a catalog generation job",
)
async def create_catalog(
    images: list[UploadFile] = File(..., description="Bookshelf images to catalog"),
    settings: Settings = Depends(get_settings),
) -> JobAcceptedResponse:
    """
    Save uploaded shelf images to disk, then dispatch the librarian Celery
    pipeline. Returns job_id immediately (HTTP 202).

    Poll GET /librarian/catalog/{job_id} to track state.
    If state == AWAITING_REVIEW, fetch GET /librarian/review/{job_id} next.
    """
    
    for image in images:
        _validate_upload(image, settings)

    job_id = str(uuid.uuid4())
        
    saved_paths = await save_uploads(images, settings.upload_dir, job_id)

    logger.info("librarian_job_dispatching", job_id=job_id, num_images=len(saved_paths))

    from app.pipelines.librarian_pipeline import run_librarian_pipeline
    run_librarian_pipeline.apply_async(
        args=[job_id, saved_paths],
        task_id=job_id,
    )

    return JobAcceptedResponse(job_id=job_id)


# ----------------------------------------------------------------------------------------------


@router.get(
    "/catalog/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll catalog job state",
    description=(
        "States: PENDING → STARTED → AWAITING_REVIEW → SUCCESS | FAILURE.\n"
        "When AWAITING_REVIEW, fetch GET /librarian/review/{job_id} to get "
        "the books that need manual correction."
    ),
)
async def get_catalog_job(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> JobStatusResponse:
    """
    Returns the job's current state.

    States:
      PENDING          → Still queued, not yet started.
      STARTED          → Detection + OCR running.
      AWAITING_REVIEW  → Low-confidence books found; corrections needed.
      SUCCESS          → Catalog ready; full CatalogResponse in `result`.
      FAILURE          → Pipeline error; `error` field populated.
    
    Checks our custom AWAITING_REVIEW Redis key first, then falls back to
    Celery's result backend for all other states.
    """
    return JobStore(redis_url=settings.redis_url).get_status(job_id)


# ----------------------------------------------------------------------------------------------


@router.get(
    "/review/{job_id}",
    response_model=ReviewQueueResponse,
    summary="Fetch all books pending librarian review for a job",
    description=(
        "Returns each flagged book's crop image URL and OCR-extracted text. "
        "The UI should display the image alongside editable text fields "
        "pre-filled with the OCR text."
    ),
)
async def get_review_queue(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> ReviewQueueResponse:
    """
    Returns each flagged book's crop image URL and OCR-extracted text.
    The review UI displays the crop image alongside editable text fields
    pre-filled with the OCR text.
    """
    svc = RedisReviewQueueService(
        redis_url=settings.redis_url,
        ttl_seconds=settings.review_session_ttl_seconds,
    )
    # Raises ReviewSessionNotFoundError (→ 404) if session doesn't exist
    flagged: OCRValidatedResult = svc.get_pending(job_id)

    items = [
        ReviewItem(
            book_id=ocr.book_id,
            crop_image_url="/uploads/" + ocr.crop_image_path,
            ocr_title=ocr.title,
            ocr_author=ocr.author,
            confidence=ocr.ocr_confidence,
        )
        for ocr in flagged
    ]

    logger.info("review_queue_fetched", job_id=job_id, count=len(items))

    return ReviewQueueResponse(
        job_id=job_id,
        total_flagged=len(items),
        items=items,
    )
       
    
# ----------------------------------------------------------------------------------------------

@router.post(
    "/review/{job_id}",
    status_code=status.HTTP_200_OK,
    summary="Submit librarian corrections for all flagged books and resume the pipeline",
    description=(
        "All flagged book_ids must be included in the submission. "
        "Partial submissions are rejected. After submission, the pipeline "
        "resumes and the catalog is generated."
    ),
    response_model=JobStatusResponse,
)
async def submit_corrections(
    job_id: str,
    submission: ReviewSubmission,
    settings: Settings = Depends(get_settings),
) -> JobStatusResponse:
    """
    Validates that every flagged book has a correction, persists them, then
    dispatches the 'resume_after_review' Celery task to finish the catalog.
    """
    svc = RedisReviewQueueService(
        redis_url=settings.redis_url,
        ttl_seconds=settings.review_session_ttl_seconds,
    )

    # Parse corrections
    corrections = [
        BookCorrection(
            book_id=item["book_id"],
            corrected_title=item["corrected_title"],
            corrected_author=item.get("corrected_author", ""),
        )
        for item in submission.corrections
    ]

    # Validate completeness — every flagged book must have a correction
    flagged = svc.get_pending(job_id)
    flagged_ids = {ocr.book_id for ocr in flagged}
    submitted_ids = {c.book_id for c in corrections}
    missing = flagged_ids - submitted_ids

    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "incomplete_submission",
                "message": "Corrections missing for some flagged books.",
                "missing_book_ids": list(missing),
            },
        )

    # Persist corrections and mark session complete
    svc.submit_corrections(job_id, corrections)

    logger.info("corrections_submitted", job_id=job_id, count=len(corrections))

    # Dispatch the resume task — uses a new task ID derived from the job_id
    from app.pipelines.librarian_pipeline import resume_after_review
    resume_task_id = f"{job_id}_resume"
    resume_after_review.apply_async(
        args=[job_id],
        task_id=resume_task_id,
    )

    # Update the job store so the poll endpoint reflects the new task ID
    job_store = JobStore(redis_url=settings.redis_url)
    job_store.clear_awaiting_review(job_id)            # Deleting Redis Key set with await_review value
    job_store.set_progress(job_id, "Corrections received — generating catalog…")

    return JobStatusResponse(
        job_id=job_id,
        state="STARTED",
        progress_message="Corrections received. Generating catalog…",
    )


# ----------------------------------------------------------------------------------------------


@router.get(
    "/download/{job_id}",
    response_class=PlainTextResponse,
    summary="Download the completed catalog as a CSV file",
)
async def download_catalog_csv(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> PlainTextResponse:
    """
    Fetches the completed catalog from the Celery result backend and
    streams it as a downloadable CSV file.

    Returns 404 if the job is not found, 202 if still processing.
    """
    job_store = JobStore(redis_url=settings.redis_url)
    status_resp = job_store.get_status(f"{job_id}_resume") 
                                                        # Now, since the Redis key with await_review value
                                                        # is deleted. We'll look for celery task status
                                                        # check logic. If SUCCESS, fetch the task result 
                                                        # & create a CSV File and return it.

    if status_resp.state != "SUCCESS":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND
            if status_resp.state == "FAILURE"
            else status.HTTP_202_ACCEPTED,
            detail=f"Catalog not ready yet. Current state: {status_resp.state}",
        )

    result = status_resp.result
    if not result or "books" not in result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Catalog result not available.",
        )

    # Reconstruct BookRecord objects from the serialised result
    books = [BookRecord(**b) for b in result["books"]]

    # Generate CSV using LLMCatalogService.to_csv() (no LLM call — pure export)
    catalog_svc = PlainCatalogService(
        emb_model=None,
        genre_emb_file_path=None,
    )
    csv_content = catalog_svc.to_csv(books)

    logger.info("catalog_csv_downloaded", job_id=job_id, num_books=len(books))

    return PlainTextResponse(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="catalog_{job_id[:8]}.csv"'
        },
    )



# ── Shared helpers ────────────────────────────────────────────────────────────

def _validate_upload(image: UploadFile, settings: Settings) -> None:
    """Validate a single uploaded file for type and size."""
    ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{image.content_type}'. Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    if image.size and image.size > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File '{image.filename}' exceeds the {settings.max_upload_size_bytes // (1024*1024)} MB limit.",
        )

