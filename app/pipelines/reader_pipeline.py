# ─────────────────────────────────────────────────────────────────────────────
# app/pipelines/reader_pipeline.py
#
# Celery task for the Reader pipeline.
#
# FULL PIPELINE (end-to-end):
#   For each uploaded image:
#     1. Load image from disk → PIL Image
#     2. Detection service → bounding boxes + crop paths
#   For each crop:
#     3. Vector store search → cache hit? return stored BookRecord : continue
#     4. OCR service → OCRResult (title, author, confidence)
#   After all crops:
#     5. Confidence router → auto_accepted + flagged (no HITL for readers;
#        flagged books are still sent to recommendation but tagged low-confidence)
#     6. Result merger → list[BookRecord]  (all books, auto + flagged merged)
#     7. Recommendation service → ranked BookRecommendation list
#     8. Vector store upsert_batch → persist enriched records for future cache hits
#
#   NOTE: ON HITL FOR READERS:
#   Readers do not get a review UI — they are not curating a library, they are
#   choosing their next book. Low-confidence OCR results are still passed to the
#   LLM; the LLM is instructed (via prompt) to note garbled text and attempt
#   identification via web search. If it cannot identify a book, it simply
#   ranks it last. This gives readers a best-effort experience without blocking
#   on manual correction.
#
# SOLID (Single Responsibility):
#   This file only orchestrates. Each step delegates to a service class.
#   Replacing any service (e.g. a new OCR engine) requires zero changes here.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path
from PIL import Image as PILImage

from app.core.celery_app import celery_app

from app.core.exceptions import BookLensError
from app.core.logging import get_logger
from app.models.book import BookRecord, OCRValidatedResult, BookRecommendation
from app.models.request import ReaderPreferences
from app.models.response import ReaderResponse

logger = get_logger(__name__)


@celery_app.task(
    bind=True,
    name="pipelines.reader.run",
    max_retries=3,
    default_retry_delay=10,
    track_started=True,
)
def run_reader_pipeline(
    self,
    job_id: str,
    image_paths: list[str],
    preferences_dict: dict,
) -> dict:
    """
    Celery task: run the full reader recommendation pipeline.

    Args:
        job_id:           UUID for this job (used for logging and crop namespacing).
        image_paths:      Absolute paths to the saved uploaded shelf images.
        preferences_dict: Serialised ReaderPreferences dict (JSON-safe).

    Returns:
        Serialised ReaderResponse dict (stored in Celery result backend).
        The route handler deserialises this on the next poll.

    Celery handles retries on transient failures (e.g. Redis blip).
    BookLensError subclasses are NOT retried — they represent permanent failures
    (e.g. no books in image) that should surface immediately to the user.
    """
    logger.info("reader_pipeline_start", job_id=job_id, num_images=len(image_paths))
    
    
    # Re-hydrate preferences from the JSON-serialisable dict
    preferences = ReaderPreferences(**preferences_dict)

    # ── Instantiate services (singletons via lru_cache in dependencies) ────
    # We import and instantiate directly here rather than via FastAPI's DI
    # because Celery workers are separate processes with no FastAPI context.
    services = _build_services()

    all_books: list = []

    # ── Steps 1–4: Book detection → OCR on every Crop → Cache Check in Vector DB for every image ────
    all_books = _process_image(
            all_image_paths=image_paths,
            job_id=job_id,
            services=services,
            )

    # NOTE: all_books - contains either a OCRValidatedResult object or 
    #                   a BookRecord cached from Vector DB

    logger.info(
        "reader_pipeline_ocr_complete",
        job_id=job_id,
        total_books=len(all_books),
    )

    if not all_books:
        # Return a valid (empty) response rather than raising error — the user sees
        # "0 books detected" rather than an error page.
        return ReaderResponse(
            job_id=job_id,
            total_books_detected=0,
            recommendations=[],
            all_books=[],
        ).model_dump(mode="json")

    # ── Step 5 & 6: route by confidence + merge into BookRecords ──────────
    # For readers, low-confidence books are NOT held back for review — they
    # are tagged and passed to the LLM along with confident results.
    # auto_accepted, flagged = services["router"].route(all_books)

    # Merge: auto_accepted come first, then flagged (LLM de-prioritises these)
    # merged_records = services["merger"].merge(
    #     auto_accepted=auto_accepted,
    #     corrections=flagged,                             # No human corrections in reader flow
    #     flagged_originals=flagged,
    # )

    # NOTE: No meed for routing and merging tasks for Reader Workload!
    

    # ── Step 7: LLM recommendation ────────────────────────────────────────   
    enriched_books, recommendations = services["recommender"].recommend(all_books, preferences)

    # ── Step 8: Vector store write-back ───────────────────────────────────
    # enriched_books now have genre/year enriched
    try:
        services["vector_store"].upsert_batch(enriched_books)
        logger.info("reader_pipeline_vector_upsert_ok", job_id=job_id)
    except Exception as exc:
        # Non-fatal: log and continue — recommendations are still returned.
        logger.warning("reader_pipeline_vector_upsert_failed", error=str(exc))

    response = ReaderResponse(
        job_id=job_id,
        total_books_detected=len(enriched_books),
        recommendations=recommendations,                # List of BookRecommendation objects
        all_books=enriched_books,
    )

    logger.info(
        "reader_pipeline_complete",
        job_id=job_id,
        total_books=len(enriched_books),
        recommendations=len(recommendations),
    )

    return response.model_dump(mode="json")



# ── Private helpers ────────────────────────────────────────────────────────────

def _process_image(
    all_image_paths: str,
    job_id: str,
    services: dict,
) -> list[BookRecord]:
    """
    Run detection + optional vector cache lookup + OCR for a single shelf image.

    Returns a list of BookRecord objects (one per detected book spine).
    Books that are already in the vector store are returned from cache;
    others go through OCR.

    Args:
        image_path: Absolute path to the shelf image on disk.
        job_id:     Job identifier for crop file namespacing.
        services:   Dict of instantiated service objects.

    Returns:
        List of BookRecord objects for all books found in this image.
    """
    # Putting all the image through detection process and getting their crop paths:
    all_crop_paths: list[str] = []
        
    for img_idx, img_path in enumerate(all_image_paths):
        # Load the image from disk
        try:
            pil_image = PILImage.open(img_path).convert("RGB")
                        
        except Exception as exc:
            logger.error("reader_image_load_failed", path=img_path, error=str(exc))
            return []

        # ── Step 2: Detection ─────────────────────────────────────────────────
        try:
            _, crop_paths = services["detector"].detect(pil_image, job_id, img_idx)
            all_crop_paths.extend(crop_paths)
            
        except BookLensError as exc:
            logger.warning("reader_detection_failed", path=img_path, error=exc.message)
            return []
        

    books: list = []       # Every element will either be OCRValidatedResult if not cached or 
                           # BookRecord if cached from Vector DB.

    for idx, crop_path in enumerate(all_crop_paths):
        book_id = f"{job_id}_{idx}"

        # ── Step 3: Running OCR, validating OCR results with Google Books Records ──────────────────────
        # Run OCR first to get text on book spine
        # Then feed that text to Google Books API to get similar book records 
        # Then at last applying Fuzzy matching Logic to get best suitable book record from top results recieved via API
        # Then we'll use book title + author embedding and check our Vector DB for Cache hit
        
        # NOTE: We need text to query; there's no image-based vector search here.
        try:
            book_spine_text, ocr_conf_score = services["ocr"].extract(crop_path=crop_path, 
                                                                    book_id=book_id,
                                                                    )
        except BookLensError as exc:
            logger.warning(
                "reader_ocr_failed",
                crop_path=crop_path,
                error=exc.message,)
            continue
        
        try:
            ocr_result: OCRValidatedResult = services["ocr"].validate(book_spine_text,
                                                                    crop_path=crop_path, 
                                                                    book_id=book_id,
                                                                    )
        except BookLensError as exc:
            logger.warning(
                "reader_ocr_validation_failed",
                crop_path=crop_path,
                error=exc.message,)
            continue
        
        # Appending OCR confidence score in OCRValidatedResult object
        ocr_result.ocr_confidence = ocr_conf_score
        
        if ocr_result.title:
            # Query the vector store using the OCR-extracted text
            cached = services["vector_store"].search(         
                title=ocr_result.title,
                author=ocr_result.author,
            )

            if cached:   # cached will be a highly enriched BookRecord object - ready for final output 
                # Cache hit: use the stored (enriched) BookRecord
                # NOTE: Don't cache if BookRecord lacks description. 
                #       Get summary and update the one in vector store.
                logger.info(
                    "reader_cache_hit",
                    book_id=book_id,
                    title=cached.title,
                )
                # Update crop path to the freshly saved one (image may have moved)
                cached.crop_image_path = crop_path
                books.append(cached) 
            else:
                # Cache miss: 
                # append OCRValidatedResult to list → later we'll enrich it and make it a BookRecord
                books.append(ocr_result)

    return books
    
    

def _build_services() -> dict:
    """
    Instantiate all services needed by the reader pipeline.

    In a Celery worker there is no FastAPI DI container, so we build
    services directly here using the application settings.

    Returns:
        Dict mapping service name → service instance.
    """
    # Use the per-process cached services factory so models are loaded once
    # per worker process and reused across tasks.
    from app.core.worker_services import (get_worker_services_Det_OCR_Mer_Rou, 
                                          get_worker_services_REC,
                                          get_worker_services_Reader_VS,
                                          )

    services = get_worker_services_Det_OCR_Mer_Rou()
    services['recommender'] = get_worker_services_REC()['recommender']
    services['vector_store'] = get_worker_services_Reader_VS()['vector_store']

    return services
