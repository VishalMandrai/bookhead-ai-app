# ─────────────────────────────────────────────────────────────────────────────
# app/pipelines/librarian_pipeline.py
#
# Celery tasks for the Librarian catalog pipeline.
#
# TWO TASKS to support the HITL review pause:
#
#   run_librarian_pipeline(job_id, image_paths)
#     ├── detect + OCR all images
#     ├── route by confidence → auto_accepted + flagged
#     ├── if flagged:
#     │     ├── store in review queue (Redis)
#     │     ├── set job state = AWAITING_REVIEW
#     │     └── STOP — wait for librarian input
#     └── if no flagged books:
#           └── call _run_catalog_stage(job_id, auto_accepted, [])
#
#   resume_after_review(job_id)
#     ├── fetch corrections from review queue
#     ├── merge auto_accepted + corrections → BookRecord list
#     └── call _run_catalog_stage(job_id, auto_accepted, corrections)
#
#   _run_catalog_stage(job_id, auto_accepted, corrections)  [-> shared helper]
#     ├── merge results via DefaultResultMerger
#     ├── vector store lookup per book (skip LLM for cache hits)
#     ├── LLM catalog enrichment for cache misses
#     ├── vector store upsert_batch
#     └── return CatalogResponse
#
# WHY TWO TASKS?
#   Celery tasks cannot be paused mid-execution. The HITL review requires a
#   gap of unknown duration between OCR completion and catalog generation.
#   Splitting into two tasks lets each complete independently:
#     Task 1 ends with AWAITING_REVIEW state stored in Redis.
#     Task 2 starts when the route handler receives the librarian's corrections.
#
# SOLID (Single Responsibility):
#   Each task and helper owns exactly one pipeline stage.
#   Replacing the catalog service, OCR service, or vector store requires
#   zero changes to task scheduling logic.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path
from PIL import Image as PILImage

from app.core.celery_app import celery_app
from app.core.exceptions import BookLensError
from app.core.logging import get_logger
from app.models.book import BookCorrection, BookRecord, OCRResult, OCRValidatedResult
from app.models.response import CatalogResponse

logger = get_logger(__name__)


# ── Task 1: Initial pipeline (detect → OCR → route → pause or complete) ────────

@celery_app.task(
    bind=True,
    name="pipelines.librarian.run",
    max_retries=3,
    default_retry_delay=10,
    track_started=True,
)
def run_librarian_pipeline(
    self,
    job_id: str,
    image_paths: list[str],
) -> dict:
    """
    Celery task: detect books, run OCR, and either pause for review or
    proceed directly to catalog generation.

    Args:
        job_id:       UUID for this job.
        image_paths:  Absolute paths to the saved uploaded shelf images.

    Returns:
        Serialised CatalogResponse if no books needed review, OR
        a minimal status dict with state="AWAITING_REVIEW" if flagged
        books exist (actual result stored after resume_after_review completes).
    """
    logger.info("librarian_pipeline_start", job_id=job_id, num_images=len(image_paths))
    services = _build_services()

    # ── Steps 1–4: detect + OCR for all images ────────────────────────────
    all_ocr_results: list = []

    all_ocr_results = _detect_and_ocr_image(all_image_paths=image_paths,
                                            job_id=job_id,
                                            services=services,
                                            )
    # NOTE: all_books - contains either a OCRValidatedResult object or 
    #                   a BookRecord cached from Vector DB
    
    logger.info(
        "librarian_ocr_&_ocr_validation_complete",
        job_id=job_id,
        total_books=len(all_ocr_results),
    )

    if not all_ocr_results:
        return CatalogResponse(
            job_id=job_id,
            total_books=0,
            auto_accepted_count=0,
            human_corrected_count=0,
            books=[],
            csv_download_url=f"/downloads/{job_id}/catalog.csv",
        ).model_dump(mode="json")

    # ── Step 5: Confidence routing ────────────────────────────────────────
    auto_accepted, flagged = services["router"].route(all_ocr_results)

    logger.info(
        "librarian_routing_complete",
        job_id=job_id,
        auto_accepted=len(auto_accepted),
        flagged=len(flagged),
    )

    if flagged:
        # ── Pause for HITL review ─────────────────────────────────────────
        # Store flagged books in the review queue (Redis)
        services["review_queue"].create_session(job_id, flagged)

        # Store auto_accepted books so resume_after_review can access them.
        # We serialise them to Redis directly (not via Celery) because Celery
        # results are immutable once a task completes.
        _store_auto_accepted(job_id, auto_accepted, services["job_store"])

        # Putting a key-value pair in Redis DB, which says cataloging task for the job_id
        # awaits librarian review. When we poll "Get librarian/catalog/{job_id}" we either 
        # hit celery task status check logic if no books flagged for review and get catalog on task
        # SUCCESS, or we hit Redis Key that we only set (via services["review_queue"].create_session()) 
        # after getting books flagged for review.
        services["job_store"].set_awaiting_review(job_id, num_flagged=len(flagged))

        logger.info(
            "librarian_awaiting_review",
            job_id=job_id,
            flagged_count=len(flagged),
        )

        # Return a sentinel indicating AWAITING_REVIEW — the actual result
        # will be produced by resume_after_review() once corrections arrive.
        return {
            "state": "AWAITING_REVIEW",
            "flagged_count": len(flagged),
            "auto_accepted_count": len(auto_accepted),
        }

    # ── No flagged books: proceed directly to catalog ─────────────────────
    return _run_catalog_stage(
        job_id=job_id,
        auto_accepted=auto_accepted,
        corrections=[],
        flagged_originals=[],
        services=services,
    )


# ── Task 2: Resume after review (triggered when librarian submits corrections) ─

@celery_app.task(
    bind=True,
    name="pipelines.librarian.resume",
    max_retries=3,
    default_retry_delay=10,
    track_started=True,
)
def resume_after_review(self, job_id: str) -> dict:
    """
    Celery task: resume the librarian pipeline after HITL corrections arrive.

    Called by POST /librarian/review/{job_id} after the librarian submits
    corrections for all flagged books.

    Args:
        job_id: UUID for the job being resumed.

    Returns:
        Serialised CatalogResponse dict.
    """
    logger.info("librarian_pipeline_resume", job_id=job_id)
    services = _build_services()

    # ── Retrieve auto_accepted books stored during Task 1 ─────────────────
    auto_accepted = _load_auto_accepted(job_id, services["job_store"])

    # ── Retrieve librarian corrections ────────────────────────────────────
    corrections: list[BookCorrection] = services["review_queue"].get_corrections(job_id)
    flagged_originals: list[OCRValidatedResult] = services["review_queue"].get_pending(job_id)

    # ── Clear the AWAITING_REVIEW state ───────────────────────────────────
    services["job_store"].clear_awaiting_review(job_id)

    logger.info(
        "librarian_resume_data_loaded",
        job_id=job_id,
        auto_accepted=len(auto_accepted),
        corrections=len(corrections),
    )

    # ── Run catalog stage with merged results ─────────────────────────────
    return _run_catalog_stage(
        job_id=job_id,
        auto_accepted=auto_accepted,
        corrections=corrections,
        flagged_originals=flagged_originals,
        services=services,
    )


# ── Shared catalog stage ───────────────────────────────────────────────────────

def _run_catalog_stage(
    job_id: str,
    auto_accepted: list[OCRResult],
    corrections: list[BookCorrection],
    flagged_originals: list[OCRResult],
    services: dict,
) -> dict:
    """
    Merge, enrich via LLM, and persist the final catalog.

    Called either directly from run_librarian_pipeline (no flagged books)
    or from resume_after_review (after corrections submitted).

    Args:
        job_id:            Job identifier.
        auto_accepted:     High-confidence OCR results.
        corrections:       Human-supplied corrections for flagged books.
        flagged_originals: Original OCR results for flagged books.
        services:          Dict of instantiated service objects.

    Returns:
        Serialised CatalogResponse dict.
    """
    # ── Step 6: Merge OCR results + corrections → BookRecords / OCRValidatedResult ───
    merged_records: list = []
    
    corrections_dict: dict = {book.book_id : book for book in corrections}
    flagged_dict: dict = {book.book_id : book for book in flagged_originals}
    
    # Traverse through each flagged book -> take corrections -> do validation -> return Validated result
    for bk_id, book in flagged_dict.items():
        try:
            record = corrections_dict[bk_id]
        except:
            logger.warning(f"Flagged book with book Id: {bk_id} is missing from corrections.")
        
        title_author = f"{record.corrected_title} {record.corrected_author}"
        
        try:
            val_book: OCRValidatedResult = services["ocr"].validate(title_author,
                                                                    crop_path=book.crop_image_path, 
                                                                    book_id=book.book_id,
                                                                    )
            
            # Updating OCR spine text data, otherwise it will be "title_author"
            val_book.ori_ocr_ext_spine_txt = book.ori_ocr_ext_spine_txt   
            val_book.ocr_confidence = book.ocr_confidence
            
            logger.info("validation_for_a_corrected_book_complete",
                crop_path=book.crop_image_path,)
            
        except BookLensError as exc:
            logger.warning(
                "librarian_validation_for_a_corrected_book_failed",
                crop_path=book.crop_image_path, 
                error=exc.message,)
        
        # Appending validated book record to merged_records:
        merged_records.append(val_book)
    
    
    # Finally appending all "auto_accepted" book records to "merged_records":
    merged_records.extend(auto_accepted)
    
    
    # ── Step 7: Catalog Generation - Enrichment and Conversion to BookRecords ────────
    final_enriched_books: list[BookRecord] = services["catalog"].generate_catalog(merged_records)
    

    # ── Step 8: Vector store write-back ───────────────────────────────────
    try:
        services["vector_store"].upsert_batch(final_enriched_books)
        logger.info("librarian_vector_upsert_ok", job_id=job_id)
    except Exception as exc:
        logger.warning("librarian_vector_upsert_failed", error=str(exc))

    # ── Build final catalog response ──────────────────────────────────────
    auto_count    = len(auto_accepted)
    corrected_count = len(corrections)

    response = CatalogResponse(
        job_id=job_id,
        total_books=len(final_enriched_books),
        auto_accepted_count=auto_count,
        human_corrected_count=corrected_count,
        books=final_enriched_books,
        csv_download_url=f"/downloads/{job_id}/catalog.csv",
    )

    logger.info(
        "librarian_catalog_complete",
        job_id=job_id,
        total=len(final_enriched_books),
        auto=auto_count,
        corrected=corrected_count,
    )

    return response.model_dump(mode="json")



# ── Image processing helper ────────────────────────────────────────────────────
def _detect_and_ocr_image(
    all_image_paths: list[str],
    job_id: str,
    services: dict,
) -> list:
    """
    Run detection + OCR for a single shelf image.

    Returns a list of OCRResult objects (one per detected book spine).
    Errors in detection or OCR are logged and the image is skipped rather
    than crashing the entire job.

    Args:
        image_path: Absolute path to the shelf image.
        job_id:     Job identifier for crop namespacing.
        services:   Dict of service instances.

    Returns:
        List of OCRResult objects for books found in this image.
    """
    # Putting all the image through detection process and getting their crop paths:
    all_crop_paths: list[str] = []
        
    for img_idx, img_path in enumerate(all_image_paths):
        try:
            pil_image = PILImage.open(img_path).convert("RGB")
        except Exception as exc:
            logger.error("librarian_image_load_failed", path=img_path, error=str(exc))
            return []

        # ── Detection ─────────────────────────────────────────────────────────
        try:
            _, crop_paths = services["detector"].detect(pil_image, job_id, img_idx)
            all_crop_paths.extend(crop_paths)
            
        except BookLensError as exc:
            logger.warning("librarian_detection_failed", path=img_path, error=exc.message)
            return []

    # -------------------------------- OCR --------------------------------------
    final_ocr_results: list = []
    
    for idx, crop_path in enumerate(all_crop_paths):
        book_id = f"{job_id}_{idx}"
        try:
            book_spine_text, ocr_conf_score = services["ocr"].extract(crop_path=crop_path, 
                                                                    book_id=book_id,
                                                                    )
        except BookLensError as exc:
            logger.warning(
                "librarian_ocr_failed",
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
                "librarian_ocr_validation_failed",
                crop_path=crop_path,
                error=exc.message,)

        # Appending OCR confidence score in OCRValidatedResult object
        ocr_result.ocr_confidence = ocr_conf_score
        
        # ---------------------------- Vector DB Caching ---------------------------------
        if ocr_result.title:
            # Query the vector store using the OCR-extracted text
            cached = services["vector_store"].search(         
                title=ocr_result.title,
                author=ocr_result.author,
            )

            if cached:   # cached will be a highly enriched BookRecord object - ready for final output 
                # Cache hit: use the stored (enriched) BookRecord
                # NOTE: Future Improvement - Don't cache if BookRecord lacks description. 
                #                            Get summary and update the one in vector store.
                logger.info(
                    "librarian_cache_hit",
                    book_id=book_id,
                    title=cached.title,
                )
                # Update crop path to the freshly saved one (image may have moved)
                cached.crop_image_path = crop_path
                final_ocr_results.append(cached) 
            else:
                # Cache miss: 
                # append OCRValidatedResult to list → later we'll enrich it and make it a BookRecord
                final_ocr_results.append(ocr_result)
        
        else:
            final_ocr_results.append(ocr_result)   ## NOTE: These are the ones that must go for REVIEW.
        
    return final_ocr_results



# def _detect_and_ocr_image(
#     image_path: str,
#     job_id: str,
#     services: dict,
# ) -> list[OCRResult]:
#     """
#     Run detection + OCR for a single shelf image.

#     Returns a list of OCRResult objects (one per detected book spine).
#     Errors in detection or OCR are logged and the image is skipped rather
#     than crashing the entire job.

#     Args:
#         image_path: Absolute path to the shelf image.
#         job_id:     Job identifier for crop namespacing.
#         services:   Dict of service instances.

#     Returns:
#         List of OCRResult objects for books found in this image.
#     """
#     try:
#         pil_image = PILImage.open(image_path).convert("RGB")
#     except Exception as exc:
#         logger.error("librarian_image_load_failed", path=image_path, error=str(exc))
#         return []

#     # ── Detection ─────────────────────────────────────────────────────────
#     try:
#         _, crop_paths = services["detector"].detect(pil_image, job_id)
#     except BookLensError as exc:
#         logger.warning("librarian_detection_failed", path=image_path, error=exc.message)
#         return []

#     ocr_results: list[OCRResult] = []
#     for idx, crop_path in enumerate(crop_paths):
#         book_id = f"{job_id}_{Path(image_path).stem}_{idx}"
#         try:
#             result = services["ocr"].extract(crop_path=crop_path, book_id=book_id)
#             ocr_results.append(result)
#         except BookLensError as exc:
#             logger.warning(
#                 "librarian_ocr_failed",
#                 crop_path=crop_path,
#                 error=exc.message,
#             )

#     return ocr_results


# ── Auto-accepted books persistence helpers ────────────────────────────────────
# When a job pauses for review, auto-accepted books must be stored so
# resume_after_review() can retrieve them. We use Redis (already in the stack)
# with a key namespaced under the job_id.

def _store_auto_accepted(
    job_id: str,
    auto_accepted: list,
    job_store,
) -> None:
    """Serialise auto-accepted OCR results to Redis for later retrieval."""
    import json
    r = job_store._get_redis()   # Create and return redis client
    data = json.dumps([ocr.model_dump(mode="json") for ocr in auto_accepted])
    r.set(f"job:{job_id}:auto_accepted", data, ex=86_400)


def _load_auto_accepted(job_id: str, job_store) -> list:
    """Deserialise auto-accepted OCR results from Redis."""
    import json
    r = job_store._get_redis()
    raw = r.get(f"job:{job_id}:auto_accepted")
    if not raw:
        return []
    
    return [OCRValidatedResult(**item) if "about_book" in item.keys() else BookRecord(**item) for item in json.loads(raw)]


# ── Service factory ────────────────────────────────────────────────────────────
def _build_services() -> dict:
    """
    Instantiate all services needed by the reader pipeline.
    Same pattern as reader_pipeline._build_services().

    Returns:
        Dict mapping service name → service instance.
    """
    # Use the per-process cached services factory so models are loaded once
    # per worker process and reused across tasks.
    from app.core.worker_services import (get_worker_services_Det_OCR_Mer_Rou, 
                                          get_worker_services_Librarian_VS,
                                          get_worker_services_CAT_ESS
                                          )

    services = get_worker_services_Det_OCR_Mer_Rou()
    services['vector_store'] = get_worker_services_Librarian_VS()['vector_store']
    
    _cat_services = get_worker_services_CAT_ESS()
    services['catalog'] = _cat_services['catalog']
    services['job_store'] = _cat_services['job_store']
    services['review_queue'] = _cat_services['review_queue']

    return services

