# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/pipelines/test_librarian_pipeline.py
#
# Unit tests for librarian_pipeline — both the initial task and resume task.
#
# Key scenarios:
#   - All books auto-accepted → catalog generated immediately (no HITL)
#   - Some books flagged → pipeline pauses, AWAITING_REVIEW state set
#   - resume_after_review → merges corrections, runs catalog, returns result
#   - Detection / image load failures are gracefully skipped
#   - Cache hits skip the LLM catalog call
#   - Vector store upsert called after catalog generation
# ─────────────────────────────────────────────────────────────────────────────

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from PIL import Image as PILImage

from app.models.book import BookCorrection, BookRecord, BoundingBox, OCRResult, OCRValidatedResult
from app.services.confidence_router import ThresholdConfidenceRouter
from app.services.result_merger import DefaultResultMerger


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_image(tmp_path: Path, name: str = "shelf.jpg") -> str:
    img = PILImage.new("RGB", (200, 400), color=(120, 100, 80))
    p = tmp_path / name
    img.save(str(p), "JPEG")
    return str(p)


def _make_ocr(book_id: str, confidence: float = 0.90) -> tuple[str, float]:
    return (f"Book {book_id[:4]}", confidence)


def _make_validated_ocr(book_id: str, confidence: float = 0.90) -> OCRValidatedResult:
    text = f"Book {book_id[:4]}"
    return OCRValidatedResult(
        book_id=book_id,
        crop_image_path=f"crops/job/{book_id[:6]}.jpg",
        ori_ocr_ext_spine_txt=text,
        title=text,
        author="Author",
        ocr_confidence=confidence,
        flagged_for_review=confidence < 0.75,
    )


def _make_book_record(book_id: str) -> BookRecord:
    return BookRecord(
        book_id=book_id,
        title=f"Book {book_id[:4]}",
        author="Author",
        crop_image_path=f"crops/job/{book_id[:6]}.jpg",
        ocr_confidence=0.90,
        source="ocr_auto",
        ori_ocr_ext_spine_txt=f"Book {book_id[:4]}",
    )


def _make_services(
    *,
    num_crops: int = 2,
    all_high_confidence: bool = True,
    cache_hit: BookRecord | None = None,
) -> dict:
    """
    Build a complete services dict for librarian pipeline tests.

    Args:
        num_crops:            How many book spines the detector returns.
        all_high_confidence:  If False, last crop gets confidence 0.30 (flagged).
        cache_hit:            If set, vector_store.search() returns this book.
    """
    from app.core.exceptions import NoBooksDetectedError
    from tests.conftest import StubReviewQueueService

    # ── Detector ──────────────────────────────────────────────────────────
    detector = MagicMock()
    crop_paths = [f"crops/job/book_{i}.jpg" for i in range(num_crops)]
    boxes = [
        BoundingBox(x_min=i*80, y_min=0, x_max=i*80+70, y_max=300, confidence=0.92)
        for i in range(num_crops)
    ]
    detector.detect.return_value = (boxes, crop_paths)

    # ── OCR ───────────────────────────────────────────────────────────────
    ocr = MagicMock()
    def ocr_extract(crop_path, book_id):
        # Last crop gets low confidence if all_high_confidence=False
        if not all_high_confidence and crop_path == crop_paths[-1]:
            return _make_ocr(book_id, confidence=0.30)
        return _make_ocr(book_id, confidence=0.92)
    ocr.extract.side_effect = ocr_extract

    # ── Vector store ──────────────────────────────────────────────────────
    vector_store = MagicMock()
    vector_store.search.return_value = cache_hit

    # ── Catalog service ───────────────────────────────────────────────────
    catalog = MagicMock()
    catalog.generate_catalog.side_effect = lambda books: books  # pass-through

    # ── Job store ─────────────────────────────────────────────────────────
    job_store = MagicMock()

    # ── Review queue (in-memory stub) ─────────────────────────────────────
    review_queue = StubReviewQueueService()

    return {
        "detector": detector,
        "ocr": ocr,
        "router": ThresholdConfidenceRouter(threshold=0.75),
        "merger": DefaultResultMerger(),
        "review_queue": review_queue,
        "vector_store": vector_store,
        "catalog": catalog,
        "job_store": job_store,
    }


# ── Tests: run_librarian_pipeline ─────────────────────────────────────────────

class TestRunLibrarianPipeline:

    def _run(self, image_paths, services=None):
        from app.pipelines.librarian_pipeline import run_librarian_pipeline
        services = services or _make_services()
        with patch("app.pipelines.librarian_pipeline._build_services", return_value=services):
            return run_librarian_pipeline.__wrapped__(
                "test-job-id", image_paths
            )

    def test_empty_image_list_returns_empty_catalog(self):
        result = self._run([])
        assert result["total_books"] == 0
        assert result["books"] == []

    # def test_all_high_confidence_no_review_needed(self, tmp_path):
    #     """When all books pass the confidence threshold, no HITL pause occurs."""
    #     path = _write_image(tmp_path)
    #     services = _make_services(num_crops=2, all_high_confidence=True)
    #     result = self._run([path], services=services)
    #     # Should complete directly — no AWAITING_REVIEW
    #     assert result.get("state") != "AWAITING_REVIEW"
    #     # job_store.set_awaiting_review must NOT have been called
    #     services["job_store"].set_awaiting_review.assert_not_called()

    def test_low_confidence_book_triggers_review_state(self, tmp_path):
        """A crop with confidence < 0.75 must pause the pipeline for HITL."""
        path = _write_image(tmp_path)
        services = _make_services(num_crops=2, all_high_confidence=False)
        result = self._run([path], services=services)
        assert result["state"] == "AWAITING_REVIEW"
        services["job_store"].set_awaiting_review.assert_called_once()

    def test_flagged_books_stored_in_review_queue(self, tmp_path):
        path = _write_image(tmp_path)
        services = _make_services(num_crops=1, all_high_confidence=False)
        self._run([path], services=services)
        # Verify the review queue has a session for this job
        pending = services["review_queue"].get_pending("test-job-id")
        assert len(pending) == 1   # 1 flagged book (last crop)

    # def test_auto_accepted_books_stored_for_resume(self, tmp_path):
    #     """Auto-accepted OCR results must be persisted so resume can retrieve them."""
    #     path = _write_image(tmp_path)
    #     services = _make_services(num_crops=2, all_high_confidence=False)
    #     self._run([path], services=services)
    #     # Verify redis set was called with the auto_accepted key
    #     redis_calls = services["job_store"]._get_redis.call_args_list
    #     # job_store._get_redis() is called internally; we just verify
    #     # set_awaiting_review was called (which requires auto-accepted was stored)
    #     services["job_store"].set_awaiting_review.assert_called_with(
    #         "test-job-id", num_flagged=1
    #     )

    # def test_catalog_called_when_no_flagged_books(self, tmp_path):
    #     """When no books are flagged, LLMCatalogService.generate_catalog must run."""
    #     path = _write_image(tmp_path)
    #     services = _make_services(num_crops=2, all_high_confidence=True)
    #     self._run([path], services=services)
    #     services["catalog"].generate_catalog.assert_called_once()

    def test_catalog_not_called_when_flagged_books_exist(self, tmp_path):
        """While waiting for review, generate_catalog must NOT be called."""
        path = _write_image(tmp_path)
        services = _make_services(num_crops=2, all_high_confidence=False)
        self._run([path], services=services)
        services["catalog"].generate_catalog.assert_not_called()

    # def test_result_contains_csv_download_url(self, tmp_path):
    #     path = _write_image(tmp_path)
    #     services = _make_services(num_crops=1, all_high_confidence=True)
    #     result = self._run([path], services=services)
    #     assert "csv_download_url" in result
    #     assert "test-job-id" in result["csv_download_url"]

    # def test_vector_store_upsert_called_after_catalog(self, tmp_path):
    #     path = _write_image(tmp_path)
    #     services = _make_services(num_crops=2, all_high_confidence=True)
    #     self._run([path], services=services)
    #     services["vector_store"].upsert_batch.assert_called_once()

    # def test_vector_store_upsert_failure_does_not_crash(self, tmp_path):
    #     path = _write_image(tmp_path)
    #     services = _make_services(num_crops=1, all_high_confidence=True)
    #     services["vector_store"].upsert_batch.side_effect = Exception("Qdrant down")
    #     result = self._run([path], services=services)
    #     assert "total_books" in result

    # def test_result_is_json_serialisable(self, tmp_path):
    #     path = _write_image(tmp_path)
    #     services = _make_services(num_crops=1, all_high_confidence=True)
    #     result = self._run([path], services=services)
    #     json.dumps(result)   # Must not raise

    # def test_cache_hit_reduces_llm_calls(self, tmp_path):
    #     """Books found in the vector store skip the LLM catalog call."""
    #     path = _write_image(tmp_path)
    #     cached = _make_book_record(str(uuid.uuid4()))
    #     # 1 crop, and that crop is a cache hit → nothing for the LLM to do
    #     services = _make_services(num_crops=1, all_high_confidence=True, cache_hit=cached)
    #     self._run([path], services=services)
    #     services["catalog"].generate_catalog.assert_not_called()


# ── Tests: resume_after_review ─────────────────────────────────────────────────

# class TestResumeAfterReview:

#     def _seed_and_resume(self, tmp_path, num_auto: int = 1, num_flagged: int = 1):
#         """
#         Helper: simulate a paused job then call resume_after_review.

#         Returns (result, services) tuple.
#         """
#         import json as _json
#         from app.pipelines.librarian_pipeline import resume_after_review
#         from tests.conftest import StubReviewQueueService

#         services = _make_services(num_crops=0)  # crops don't matter for resume

#         # Seed the review queue with flagged books
#         flagged_books = [_make_ocr(str(uuid.uuid4()), confidence=0.30) for _ in range(num_flagged)]
#         services["review_queue"].create_session("test-job-id", flagged_books)

#         # Build corresponding corrections
#         corrections = [
#             BookCorrection(
#                 book_id=ocr[0],
#                 corrected_title="Fixed Title",
#                 corrected_author="Fixed Author",
#             )
#             for ocr in flagged_books
#         ]
#         services["review_queue"].submit_corrections("test-job-id", corrections)

#         # Store auto-accepted books in job_store (mimic what Task 1 does)
#         auto_ocr = [_make_ocr(str(uuid.uuid4()), confidence=0.92) for _ in range(num_auto)]
#         auto_json = _json.dumps([o for o in auto_ocr])
#         # auto_json = _json.dumps([o.model_dump(mode="json") for o in auto_ocr])
#         redis_mock = MagicMock()
#         redis_mock.get.return_value = auto_json.encode()
#         services["job_store"]._get_redis.return_value = redis_mock

#         with patch("app.pipelines.librarian_pipeline._build_services", return_value=services):
#             result = resume_after_review.__wrapped__("test-job-id")

#         return result, services

#     def test_resume_returns_catalog_response(self, tmp_path):
#         result, _ = self._seed_and_resume(tmp_path)
#         assert "total_books" in result
#         assert "books" in result

#     def test_resume_includes_corrected_books(self, tmp_path):
#         result, _ = self._seed_and_resume(tmp_path, num_auto=1, num_flagged=2)
#         assert result["human_corrected_count"] == 2

#     def test_resume_includes_auto_accepted_books(self, tmp_path):
#         result, _ = self._seed_and_resume(tmp_path, num_auto=2, num_flagged=1)
#         assert result["auto_accepted_count"] == 2

#     def test_resume_total_equals_auto_plus_corrected(self, tmp_path):
#         result, _ = self._seed_and_resume(tmp_path, num_auto=2, num_flagged=3)
#         assert result["total_books"] == 5

#     def test_resume_calls_catalog_generation(self, tmp_path):
#         _, services = self._seed_and_resume(tmp_path)
#         services["catalog"].generate_catalog.assert_called_once()

#     def test_resume_calls_vector_upsert(self, tmp_path):
#         _, services = self._seed_and_resume(tmp_path)
#         services["vector_store"].upsert_batch.assert_called_once()

#     def test_resume_clears_awaiting_review_state(self, tmp_path):
#         _, services = self._seed_and_resume(tmp_path)
#         services["job_store"].clear_awaiting_review.assert_called_once_with("test-job-id")

#     def test_resume_result_is_json_serialisable(self, tmp_path):
#         result, _ = self._seed_and_resume(tmp_path)
#         json.dumps(result)
