# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/pipelines/test_reader_pipeline.py
#
# Unit tests for reader_pipeline.run_reader_pipeline.
#
# Strategy: call the pipeline function directly (bypassing Celery task
# machinery) with injected stub services. We patch _build_services() to return
# a dict of lightweight stubs so no ML models, Redis, or Qdrant are needed.
#
# What we verify:
#   - Empty image list → valid empty ReaderResponse (no crash)
#   - Detection failure → graceful skip of that image
#   - Cache hit → uses vector store record, skips LLM re-enrichment
#   - Cache miss → goes through OCR → recommendations
#   - Vector store upsert is called after recommendations
#   - Returned dict has all required ReaderResponse fields
# ─────────────────────────────────────────────────────────────────────────────

import io
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from app.models.book import BookRecord, BoundingBox, OCRResult
from app.models.request import ReaderPreferences


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_image(tmp_path: Path, name: str = "shelf.jpg") -> str:
    """Write a tiny RGB JPEG to tmp_path and return its absolute path."""
    img = PILImage.new("RGB", (200, 400), color=(100, 80, 60))
    path = tmp_path / name
    img.save(str(path), format="JPEG")
    return str(path)


def _make_ocr(book_id: str, title: str = "Dune", confidence: float = 0.90) -> OCRResult:
    return OCRResult(
        book_id=book_id,
        crop_image_path=f"crops/job/{book_id[:6]}.jpg",
        raw_title=title,
        raw_author="Frank Herbert",
        confidence=confidence,
        flagged_for_review=False,
    )


def _make_book(title: str = "Dune") -> BookRecord:
    bid = str(uuid.uuid4())
    return BookRecord(
        book_id=bid,
        title=title,
        author="Frank Herbert",
        crop_image_path=f"crops/job/{bid[:6]}.jpg",
        ocr_confidence=0.90,
        source="ocr_auto",
        ori_ocr_ext_spine_txt=title,
    )


def _make_services(
    *,
    detect_raises=False,
    cache_hit: BookRecord | None = None,
    num_crops: int = 2,
) -> dict:
    """
    Build a complete services dict with MagicMock stubs.

    Args:
        detect_raises: If True, detector.detect() raises NoBooksDetectedError.
        cache_hit:     If provided, vector_store.search() returns this book.
        num_crops:     Number of book crops the detector returns.
    """
    from app.core.exceptions import NoBooksDetectedError
    from app.services.confidence_router import ThresholdConfidenceRouter
    from app.services.result_merger import DefaultResultMerger

    # ── Detector ──────────────────────────────────────────────────────────
    detector = MagicMock()
    if detect_raises:
        detector.detect.side_effect = NoBooksDetectedError("No books found")
    else:
        boxes = [
            BoundingBox(x_min=i*80, y_min=0, x_max=i*80+70, y_max=300, confidence=0.90)
            for i in range(num_crops)
        ]
        crop_paths = [f"crops/job/book_{i}.jpg" for i in range(num_crops)]
        detector.detect.return_value = (boxes, crop_paths)

    # ── OCR ───────────────────────────────────────────────────────────────
    ocr = MagicMock()
    ocr.extract.side_effect = lambda crop_path, book_id: _make_ocr(book_id)

    # ── Vector store ──────────────────────────────────────────────────────
    vector_store = MagicMock()
    vector_store.search.return_value = cache_hit   # None = miss, BookRecord = hit

    # ── Recommender ───────────────────────────────────────────────────────
    from app.models.book import BookRecommendation
    recommender = MagicMock()
    recommender.recommend.side_effect = lambda books, prefs: [
        BookRecommendation(
            book_id=b.book_id,
            title=b.title,
            author=b.author,
            crop_image_path=b.crop_image_path,
            summary="A great book.",
            rank=i + 1,
            match_reason="Matches preferences.",
        )
        for i, b in enumerate(books[:prefs.max_recommendations])
    ]

    return {
        "detector": detector,
        "ocr": ocr,
        "router": ThresholdConfidenceRouter(threshold=0.75),
        "merger": DefaultResultMerger(),
        "vector_store": vector_store,
        "recommender": recommender,
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestReaderPipeline:

    def _run(self, image_paths, preferences_dict=None, services=None, tmp_path=None):
        """Run the pipeline function directly with patched _build_services."""
        from app.pipelines.reader_pipeline import run_reader_pipeline

        services = services or _make_services()
        prefs = preferences_dict or ReaderPreferences().model_dump(mode="json")

        with patch("app.pipelines.reader_pipeline._build_services", return_value=services):
            # Call the underlying function, not the Celery task wrapper
            return run_reader_pipeline.__wrapped__(
                None,  # self (task instance) — unused in tests
                "test-job-id",
                image_paths,
                prefs,
            )

    def test_empty_image_list_returns_empty_response(self):
        result = self._run([])
        assert result["total_books_detected"] == 0
        assert result["recommendations"] == []
        assert result["all_books"] == []

    def test_no_books_detected_returns_empty_response(self, tmp_path):
        path = _write_image(tmp_path)
        services = _make_services(detect_raises=True)
        result = self._run([path], services=services)
        assert result["total_books_detected"] == 0

    def test_two_crops_produce_two_books(self, tmp_path):
        path = _write_image(tmp_path)
        services = _make_services(num_crops=2)
        result = self._run([path], services=services)
        assert result["total_books_detected"] == 2

    def test_result_has_job_id(self, tmp_path):
        path = _write_image(tmp_path)
        result = self._run([path])
        assert result["job_id"] == "test-job-id"

    def test_recommendations_returned(self, tmp_path):
        path = _write_image(tmp_path)
        services = _make_services(num_crops=3)
        result = self._run(
            [path],
            preferences_dict=ReaderPreferences(max_recommendations=3).model_dump(mode="json"),
            services=services,
        )
        assert len(result["recommendations"]) <= 3
        assert len(result["recommendations"]) > 0

    def test_recommendations_sorted_by_rank(self, tmp_path):
        path = _write_image(tmp_path)
        services = _make_services(num_crops=3)
        result = self._run([path], services=services)
        ranks = [r["rank"] for r in result["recommendations"]]
        assert ranks == sorted(ranks)

    def test_cache_hit_skips_recommendation_for_that_book(self, tmp_path):
        """
        When the vector store returns a hit, that book's data comes from
        the cache. The recommender still receives it as part of the full list.
        """
        path = _write_image(tmp_path)
        cached_book = _make_book(title="Cached Dune")
        services = _make_services(num_crops=1, cache_hit=cached_book)
        result = self._run([path], services=services)
        # The book from cache should appear in all_books
        titles = [b["title"] for b in result["all_books"]]
        assert "Cached Dune" in titles

    def test_vector_store_upsert_called_after_recommendations(self, tmp_path):
        """After generating recommendations, books must be upserted."""
        path = _write_image(tmp_path)
        services = _make_services(num_crops=2)
        self._run([path], services=services)
        services["vector_store"].upsert_batch.assert_called_once()

    def test_vector_store_upsert_failure_does_not_crash(self, tmp_path):
        """A vector store failure must not propagate — results still returned."""
        path = _write_image(tmp_path)
        services = _make_services(num_crops=1)
        services["vector_store"].upsert_batch.side_effect = Exception("Qdrant down")
        # Should not raise
        result = self._run([path], services=services)
        assert result["total_books_detected"] >= 0

    def test_multiple_images_aggregated(self, tmp_path):
        """Books from multiple shelf images must all appear in the response."""
        path1 = _write_image(tmp_path, "shelf1.jpg")
        path2 = _write_image(tmp_path, "shelf2.jpg")
        services = _make_services(num_crops=2)
        result = self._run([path1, path2], services=services)
        # 2 images × 2 crops = 4 total books
        assert result["total_books_detected"] == 4

    def test_result_is_json_serialisable(self, tmp_path):
        """The pipeline result must be serialisable to JSON (for Celery backend)."""
        path = _write_image(tmp_path)
        result = self._run([path])
        # Should not raise
        json.dumps(result)

    def test_all_books_field_present(self, tmp_path):
        path = _write_image(tmp_path)
        result = self._run([path])
        assert "all_books" in result

    def test_ocr_called_once_per_crop(self, tmp_path):
        path = _write_image(tmp_path)
        services = _make_services(num_crops=3)
        self._run([path], services=services)
        assert services["ocr"].extract.call_count == 3
