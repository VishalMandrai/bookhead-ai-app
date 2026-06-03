# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_result_merger.py
#
# Unit tests for DefaultResultMerger.
# ─────────────────────────────────────────────────────────────────────────────

import uuid
import pytest

from app.models.book import BookCorrection, BookRecord, OCRResult
from app.services.result_merger import DefaultResultMerger


def _auto(book_id: str | None = None, title: str = "Auto Title") -> OCRResult:
    """Build a high-confidence (auto-accepted) OCRResult."""
    return OCRResult(
        book_id=book_id or str(uuid.uuid4()),
        crop_image_path=f"crops/job/{uuid.uuid4().hex[:6]}.jpg",
        raw_title=title,
        raw_author="Auto Author",
        confidence=0.92,
        flagged_for_review=False,
    )


def _flagged(book_id: str | None = None, title: str = "Blurry Title") -> OCRResult:
    """Build a low-confidence (flagged) OCRResult."""
    return OCRResult(
        book_id=book_id or str(uuid.uuid4()),
        crop_image_path=f"crops/job/{uuid.uuid4().hex[:6]}.jpg",
        raw_title=title,
        raw_author="Unkwn Auth",
        confidence=0.40,
        flagged_for_review=True,
    )


def _correction(book_id: str, title: str = "Corrected Title") -> BookCorrection:
    return BookCorrection(
        book_id=book_id,
        corrected_title=title,
        corrected_author="Corrected Author",
    )


class TestDefaultResultMerger:

    def test_auto_accepted_books_have_source_ocr_auto(self):
        merger = DefaultResultMerger()
        auto = _auto()
        records = merger.merge(
            auto_accepted=[auto],
            corrections=[],
            flagged_originals=[],
        )
        assert len(records) == 1
        assert records[0].source == "ocr_auto"

    def test_corrected_books_have_source_human_corrected(self):
        merger = DefaultResultMerger()
        flagged = _flagged()
        correction = _correction(flagged.book_id)
        records = merger.merge(
            auto_accepted=[],
            corrections=[correction],
            flagged_originals=[flagged],
        )
        assert len(records) == 1
        assert records[0].source == "human_corrected"

    def test_auto_accepted_title_comes_from_ocr(self):
        merger = DefaultResultMerger()
        auto = _auto(title="Great Expectations")
        records = merger.merge([auto], [], [])
        assert records[0].title == "Great Expectations"

    def test_corrected_title_overrides_ocr(self):
        merger = DefaultResultMerger()
        flagged = _flagged(title="Grbld Ttl")
        correction = _correction(flagged.book_id, title="Clean Title")
        records = merger.merge(
            auto_accepted=[],
            corrections=[correction],
            flagged_originals=[flagged],
        )
        assert records[0].title == "Clean Title"

    def test_corrected_author_overrides_ocr(self):
        merger = DefaultResultMerger()
        flagged = _flagged()
        correction = BookCorrection(
            book_id=flagged.book_id,
            corrected_title="Real Title",
            corrected_author="Real Author",
        )
        records = merger.merge([], [correction], [flagged])
        assert records[0].author == "Real Author"

    def test_crop_path_preserved_for_corrected_books(self):
        """Human-corrected records should still carry the original crop path."""
        merger = DefaultResultMerger()
        flagged = _flagged()
        original_path = flagged.crop_image_path
        correction = _correction(flagged.book_id)
        records = merger.merge([], [correction], [flagged])
        assert records[0].crop_image_path == original_path

    def test_ocr_confidence_preserved_for_corrected_books(self):
        """The original OCR confidence should be preserved on corrected records."""
        merger = DefaultResultMerger()
        flagged = _flagged()
        records = merger.merge([], [_correction(flagged.book_id)], [flagged])
        assert records[0].ocr_confidence == flagged.confidence

    def test_empty_inputs_return_empty_list(self):
        merger = DefaultResultMerger()
        records = merger.merge([], [], [])
        assert records == []

    def test_mixed_auto_and_corrected_all_returned(self):
        merger = DefaultResultMerger()
        auto1 = _auto(title="Book A")
        auto2 = _auto(title="Book B")
        flagged = _flagged()
        correction = _correction(flagged.book_id, title="Book C")

        records = merger.merge(
            auto_accepted=[auto1, auto2],
            corrections=[correction],
            flagged_originals=[flagged],
        )
        assert len(records) == 3

    def test_order_preserved_auto_first_then_flagged(self):
        """
        Records should appear in order: auto_accepted first (in input order),
        then human-corrected (in flagged_originals order).
        """
        merger = DefaultResultMerger()
        auto1 = _auto(title="First Auto")
        auto2 = _auto(title="Second Auto")
        f1 = _flagged(title="First Flagged")
        f2 = _flagged(title="Second Flagged")

        records = merger.merge(
            auto_accepted=[auto1, auto2],
            corrections=[_correction(f1.book_id, "Fixed F1"), _correction(f2.book_id, "Fixed F2")],
            flagged_originals=[f1, f2],
        )

        titles = [r.title for r in records]
        assert titles == ["First Auto", "Second Auto", "Fixed F1", "Fixed F2"]

    def test_all_records_are_book_record_instances(self):
        merger = DefaultResultMerger()
        auto = _auto()
        flagged = _flagged()
        records = merger.merge(
            auto_accepted=[auto],
            corrections=[_correction(flagged.book_id)],
            flagged_originals=[flagged],
        )
        for record in records:
            assert isinstance(record, BookRecord)
