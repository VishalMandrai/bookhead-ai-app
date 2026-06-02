# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_review_queue.py
#
# Unit tests for the review queue service (using the StubReviewQueueService).
#
# These tests exercise the HITL correction session lifecycle:
#   create → get_pending → submit_corrections → get_corrections → is_complete
# ─────────────────────────────────────────────────────────────────────────────

import uuid
import pytest

from app.core.exceptions import ReviewAlreadyCompleteError, ReviewSessionNotFoundError
from app.models.book import BookCorrection, OCRResult


def _make_flagged(book_id: str | None = None) -> OCRResult:
    return OCRResult(
        book_id=book_id or str(uuid.uuid4()),
        crop_image_path="crop.jpg",
        raw_title="Garbled Text",
        raw_author="Unkwn",
        confidence=0.40,
        flagged_for_review=True,
    )


def _make_correction(book_id: str) -> BookCorrection:
    return BookCorrection(
        book_id=book_id,
        corrected_title="Clean Title",
        corrected_author="Clean Author",
    )


class TestStubReviewQueueService:
    """Tests for the review queue session lifecycle."""

    def test_create_and_get_pending(self):
        """Creating a session and fetching pending books returns the same list."""
        from tests.conftest import StubReviewQueueService

        svc = StubReviewQueueService()
        job_id = str(uuid.uuid4())
        books = [_make_flagged(), _make_flagged()]

        svc.create_session(job_id, books)
        pending = svc.get_pending(job_id)

        assert len(pending) == 2
        assert {b.book_id for b in pending} == {b.book_id for b in books}

    def test_get_pending_raises_for_unknown_job(self):
        """Accessing a non-existent session must raise ReviewSessionNotFoundError."""
        from tests.conftest import StubReviewQueueService

        svc = StubReviewQueueService()

        with pytest.raises(ReviewSessionNotFoundError):
            svc.get_pending("does-not-exist")

    def test_is_complete_false_before_submission(self):
        """A newly created session must not be marked complete."""
        from tests.conftest import StubReviewQueueService

        svc = StubReviewQueueService()
        job_id = str(uuid.uuid4())
        svc.create_session(job_id, [_make_flagged()])

        assert svc.is_complete(job_id) is False

    def test_submit_corrections_marks_session_complete(self):
        """Submitting corrections must mark the session as complete."""
        from tests.conftest import StubReviewQueueService

        svc = StubReviewQueueService()
        job_id = str(uuid.uuid4())
        book = _make_flagged()
        svc.create_session(job_id, [book])

        svc.submit_corrections(job_id, [_make_correction(book.book_id)])

        assert svc.is_complete(job_id) is True

    def test_get_corrections_returns_submitted_data(self):
        """get_corrections must return exactly what was submitted."""
        from tests.conftest import StubReviewQueueService

        svc = StubReviewQueueService()
        job_id = str(uuid.uuid4())
        book = _make_flagged()
        correction = _make_correction(book.book_id)

        svc.create_session(job_id, [book])
        svc.submit_corrections(job_id, [correction])

        corrections = svc.get_corrections(job_id)

        assert len(corrections) == 1
        assert corrections[0].corrected_title == "Clean Title"
        assert corrections[0].corrected_author == "Clean Author"

    def test_is_complete_false_for_unknown_job(self):
        """is_complete must return False (not raise) for an unknown job_id."""
        from tests.conftest import StubReviewQueueService

        svc = StubReviewQueueService()
        # Should not raise – just returns False
        assert svc.is_complete("ghost-job") is False
