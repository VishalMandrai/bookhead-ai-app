# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_confidence_router.py
#
# Unit tests for ThresholdConfidenceRouter.
#
# These tests verify the routing logic in complete isolation – no FastAPI,
# no database, no models. Just the router and some OCRResult objects.
# ─────────────────────────────────────────────────────────────────────────────

import uuid
import pytest
from app.models.book import OCRResult


def _make_ocr_result(confidence: float) -> OCRResult:
    """Helper: build a minimal OCRResult with the given confidence score."""
    return OCRResult(
        book_id=str(uuid.uuid4()),
        crop_image_path=f"crop_{uuid.uuid4().hex[:6]}.jpg",
        raw_title="Some Title",
        raw_author="Some Author",
        confidence=confidence,
        flagged_for_review=False,
    )


class TestThresholdConfidenceRouter:
    """Tests for the threshold-based confidence routing logic."""

    def test_high_confidence_books_are_auto_accepted(self):
        """Books at or above the threshold must go to auto_accepted."""
        from tests.conftest import StubConfidenceRouter

        router = StubConfidenceRouter()  # Threshold = 0.75
        high = _make_ocr_result(0.90)
        at_threshold = _make_ocr_result(0.75)

        auto_accepted, flagged = router.route([high, at_threshold])

        assert len(auto_accepted) == 2
        assert len(flagged) == 0

    def test_low_confidence_books_are_flagged(self):
        """Books below the threshold must go to the flagged list."""
        from tests.conftest import StubConfidenceRouter

        router = StubConfidenceRouter()
        low = _make_ocr_result(0.50)
        very_low = _make_ocr_result(0.10)

        auto_accepted, flagged = router.route([low, very_low])

        assert len(auto_accepted) == 0
        assert len(flagged) == 2

    def test_flagged_books_have_flag_set(self):
        """The router must set flagged_for_review=True on flagged results."""
        from tests.conftest import StubConfidenceRouter

        router = StubConfidenceRouter()
        low = _make_ocr_result(0.40)

        _, flagged = router.route([low])

        assert flagged[0].flagged_for_review is True

    def test_auto_accepted_books_preserve_flag_as_false(self):
        """Auto-accepted books must NOT have their flag set to True."""
        from tests.conftest import StubConfidenceRouter

        router = StubConfidenceRouter()
        high = _make_ocr_result(0.95)

        auto_accepted, _ = router.route([high])

        assert auto_accepted[0].flagged_for_review is False

    def test_mixed_batch_split_correctly(self):
        """A mixed batch must be split into correct buckets."""
        from tests.conftest import StubConfidenceRouter

        router = StubConfidenceRouter()
        results = [
            _make_ocr_result(0.95),  # auto
            _make_ocr_result(0.30),  # flagged
            _make_ocr_result(0.75),  # auto (at boundary)
            _make_ocr_result(0.60),  # flagged
            _make_ocr_result(0.80),  # auto
        ]

        auto_accepted, flagged = router.route(results)

        assert len(auto_accepted) == 3
        assert len(flagged) == 2

    def test_empty_input_returns_two_empty_lists(self):
        """An empty input must return two empty lists without errors."""
        from tests.conftest import StubConfidenceRouter

        router = StubConfidenceRouter()
        auto_accepted, flagged = router.route([])

        assert auto_accepted == []
        assert flagged == []

    def test_original_order_preserved_in_auto_accepted(self):
        """The relative order of results must be preserved in each bucket."""
        from tests.conftest import StubConfidenceRouter

        router = StubConfidenceRouter()
        r1 = _make_ocr_result(0.90)
        r2 = _make_ocr_result(0.85)
        r3 = _make_ocr_result(0.80)

        auto_accepted, _ = router.route([r1, r2, r3])

        assert [r.book_id for r in auto_accepted] == [r1.book_id, r2.book_id, r3.book_id]
