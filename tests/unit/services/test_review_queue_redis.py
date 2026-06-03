# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_review_queue_redis.py
#
# Unit tests for RedisReviewQueueService.
#
# Redis is NEVER contacted — a MagicMock replaces the redis client.
# We use fakeredis for integration-level tests that verify the full
# serialise→store→deserialise cycle with an in-memory Redis implementation.
# ─────────────────────────────────────────────────────────────────────────────

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import ReviewAlreadyCompleteError, ReviewSessionNotFoundError
from app.models.book import BookCorrection, OCRResult
from app.services.review_queue import RedisReviewQueueService


# ── Helpers ────────────────────────────────────────────────────────────────────

def _flagged(book_id: str | None = None) -> OCRResult:
    return OCRResult(
        book_id=book_id or str(uuid.uuid4()),
        crop_image_path="crops/job/book_0.jpg",
        raw_title="Blurry Title",
        raw_author="Unkwn",
        confidence=0.42,
        flagged_for_review=True,
    )


def _correction(book_id: str) -> BookCorrection:
    return BookCorrection(
        book_id=book_id,
        corrected_title="Clear Title",
        corrected_author="Clear Author",
    )


class _FakeRedis:
    """
    Lightweight in-memory Redis substitute.
    Implements only the methods used by RedisReviewQueueService:
      get, set, pipeline
    """

    def __init__(self):
        self._store: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    def set(self, key: str, value: str | bytes, ex: int = None) -> None:
        if isinstance(value, str):
            value = value.encode()
        self._store[key] = value

    def pipeline(self) -> "_FakePipeline":
        return _FakePipeline(self)


class _FakePipeline:
    """Buffered pipeline that executes all commands on execute()."""

    def __init__(self, redis: _FakeRedis):
        self._redis = redis
        self._commands: list[tuple] = []

    def set(self, key: str, value: str | bytes, ex: int = None) -> "_FakePipeline":
        self._commands.append(("set", key, value, ex))
        return self

    def execute(self):
        for cmd in self._commands:
            if cmd[0] == "set":
                _, key, value, ex = cmd
                self._redis.set(key, value, ex=ex)
        self._commands.clear()


def _make_service() -> tuple[RedisReviewQueueService, _FakeRedis]:
    """Build a service with a FakeRedis injected."""
    svc = RedisReviewQueueService(redis_url="redis://fake", ttl_seconds=3600)
    fake_redis = _FakeRedis()
    svc._redis = fake_redis
    return svc, fake_redis


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestRedisReviewQueueService:

    def test_create_session_stores_flagged_books(self):
        """After create_session, get_pending must return the same books."""
        svc, _ = _make_service()
        job_id = str(uuid.uuid4())
        book = _flagged()

        svc.create_session(job_id, [book])
        pending = svc.get_pending(job_id)

        assert len(pending) == 1
        assert pending[0].book_id == book.book_id

    def test_create_session_stores_multiple_books(self):
        svc, _ = _make_service()
        job_id = str(uuid.uuid4())
        books = [_flagged() for _ in range(4)]

        svc.create_session(job_id, books)
        pending = svc.get_pending(job_id)

        assert len(pending) == 4

    def test_create_session_marks_as_incomplete(self):
        svc, _ = _make_service()
        job_id = str(uuid.uuid4())
        svc.create_session(job_id, [_flagged()])

        assert svc.is_complete(job_id) is False

    def test_get_pending_raises_for_unknown_job(self):
        svc, _ = _make_service()

        with pytest.raises(ReviewSessionNotFoundError):
            svc.get_pending("does-not-exist")

    def test_serialisation_round_trip_preserves_all_fields(self):
        """OCRResult fields must survive the JSON serialise→store→load cycle."""
        svc, _ = _make_service()
        job_id = str(uuid.uuid4())
        book = _flagged(book_id="fixed-book-id")
        book.raw_title = "Specific Title"
        book.raw_author = "Specific Author"
        book.confidence = 0.38

        svc.create_session(job_id, [book])
        pending = svc.get_pending(job_id)

        assert pending[0].raw_title == "Specific Title"
        assert pending[0].raw_author == "Specific Author"
        assert pending[0].confidence == 0.38

    def test_submit_corrections_marks_session_complete(self):
        svc, _ = _make_service()
        job_id = str(uuid.uuid4())
        book = _flagged()
        svc.create_session(job_id, [book])

        svc.submit_corrections(job_id, [_correction(book.book_id)])

        assert svc.is_complete(job_id) is True

    def test_submit_corrections_stores_corrections(self):
        svc, _ = _make_service()
        job_id = str(uuid.uuid4())
        book = _flagged()
        svc.create_session(job_id, [book])

        corr = _correction(book.book_id)
        svc.submit_corrections(job_id, [corr])
        corrections = svc.get_corrections(job_id)

        assert len(corrections) == 1
        assert corrections[0].corrected_title == "Clear Title"

    def test_submit_corrections_raises_for_unknown_job(self):
        svc, _ = _make_service()

        with pytest.raises(ReviewSessionNotFoundError):
            svc.submit_corrections("ghost-job", [_correction("some-id")])

    def test_submit_corrections_raises_if_already_complete(self):
        """Submitting corrections twice for the same job must raise."""
        svc, _ = _make_service()
        job_id = str(uuid.uuid4())
        book = _flagged()
        svc.create_session(job_id, [book])
        svc.submit_corrections(job_id, [_correction(book.book_id)])

        with pytest.raises(ReviewAlreadyCompleteError):
            svc.submit_corrections(job_id, [_correction(book.book_id)])

    def test_get_corrections_raises_before_submission(self):
        """get_corrections called before submit_corrections must raise."""
        svc, _ = _make_service()
        job_id = str(uuid.uuid4())
        svc.create_session(job_id, [_flagged()])

        with pytest.raises(ReviewSessionNotFoundError):
            svc.get_corrections(job_id)

    def test_is_complete_returns_false_for_unknown_job(self):
        """is_complete must return False (not raise) for unknown job_ids."""
        svc, _ = _make_service()
        assert svc.is_complete("no-such-job") is False

    def test_corrections_round_trip_preserves_author(self):
        """BookCorrection fields must survive the JSON round-trip."""
        svc, _ = _make_service()
        job_id = str(uuid.uuid4())
        book = _flagged()
        svc.create_session(job_id, [book])

        corr = BookCorrection(
            book_id=book.book_id,
            corrected_title="Real Book Title",
            corrected_author="Real Author Name",
        )
        svc.submit_corrections(job_id, [corr])
        corrections = svc.get_corrections(job_id)

        assert corrections[0].corrected_author == "Real Author Name"

    def test_redis_keys_use_job_id_namespace(self):
        """All Redis keys for a session must include the job_id."""
        svc, fake_redis = _make_service()
        job_id = "test-job-123"
        svc.create_session(job_id, [_flagged()])

        stored_keys = list(fake_redis._store.keys())
        assert all(job_id in k for k in stored_keys)
