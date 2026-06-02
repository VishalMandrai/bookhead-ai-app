# ─────────────────────────────────────────────────────────────────────────────
# app/services/review_queue.py
#
# RedisReviewQueueService – production implementation of BaseReviewQueueService.
#
# RESPONSIBILITY:
#   Persists the human-in-the-loop review sessions between the two HTTP
#   calls that make up the HITL flow:
#     1. POST /librarian/catalog  → pipeline flags books → session created
#     2. POST /librarian/review/{job_id} → librarian submits corrections
#
#   Since these two calls happen at different times (possibly from different
#   worker processes in a multi-worker deployment), the session state must
#   live outside the process — Redis is the natural choice because it is
#   already in the stack as the Celery broker.
#
# REDIS KEY LAYOUT:
#   review:{job_id}:flagged      → JSON list of OCRResult dicts
#   review:{job_id}:corrections  → JSON list of BookCorrection dicts
#   review:{job_id}:complete     → "1" when corrections have been submitted
#
#   All keys share a TTL (default 24 h) so abandoned sessions don't
#   accumulate forever.
#
# SOLID (Single Responsibility):
#   This service owns only session persistence. It does not make routing
#   decisions (that's ThresholdConfidenceRouter) or merge results (that's
#   DefaultResultMerger).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
from typing import Any

from app.core.exceptions import ReviewAlreadyCompleteError, ReviewSessionNotFoundError
from app.core.logging import get_logger
from app.models.book import BookCorrection, OCRValidatedResult
from app.services.base import BaseReviewQueueService

logger = get_logger(__name__)

# Time-to-live for all review session keys (seconds).
# 24 hours gives librarians plenty of time to complete a review.
SESSION_TTL_SECONDS = 86_400


class RedisReviewQueueService(BaseReviewQueueService):
    """
    Redis-backed review queue for HITL OCR correction sessions.

    Each job gets three Redis keys namespaced under review:{job_id}:.
    All keys share the same TTL so they expire atomically.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = SESSION_TTL_SECONDS,
    ) -> None:
        """
        Args:
            redis_url:   Redis connection URL (from Settings.redis_url).
            ttl_seconds: How long to keep session data before auto-expiry.
        """
        self._redis_url = redis_url
        self._ttl = ttl_seconds
        # Lazy-loaded Redis client; None until first operation
        self._redis: Any | None = None

    # ── Public API (implements BaseReviewQueueService) ─────────────────────────

    def create_session(
        self,
        job_id: str,
        flagged_books: list[OCRValidatedResult],
    ) -> None:
        """
        Persist the list of flagged books for this job in Redis.

        Serialises each OCRResult to JSON via Pydantic's model_dump().
        Sets a TTL on all keys so stale sessions are cleaned up automatically.
        """
        r = self._get_redis()   # Lazily create and return the Redis client

        # Serialise list of OCRResult → JSON string
        flagged_json = json.dumps(
            [book.model_dump(mode="json") for book in flagged_books]
        )

        # Use a Redis pipeline to write all keys atomically in one round-trip
        pipe = r.pipeline()
        pipe.set(self._key_flagged(job_id), flagged_json, ex=self._ttl)
        pipe.set(self._key_complete(job_id), "0", ex=self._ttl)
        pipe.execute()

        logger.info(
            "review_session_created",
            job_id=job_id,
            num_flagged=len(flagged_books),
            ttl_seconds=self._ttl,
        )

    def get_pending(self, job_id: str) -> list[OCRValidatedResult]:
        """
        Return the list of flagged books awaiting review for this job.

        Raises:
            ReviewSessionNotFoundError: If no session exists for job_id.
        """
        r = self._get_redis()                      # Lazily create and return the Redis client
        raw = r.get(self._key_flagged(job_id))     # Getting Review status for sent job_id using Redis Client

        if raw is None:
            raise ReviewSessionNotFoundError(
                f"No review session found for job '{job_id}'.",
                detail={"job_id": job_id},
            )

        # Deserialise JSON → list of OCRResult via Pydantic
        data = json.loads(raw)
        return [OCRValidatedResult(**item) for item in data]

    def submit_corrections(
        self,
        job_id: str,
        corrections: list[BookCorrection],
    ) -> None:
        """
        Persist librarian corrections and mark the session as complete.

        Raises:
            ReviewSessionNotFoundError:  If no session exists for job_id.
            ReviewAlreadyCompleteError:  If corrections were already submitted.
        """
        r = self._get_redis()     # Lazily create and return the Redis client

        # Guard: session must exist
        if r.get(self._key_flagged(job_id)) is None:
            raise ReviewSessionNotFoundError(
                f"No review session found for job '{job_id}'.",
                detail={"job_id": job_id},
            )

        # Guard: must not have been submitted already
        if r.get(self._key_complete(job_id)) == b"1":
            raise ReviewAlreadyCompleteError(
                f"Corrections for job '{job_id}' have already been submitted.",
                detail={"job_id": job_id},
            )

        # Serialise corrections → JSON
        corrections_json = json.dumps(
            [c.model_dump(mode="json") for c in corrections]
        )

        # Atomically write corrections and flip the complete flag
        pipe = r.pipeline()
        pipe.set(self._key_corrections(job_id), corrections_json, ex=self._ttl)
        pipe.set(self._key_complete(job_id), "1", ex=self._ttl)
        pipe.execute()

        logger.info(
            "review_corrections_stored",
            job_id=job_id,
            num_corrections=len(corrections),
        )

    def get_corrections(self, job_id: str) -> list[BookCorrection]:
        """
        Return the stored corrections for a completed session.

        Raises:
            ReviewSessionNotFoundError: If no corrections key exists.
        """
        r = self._get_redis()           # Lazily create and return the Redis client
        raw = r.get(self._key_corrections(job_id))

        if raw is None:
            raise ReviewSessionNotFoundError(
                f"No corrections found for job '{job_id}'. "
                "Corrections may not have been submitted yet.",
                detail={"job_id": job_id},
            )

        data = json.loads(raw)
        return [BookCorrection(**item) for item in data]

    def is_complete(self, job_id: str) -> bool:
        """
        Return True if corrections have been submitted for this session.
        Returns False (not raises) for unknown job_ids, so callers can
        safely poll without try/except.
        """
        r = self._get_redis()                       # Lazily create and return the Redis client
        value = r.get(self._key_complete(job_id))
        return value == b"1"

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_redis(self):
        """
        Lazily create and return the Redis client.

        decode_responses=False so we get raw bytes back, which lets us
        distinguish None (key missing) from b"0" (key exists but not done).
        """
        if self._redis is None:
            import redis

            logger.info("review_queue_connecting_redis", url=self._redis_url)
            self._redis = redis.from_url(
                self._redis_url,
                decode_responses=False,  # Return bytes, not str
            )
        return self._redis

    # ── Redis key builders ─────────────────────────────────────────────────────

    @staticmethod
    def _key_flagged(job_id: str) -> str:
        """Redis key for the list of flagged OCRResult dicts."""
        return f"review:{job_id}:flagged"

    @staticmethod
    def _key_corrections(job_id: str) -> str:
        """Redis key for the submitted BookCorrection dicts."""
        return f"review:{job_id}:corrections"

    @staticmethod
    def _key_complete(job_id: str) -> str:
        """Redis key for the session completion flag ("0" or "1")."""
        return f"review:{job_id}:complete"
