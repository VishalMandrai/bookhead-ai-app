# ─────────────────────────────────────────────────────────────────────────────
# app/services/job_store.py
#
# Job state store — reads Celery task results from Redis and translates them
# into the JobStatusResponse models the API routes return.
#
# WHY NOT JUST USE CELERY'S AsyncResult DIRECTLY IN ROUTES?
#   1. Celery's task states (PENDING, STARTED, SUCCESS, FAILURE) don't include
#      our custom "AWAITING_REVIEW" state. We need a layer that knows to check
#      the review queue when a librarian job is in a special intermediate state.
#   2. Celery result objects are not JSON-serialisable out of the box — they
#      need to be adapted into our response models.
#   3. Keeping this logic here makes routes thin and testable without Celery.
#
# REDIS KEY LAYOUT (used in addition to Celery's own result keys):
#   job:{job_id}:state   → "AWAITING_REVIEW" (set by librarian pipeline)
#   job:{job_id}:meta    → JSON with progress_message and other metadata
#
# SOLID note (Single Responsibility):
#   This module owns job state translation. Routes ask it for a
#   JobStatusResponse; it handles all the Celery/Redis details internally.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
from typing import Any

from app.core.logging import get_logger
from app.models.response import JobStatusResponse

logger = get_logger(__name__)

# TTL for custom job metadata keys (24 h — same as review session TTL)
JOB_META_TTL = 86_400


class JobStore:
    """
    Reads and writes job state in Redis, layered on top of Celery's result backend.

    Provides:
      - get_status(job_id) → JobStatusResponse    for any job type
      - set_awaiting_review(job_id)               for librarian pipeline pause
      - set_meta(job_id, message)                 for progress updates
      - clear_awaiting_review(job_id)             when review resumes
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._redis: Any | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_status(self, job_id: str) -> JobStatusResponse:
        """
        Return the current state of a job as a JobStatusResponse.

        Resolution order:
          1. Check our custom Redis key for AWAITING_REVIEW state
             (set by the librarian pipeline after OCR confidence routing).
          2. Query Celery's AsyncResult for PENDING/STARTED/SUCCESS/FAILURE.
          3. Fall back to PENDING if neither source has information yet.
        """
        r = self._get_redis()

        # Check for our custom AWAITING_REVIEW state first
        custom_state = r.get(self._key_state(job_id))
        if custom_state == b"AWAITING_REVIEW":
            meta = self._get_meta(job_id)
            return JobStatusResponse(
                job_id=job_id,
                state="AWAITING_REVIEW",
                progress_message=meta.get(
                    "message",
                    "Some books need review before the catalog can be generated.",
                ),
            )

        # Fall back to Celery's result backend
        return self._celery_status(job_id)


    def set_awaiting_review(self, job_id: str, num_flagged: int) -> None:
        """
        Mark a librarian job as paused, waiting for human review corrections.

        Called by the librarian pipeline after OCR confidence routing reveals
        low-confidence books that need human verification.
        """
        r = self._get_redis()
        r.set(self._key_state(job_id), "AWAITING_REVIEW", ex=JOB_META_TTL)
        self._set_meta(
            job_id,
            {
                "message": (
                    f"{num_flagged} book(s) need review. "
                    "Fetch GET /librarian/review/{job_id} to see them."
                )
            },
        )
        logger.info("job_awaiting_review", job_id=job_id, num_flagged=num_flagged)

    def clear_awaiting_review(self, job_id: str) -> None:
        """
        Remove the AWAITING_REVIEW flag after corrections have been submitted
        and the pipeline is resuming.
        """
        r = self._get_redis()
        r.delete(self._key_state(job_id))
        logger.info("job_review_cleared", job_id=job_id)

    def set_progress(self, job_id: str, message: str) -> None:
        """Update the human-readable progress message for a running job."""
        self._set_meta(job_id, {"message": message})



    # ── Private helpers ────────────────────────────────────────────────────────

    def _celery_status(self, job_id: str) -> JobStatusResponse:
        """
        Query Celery's result backend and translate the task state.

        Celery states → our states:
          PENDING  → PENDING  (task not yet picked up by a worker)
          STARTED  → STARTED  (worker is processing it)
          SUCCESS  → SUCCESS  (result available)
          FAILURE  → FAILURE  (exception was raised)
          REVOKED  → FAILURE  (task was cancelled)
        """
        from app.core.celery_app import celery_app

        task_result = celery_app.AsyncResult(job_id)
        state = task_result.state

        # Map Celery state → our response state
        if state == "SUCCESS":
            return JobStatusResponse(
                job_id=job_id,
                state="SUCCESS",
                progress_message="Complete.",
                result=task_result.result,
            )
        elif state == "FAILURE":
            error_str = str(task_result.result) if task_result.result else "Unknown error"
            logger.warning("job_failed", job_id=job_id, error=error_str)
            return JobStatusResponse(
                job_id=job_id,
                state="FAILURE",
                error=error_str,
            )
        elif state == "STARTED":
            meta = self._get_meta(job_id)
            return JobStatusResponse(
                job_id=job_id,
                state="STARTED",
                progress_message=meta.get("message", "Processing…"),
            )
        else:
            # PENDING, RECEIVED, RETRY all map to PENDING for simplicity
            return JobStatusResponse(
                job_id=job_id,
                state="PENDING",
                progress_message="Queued. Processing will start shortly.",
            )
        

    def _get_meta(self, job_id: str) -> dict:
        """Read the JSON metadata dict for a job. Returns {} if absent."""
        r = self._get_redis()
        raw = r.get(self._key_meta(job_id))
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        return {}

    def _set_meta(self, job_id: str, data: dict) -> None:
        """Merge new data into the job's metadata dict."""
        r = self._get_redis()
        existing = self._get_meta(job_id)
        existing.update(data)
        r.set(self._key_meta(job_id), json.dumps(existing), ex=JOB_META_TTL)

    def _get_redis(self):
        """Lazy Redis client construction."""
        if self._redis is None:
            import redis as redis_lib
            self._redis = redis_lib.from_url(
                self._redis_url,
                decode_responses=False,
            )
        return self._redis

    @staticmethod
    def _key_state(job_id: str) -> str:
        return f"job:{job_id}:state"

    @staticmethod
    def _key_meta(job_id: str) -> str:
        return f"job:{job_id}:meta"
