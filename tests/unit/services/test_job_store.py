# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_job_store.py
#
# Unit tests for JobStore — verifies state resolution logic without
# connecting to real Redis or Celery.
# ─────────────────────────────────────────────────────────────────────────────

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.models.response import JobStatusResponse
from app.services.job_store import JobStore


def _make_store() -> tuple[JobStore, MagicMock]:
    """Return (store, fake_redis) with a pre-injected mock Redis client."""
    store = JobStore(redis_url="redis://fake")
    fake_redis = MagicMock()
    fake_redis.get.return_value = None          # Default: no custom state
    store._redis = fake_redis
    return store, fake_redis


class TestJobStoreGetStatus:

    def test_awaiting_review_state_returned_when_flag_set(self):
        store, fake_redis = _make_store()
        job_id = str(uuid.uuid4())

        def redis_get(key):
            if f"job:{job_id}:state" in key:
                return b"AWAITING_REVIEW"
            if f"job:{job_id}:meta" in key:
                return json.dumps({"message": "2 books need review."}).encode()
            return None
        fake_redis.get.side_effect = redis_get

        result = store.get_status(job_id)
        assert result.state == "AWAITING_REVIEW"

    def test_awaiting_review_includes_progress_message(self):
        store, fake_redis = _make_store()
        job_id = str(uuid.uuid4())

        def redis_get(key):
            if ":state" in key: return b"AWAITING_REVIEW"
            if ":meta" in key:  return json.dumps({"message": "3 books flagged."}).encode()
            return None
        fake_redis.get.side_effect = redis_get

        result = store.get_status(job_id)
        assert "3 books flagged" in result.progress_message

    def test_celery_pending_state_returned(self):
        store, fake_redis = _make_store()
        job_id = str(uuid.uuid4())
        fake_redis.get.return_value = None  # No custom state

        mock_async_result = MagicMock()
        mock_async_result.state = "PENDING"

        with patch("app.services.job_store.celery_app") as mock_celery:
            mock_celery.AsyncResult.return_value = mock_async_result
            result = store.get_status(job_id)

        assert result.state == "PENDING"

    def test_celery_success_state_includes_result(self):
        store, fake_redis = _make_store()
        job_id = str(uuid.uuid4())
        fake_redis.get.return_value = None

        mock_async_result = MagicMock()
        mock_async_result.state = "SUCCESS"
        mock_async_result.result = {"total_books_detected": 5}

        with patch("app.services.job_store.celery_app") as mock_celery:
            mock_celery.AsyncResult.return_value = mock_async_result
            result = store.get_status(job_id)

        assert result.state == "SUCCESS"
        assert result.result == {"total_books_detected": 5}

    def test_celery_failure_state_includes_error(self):
        store, fake_redis = _make_store()
        job_id = str(uuid.uuid4())
        fake_redis.get.return_value = None

        mock_async_result = MagicMock()
        mock_async_result.state = "FAILURE"
        mock_async_result.result = Exception("Model crashed")

        with patch("app.services.job_store.celery_app") as mock_celery:
            mock_celery.AsyncResult.return_value = mock_async_result
            result = store.get_status(job_id)

        assert result.state == "FAILURE"
        assert result.error is not None


class TestJobStoreSetState:

    def test_set_awaiting_review_writes_redis_key(self):
        store, fake_redis = _make_store()
        job_id = str(uuid.uuid4())
        fake_redis.get.return_value = None

        store.set_awaiting_review(job_id, num_flagged=3)

        # Verify Redis set was called with the state key
        set_calls = [c for c in fake_redis.set.call_args_list]
        state_keys = [c[0][0] for c in set_calls if ":state" in str(c[0][0])]
        assert len(state_keys) > 0

    def test_clear_awaiting_review_deletes_key(self):
        store, fake_redis = _make_store()
        job_id = str(uuid.uuid4())

        store.clear_awaiting_review(job_id)

        fake_redis.delete.assert_called_once_with(f"job:{job_id}:state")

    def test_set_progress_writes_meta_key(self):
        store, fake_redis = _make_store()
        job_id = str(uuid.uuid4())
        fake_redis.get.return_value = None  # No existing meta

        store.set_progress(job_id, "Running OCR…")

        meta_calls = [c for c in fake_redis.set.call_args_list
                      if ":meta" in str(c[0][0])]
        assert len(meta_calls) > 0
