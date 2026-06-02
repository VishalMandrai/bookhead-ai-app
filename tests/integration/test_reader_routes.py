# ─────────────────────────────────────────────────────────────────────────────
# tests/integration/test_reader_routes.py
#
# Integration tests for the Reader API routes.
# ─────────────────────────────────────────────────────────────────────────────

import io
import uuid
import pytest


class TestReaderAnalyze:
    """Tests for POST /reader/analyze."""

    def test_valid_upload_returns_202(self, client, sample_image_bytes):
        """A valid image upload with preferences must return 202 with a job_id."""
        response = client.post(
            "/reader/analyze",
            files=[("images", ("shelf.jpg", io.BytesIO(sample_image_bytes), "image/jpeg"))],
            data={
                "preferred_genres": "fiction,mystery",
                "mood": "something uplifting",
                "preferred_length": "medium",
                "max_recommendations": "5",
            },
        )

        assert response.status_code == 202
        body = response.json()
        assert "job_id" in body
        # Must be a valid UUID
        uuid.UUID(body["job_id"])

    def test_upload_without_preferences_uses_defaults(self, client, sample_image_bytes):
        """Omitting preferences must not cause a validation error."""
        response = client.post(
            "/reader/analyze",
            files=[("images", ("shelf.png", io.BytesIO(sample_image_bytes), "image/png"))],
        )

        assert response.status_code == 202

    def test_unsupported_file_returns_400(self, client):
        """A non-image file must be rejected with 400."""
        response = client.post(
            "/reader/analyze",
            files=[("images", ("notes.txt", io.BytesIO(b"text content"), "text/plain"))],
        )

        assert response.status_code == 400

    def test_no_images_returns_422(self, client):
        """A request with no files must fail FastAPI validation."""
        response = client.post("/reader/analyze")
        assert response.status_code == 422


class TestReaderPoll:
    """Tests for GET /reader/{job_id}."""

    def test_poll_returns_valid_status(self, client):
        """Polling any job_id must return a structured status response."""
        fake_job_id = str(uuid.uuid4())
        response = client.get(f"/reader/{fake_job_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["job_id"] == fake_job_id
        assert body["state"] in {"PENDING", "STARTED", "SUCCESS", "FAILURE"}
