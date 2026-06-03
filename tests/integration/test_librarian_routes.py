# ─────────────────────────────────────────────────────────────────────────────
# tests/integration/test_librarian_routes.py
#
# Integration tests for the Librarian API routes.
#
# These tests exercise the full HTTP layer – routing, request parsing,
# response serialisation, and exception handling – using FastAPI's TestClient.
# All ML services are replaced by stubs from conftest.py, so no models or
# databases are needed.
#
# Test flow mirrors the real librarian workflow:
#   1. Upload images → receive job_id
#   2. Fetch review queue for the job
#   3. Submit corrections
#   4. Poll for final catalog
# ─────────────────────────────────────────────────────────────────────────────

import io
import uuid
import pytest
from fastapi.testclient import TestClient


class TestLibrarianCatalogUpload:
    """Tests for POST /librarian/catalog (image upload + job queuing)."""

    def test_valid_upload_returns_202_and_job_id(self, client, sample_image_bytes):
        """A valid image upload must return 202 with a job_id."""
        response = client.post(
            "/librarian/catalog",
            files=[("images", ("shelf.png", io.BytesIO(sample_image_bytes), "image/png"))],
        )

        assert response.status_code == 202
        body = response.json()
        assert "job_id" in body
        assert body["status"] == "queued"
        # Verify job_id is a valid UUID
        uuid.UUID(body["job_id"])  # Raises ValueError if not a valid UUID

    def test_multiple_images_accepted(self, client, sample_image_bytes):
        """Multiple image files must all be accepted in a single request."""
        response = client.post(
            "/librarian/catalog",
            files=[
                ("images", ("shelf1.png", io.BytesIO(sample_image_bytes), "image/png")),
                ("images", ("shelf2.png", io.BytesIO(sample_image_bytes), "image/png")),
                ("images", ("shelf3.png", io.BytesIO(sample_image_bytes), "image/png")),
            ],
        )

        assert response.status_code == 202

    def test_unsupported_file_type_returns_400(self, client):
        """Uploading a non-image file must return 400."""
        response = client.post(
            "/librarian/catalog",
            files=[("images", ("doc.pdf", io.BytesIO(b"fake pdf"), "application/pdf"))],
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_no_files_returns_422(self, client):
        """A request with no files must fail FastAPI validation with 422."""
        response = client.post("/librarian/catalog")

        assert response.status_code == 422


class TestLibrarianReviewQueue:
    """Tests for GET /librarian/review/{job_id} and POST /librarian/review/{job_id}."""

    def test_get_review_returns_404_for_unknown_job(self, client):
        """Fetching review queue for a non-existent job must return 404."""
        response = client.get("/librarian/review/does-not-exist")

        assert response.status_code == 404
        assert response.json()["error"] == "review_session_not_found"

    def test_get_review_returns_flagged_books(self, client, settings, sample_image_bytes):
        """
        After seeding the review queue, GET /review/{job_id} must return
        the flagged books with crop image URLs.
        """
        from app.models.book import OCRValidatedResult
        from app.services.review_queue import RedisReviewQueueService

        # Seed a review session directly (simulates pipeline completion)
        job_id = str(uuid.uuid4())
        flagged = OCRValidatedResult(
            book_id=str(uuid.uuid4()),
            crop_image_path="some_crop.jpg",
            ori_ocr_ext_spine_txt="Blurry Txt",
            title="Blurry Txt",
            author="Unknwn",
            ocr_confidence=0.40,
            flagged_for_review=True,
        )
        svc = RedisReviewQueueService(redis_url=settings.redis_url, ttl_seconds=settings.review_session_ttl_seconds)
        svc.create_session(job_id, [flagged])

        response = client.get(f"/librarian/review/{job_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["job_id"] == job_id
        assert body["total_flagged"] == 1
        assert len(body["items"]) == 1

        item = body["items"][0]
        assert item["book_id"] == flagged.book_id
        assert item["ocr_title"] == "Blurry Txt"
        assert item["confidence"] == 0.40
        # Crop image URL must be a /uploads/ path
        assert item["crop_image_url"].startswith("/uploads/")

    def test_submit_corrections_success(self, client, settings):
        """
        A valid correction submission for all flagged books must return 200
        and a STARTED status.
        """
        job_id = str(uuid.uuid4())
        book_id = str(uuid.uuid4())

        # Seed the review queue with one flagged book
        from app.models.book import OCRValidatedResult
        from app.services.review_queue import RedisReviewQueueService

        flagged = OCRValidatedResult(
            book_id=book_id,
            crop_image_path="crop.jpg",
            ori_ocr_ext_spine_txt="Grbld Nme",
            title="Grbld Nme",
            author="??",
            ocr_confidence=0.35,
            flagged_for_review=True,
        )
        svc = RedisReviewQueueService(redis_url=settings.redis_url, ttl_seconds=settings.review_session_ttl_seconds)
        svc.create_session(job_id, [flagged])

        # Submit correction
        response = client.post(
            f"/librarian/review/{job_id}",
            json={
                "corrections": [
                    {
                        "book_id": book_id,
                        "corrected_title": "Clean Book Title",
                        "corrected_author": "Clean Author Name",
                    }
                ]
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["state"] == "STARTED"
        assert body["job_id"] == job_id

        # Verify the correction was stored
        assert stub_review_queue.is_complete(job_id) is True

    def test_incomplete_submission_returns_422(self, client, settings):
        """
        Submitting corrections that omit some flagged book_ids must return 422.
        """
        job_id = str(uuid.uuid4())
        book_id_1 = str(uuid.uuid4())
        book_id_2 = str(uuid.uuid4())

        from app.models.book import OCRValidatedResult
        from app.services.review_queue import RedisReviewQueueService

        svc = RedisReviewQueueService(redis_url=settings.redis_url, ttl_seconds=settings.review_session_ttl_seconds)
        svc.create_session(job_id, [
            OCRValidatedResult(book_id=book_id_1, crop_image_path="a.jpg", ori_ocr_ext_spine_txt="A", title="A", author="", ocr_confidence=0.3, flagged_for_review=True),
            OCRValidatedResult(book_id=book_id_2, crop_image_path="b.jpg", ori_ocr_ext_spine_txt="B", title="B", author="", ocr_confidence=0.2, flagged_for_review=True),
        ])

        # Only submit correction for book 1, omit book 2
        response = client.post(
            f"/librarian/review/{job_id}",
            json={
                "corrections": [
                    {"book_id": book_id_1, "corrected_title": "Book A", "corrected_author": "Auth A"}
                ]
            },
        )

        assert response.status_code == 422
        body = response.json()
        assert body["detail"]["error"] == "incomplete_submission"
        assert book_id_2 in body["detail"]["missing_book_ids"]


class TestLibrarianCatalogPoll:
    """Tests for GET /librarian/catalog/{job_id}."""

    def test_poll_returns_job_status(self, client):
        """Polling a job must return a valid status response."""
        response = client.get(f"/librarian/catalog/{uuid.uuid4()}")

        assert response.status_code == 200
        body = response.json()
        assert "job_id" in body
        assert "state" in body
        assert body["state"] in {"PENDING", "STARTED", "AWAITING_REVIEW", "SUCCESS", "FAILURE"}
