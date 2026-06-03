# ─────────────────────────────────────────────────────────────────────────────
# tests/integration/test_routes_phase6.py
#
# Integration tests for the Phase 6 wired routes.
# Celery dispatch is mocked — no actual worker needed.
# ─────────────────────────────────────────────────────────────────────────────

import io
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.book import BookRecord
from app.models.response import JobStatusResponse


def _png_bytes() -> bytes:
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (100, 200), color=(120, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_book(title: str = "Dune") -> BookRecord:
    bid = str(uuid.uuid4())
    return BookRecord(
        book_id=bid, title=title, author="Author",
        crop_image_path=f"crops/job/{bid[:6]}.jpg",
        ocr_confidence=0.90, source="ocr_auto",
        genre="SCI", genre_code="SCI",
    )


class TestReaderRoutePhase6:

    def test_analyze_dispatches_celery_task(self, client, sample_image_bytes, tmp_path):
        """POST /reader/analyze must dispatch a Celery task and return 202."""
        with patch("app.pipelines.reader_pipeline.run_reader_pipeline") as mock_task, \
             patch("app.services.image_saver.save_uploads", return_value=["/tmp/img.jpg"]):
            mock_task.apply_async = MagicMock()
            response = client.post(
                "/reader/analyze",
                files=[("images", ("shelf.png", io.BytesIO(sample_image_bytes), "image/png"))],
            )
        assert response.status_code == 202
        assert "job_id" in response.json()

    def test_analyze_returns_valid_uuid_job_id(self, client, sample_image_bytes):
        with patch("app.pipelines.reader_pipeline.run_reader_pipeline") as mock_task, \
             patch("app.services.image_saver.save_uploads", return_value=["/tmp/img.jpg"]):
            mock_task.apply_async = MagicMock()
            response = client.post(
                "/reader/analyze",
                files=[("images", ("shelf.jpg", io.BytesIO(sample_image_bytes), "image/jpeg"))],
            )
        job_id = response.json()["job_id"]
        uuid.UUID(job_id)   # Raises if not valid UUID

    def test_analyze_rejects_pdf(self, client):
        response = client.post(
            "/reader/analyze",
            files=[("images", ("doc.pdf", io.BytesIO(b"fake"), "application/pdf"))],
        )
        assert response.status_code == 400

    def test_analyze_no_files_returns_422(self, client):
        response = client.post("/reader/analyze")
        assert response.status_code == 422

    def test_poll_returns_job_status(self, client):
        job_id = str(uuid.uuid4())
        with patch("app.api.routes.reader.JobStore") as MockJobStore:
            mock_store = MockJobStore.return_value
            mock_store.get_status.return_value = JobStatusResponse(
                job_id=job_id, state="PENDING",
                progress_message="Queued.",
            )
            response = client.get(f"/reader/{job_id}")
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id
        assert response.json()["state"] == "PENDING"

    def test_poll_success_state_has_result(self, client):
        job_id = str(uuid.uuid4())
        with patch("app.api.routes.reader.JobStore") as MockJobStore:
            mock_store = MockJobStore.return_value
            mock_store.get_status.return_value = JobStatusResponse(
                job_id=job_id, state="SUCCESS",
                result={"recommendations": [], "all_books": [], "total_books_detected": 0},
            )
            response = client.get(f"/reader/{job_id}")
        assert response.json()["state"] == "SUCCESS"
        assert "result" in response.json()


class TestLibrarianRoutePhase6:

    def test_catalog_dispatches_celery_task(self, client, sample_image_bytes):
        with patch("app.pipelines.librarian_pipeline.run_librarian_pipeline") as mock_task, \
             patch("app.services.image_saver.save_uploads", return_value=["/tmp/img.jpg"]):
            mock_task.apply_async = MagicMock()
            response = client.post(
                "/librarian/catalog",
                files=[("images", ("shelf.png", io.BytesIO(sample_image_bytes), "image/png"))],
            )
        assert response.status_code == 202
        assert "job_id" in response.json()

    def test_catalog_poll_returns_awaiting_review(self, client):
        job_id = str(uuid.uuid4())
        with patch("app.api.routes.librarian.JobStore") as MockJobStore:
            mock_store = MockJobStore.return_value
            mock_store.get_status.return_value = JobStatusResponse(
                job_id=job_id,
                state="AWAITING_REVIEW",
                progress_message="2 books need review.",
            )
            response = client.get(f"/librarian/catalog/{job_id}")
        assert response.json()["state"] == "AWAITING_REVIEW"

    def test_catalog_poll_returns_success(self, client):
        job_id = str(uuid.uuid4())
        with patch("app.api.routes.librarian.JobStore") as MockJobStore:
            mock_store = MockJobStore.return_value
            mock_store.get_status.return_value = JobStatusResponse(
                job_id=job_id, state="SUCCESS",
                result={"total_books": 3, "books": [], "csv_download_url": "/downloads/x"},
            )
            response = client.get(f"/librarian/catalog/{job_id}")
        assert response.json()["state"] == "SUCCESS"

    def test_submit_corrections_dispatches_resume_task(self, client, settings):
        """POST /librarian/review/{job_id} must dispatch resume_after_review."""
        job_id = str(uuid.uuid4())
        book_id = str(uuid.uuid4())
        from app.models.book import OCRValidatedResult
        from app.services.review_queue import RedisReviewQueueService

        svc = RedisReviewQueueService(redis_url=settings.redis_url, ttl_seconds=settings.review_session_ttl_seconds)
        svc.create_session(job_id, [
            OCRValidatedResult(
                book_id=book_id, crop_image_path="crop.jpg",
                ori_ocr_ext_spine_txt="Garbled", title="Garbled", author="???",
                ocr_confidence=0.30, flagged_for_review=True,
            )
        ])

        with patch("app.pipelines.librarian_pipeline.resume_after_review") as mock_resume, \
             patch("app.api.routes.librarian.JobStore"):
            mock_resume.apply_async = MagicMock()
            response = client.post(
                f"/librarian/review/{job_id}",
                json={"corrections": [
                    {"book_id": book_id, "corrected_title": "Clean Title", "corrected_author": "Clean Author"}
                ]},
            )

        assert response.status_code == 200
        assert response.json()["state"] == "STARTED"

    def test_submit_corrections_incomplete_returns_422(self, client, settings):
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
        response = client.post(
            f"/librarian/review/{job_id}",
            json={"corrections": [
                {"book_id": book_id_1, "corrected_title": "Title A", "corrected_author": "Auth A"}
                # book_id_2 missing
            ]},
        )
        assert response.status_code == 422
        assert book_id_2 in response.json()["detail"]["missing_book_ids"]

    def test_download_csv_returns_csv_when_success(self, client):
        job_id = str(uuid.uuid4())
        book = _make_book()
        with patch("app.api.routes.librarian.JobStore") as MockJobStore, \
             patch("app.api.routes.librarian.LLMCatalogService") as MockCatalog:
            mock_store = MockJobStore.return_value
            mock_store.get_status.return_value = JobStatusResponse(
                job_id=job_id, state="SUCCESS",
                result={"books": [book.model_dump(mode="json")],
                        "total_books": 1, "csv_download_url": f"/downloads/{job_id}"},
            )
            mock_catalog = MockCatalog.return_value
            mock_catalog.to_csv.return_value = "title,author\nDune,Frank Herbert\n"
            response = client.get(f"/librarian/download/{job_id}")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "Dune" in response.text

    def test_download_csv_returns_202_if_not_ready(self, client):
        job_id = str(uuid.uuid4())
        with patch("app.api.routes.librarian.JobStore") as MockJobStore:
            mock_store = MockJobStore.return_value
            mock_store.get_status.return_value = JobStatusResponse(
                job_id=job_id, state="STARTED",
            )
            response = client.get(f"/librarian/download/{job_id}")
        assert response.status_code == 202
