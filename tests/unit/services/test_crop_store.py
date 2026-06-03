# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_crop_store.py
#
# Unit tests for DiskCropStore and MemoryCropStore.
# ─────────────────────────────────────────────────────────────────────────────

import uuid
from pathlib import Path
import tempfile

import pytest
from PIL import Image as PILImage

from app.services.crop_store import DiskCropStore, MemoryCropStore


def _solid_image(w=60, h=200) -> PILImage.Image:
    return PILImage.new("RGB", (w, h), color=(200, 100, 50))


class TestMemoryCropStore:
    """Tests for the in-memory store (used in tests as a fast substitute)."""

    def test_save_returns_relative_path(self):
        store = MemoryCropStore()
        job_id = str(uuid.uuid4())
        path = store.save(_solid_image(), job_id=job_id, book_index=0)

        assert path == f"crops/{job_id}/book_0.jpg"

    def test_save_multiple_books_unique_paths(self):
        store = MemoryCropStore()
        job_id = str(uuid.uuid4())

        paths = [store.save(_solid_image(), job_id, i) for i in range(3)]

        assert len(set(paths)) == 3    # All paths are unique

    def test_saved_bytes_are_valid_jpeg(self):
        """Bytes stored in MemoryCropStore should decode as a valid JPEG."""
        import io
        store = MemoryCropStore()
        job_id = str(uuid.uuid4())
        path = store.save(_solid_image(), job_id, 0)

        raw = store.get_bytes(path)
        assert raw is not None

        recovered = PILImage.open(io.BytesIO(raw))
        assert recovered.format == "JPEG"

    def test_saved_paths_property(self):
        store = MemoryCropStore()
        job_id = str(uuid.uuid4())

        for i in range(5):
            store.save(_solid_image(), job_id, i)

        assert len(store.saved_paths) == 5

    def test_get_bytes_returns_none_for_unknown_path(self):
        store = MemoryCropStore()
        assert store.get_bytes("crops/nonexistent/book_0.jpg") is None


class TestDiskCropStore:
    """Tests for the disk-backed store."""

    def test_save_creates_file_on_disk(self, tmp_path):
        """Saved crops should actually exist as files."""
        store = DiskCropStore(upload_dir=str(tmp_path))
        job_id = str(uuid.uuid4())

        rel_path = store.save(_solid_image(), job_id=job_id, book_index=0)
        full_path = tmp_path / rel_path

        assert full_path.exists()
        assert full_path.stat().st_size > 0

    def test_save_creates_job_directory(self, tmp_path):
        """A per-job subdirectory should be created automatically."""
        store = DiskCropStore(upload_dir=str(tmp_path))
        job_id = str(uuid.uuid4())

        store.save(_solid_image(), job_id=job_id, book_index=0)

        job_dir = tmp_path / "crops" / job_id
        assert job_dir.is_dir()

    def test_relative_path_format(self, tmp_path):
        """Returned relative path must follow the crops/{job_id}/book_{n}.jpg pattern."""
        store = DiskCropStore(upload_dir=str(tmp_path))
        job_id = str(uuid.uuid4())

        rel_path = store.save(_solid_image(), job_id=job_id, book_index=2)

        assert rel_path == f"crops/{job_id}/book_2.jpg"

    def test_get_full_path_resolves_correctly(self, tmp_path):
        """get_full_path should prepend the upload_dir to a relative path."""
        store = DiskCropStore(upload_dir=str(tmp_path))
        rel = "crops/abc123/book_0.jpg"

        full = store.get_full_path(rel)

        assert full == tmp_path / rel

    def test_multiple_books_same_job(self, tmp_path):
        """Multiple crops for the same job should all be saved correctly."""
        store = DiskCropStore(upload_dir=str(tmp_path))
        job_id = str(uuid.uuid4())

        paths = [store.save(_solid_image(), job_id=job_id, book_index=i) for i in range(4)]

        # All files should exist
        for rel_path in paths:
            assert (tmp_path / rel_path).exists()

    def test_save_is_idempotent_for_same_index(self, tmp_path):
        """
        Saving the same job_id + book_index twice should overwrite (not error).
        This handles retry scenarios gracefully.
        """
        store = DiskCropStore(upload_dir=str(tmp_path))
        job_id = str(uuid.uuid4())
        img1 = PILImage.new("RGB", (60, 200), color=(255, 0, 0))
        img2 = PILImage.new("RGB", (60, 200), color=(0, 255, 0))

        path1 = store.save(img1, job_id=job_id, book_index=0)
        path2 = store.save(img2, job_id=job_id, book_index=0)

        assert path1 == path2   # Same path
        # The file should contain the second image (overwritten)
        full = tmp_path / path2
        assert full.exists()
