# ─────────────────────────────────────────────────────────────────────────────
# app/services/crop_store.py
#
# Handles all file I/O for cropped images.
#
# SOLID note (Single Responsibility):
#   The detection service's job is to find book spines.
#   THIS module's job is to persist crop images to disk and return their paths.
#   By separating these concerns, we can swap the storage backend, any time we want, 
#   (e.g. to S3 or GCS) without touching the detection logic at all.
#
# SOLID note (Dependency Inversion):
#   The CropStore is injected into FasterRCNNDetectionService, not hard-coded.
#   Tests can pass a MemoryCropStore (defined below) to avoid touching disk.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image as PILImage

from app.core.logging import get_logger
from app.services.image_utils import image_to_bytes

logger = get_logger(__name__)


# ── Abstract contract ──────────────────────────────────────────────────────────

class BaseCropStore(ABC):
    """
    Contract for persisting and retrieving crop images.

    Abstracting storage behind this interface means the detection service
    stays testable and cloud-portable without any code changes.
    """

    @abstractmethod
    def save(
        self,
        crop: PILImage.Image,
        job_id: str,
        book_index: int,
    ) -> str:
        """
        Persist a single crop image and return its relative path.

        Args:
            crop:        The cropped PIL Image to store.
            job_id:      The job this crop belongs to (used for namespacing).
            book_index:  Zero-based index of this book in the detection results
                         (used to generate a deterministic filename).

        Returns:
            A relative path string that can be served as a static file URL,
            e.g. "crops/abc123/book_0.jpg".
        """
        ...

    @abstractmethod
    def get_full_path(self, relative_path: str) -> Path:
        """
        Resolve a relative crop path to an absolute filesystem path.

        Used by the OCR service to open the file for processing.
        """
        ...


# ── Disk implementation ────────────────────────────────────────────────────────

class DiskCropStore(BaseCropStore):
    """
    Saves crop images as JPEG files under `{upload_dir}/crops/{job_id}/`.

    Directory layout:
        /app/uploads/
          crops/
            {job_id}/        ← one sub-directory per job
              book_0.jpg
              book_1.jpg
              ...

    The relative paths returned (e.g. "crops/{job_id}/book_0.jpg") are
    served by FastAPI's StaticFiles mount at /uploads/, so a relative path
    can be turned into a public URL as: /uploads/{relative_path}.
    """

    def __init__(self, upload_dir: str) -> None:
        """
        Args:
            upload_dir: Absolute path to the root uploads directory.
                        Comes from Settings.upload_dir.
        """
        self._root = Path(upload_dir)

    def save(
        self,
        crop: PILImage.Image,
        job_id: str,
        book_index: int,
        image_idx: int = 0,
    ) -> str:
        """
        Saves the crop as a JPEG and returns its relative path.

        JPEG is chosen over PNG because:
          - Smaller file size (important: a shelf may contain 30+ crops)
          - Sufficient quality for OCR at quality=92
          - Widely supported by image viewers and the browser's <img> tag
        """
        # Create the per-job directory if it doesn't exist yet
        # exist_ok=True makes this idempotent (safe to call multiple times)
        job_dir = self._root / "crops" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Deterministic filename based on detection order: book_0.jpg, book_1.jpg…
        filename = f"book_{image_idx}_{book_index}.jpg"
        full_path = job_dir / filename

        # Serialise PIL Image → JPEG bytes → write to disk
        jpeg_bytes = image_to_bytes(crop, fmt="JPEG", quality=92)
        full_path.write_bytes(jpeg_bytes)

        # Return the RELATIVE path so callers get a URL-ready string
        relative_path = f"crops/{job_id}/{filename}"

        logger.debug(
            "crop_saved",
            job_id=job_id,
            book_index=book_index,
            path=relative_path,
            size_bytes=len(jpeg_bytes),
        )

        return relative_path


    def get_full_path(self, relative_path: str) -> Path:
        """Resolve a relative crop path to its absolute location on disk."""
        return self._root / relative_path


# ── In-memory implementation (for tests) ──────────────────────────────────────

class MemoryCropStore(BaseCropStore):
    """
    Stores crop images in a dict instead of on disk.

    Used in unit tests to avoid any filesystem I/O.
    The internal store can be inspected by tests to assert what was saved.
    """

    def __init__(self) -> None:
        # Maps relative_path → raw JPEG bytes
        self._store: dict[str, bytes] = {}

    def save(
        self,
        crop: PILImage.Image,
        job_id: str,
        book_index: int,
    ) -> str:
        relative_path = f"crops/{job_id}/book_{book_index}.jpg"
        self._store[relative_path] = image_to_bytes(crop, fmt="JPEG", quality=92)
        return relative_path

    def get_full_path(self, relative_path: str) -> Path:
        # In memory mode, return a sentinel path – callers that need to open
        # the file should use get_bytes() instead during testing.
        return Path("/memory") / relative_path

    def get_bytes(self, relative_path: str) -> bytes | None:
        """Test helper: retrieve raw bytes for a saved crop."""
        return self._store.get(relative_path)

    @property
    def saved_paths(self) -> list[str]:
        """Test helper: list all relative paths saved so far."""
        return list(self._store.keys())
