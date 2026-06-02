# ─────────────────────────────────────────────────────────────────────────────
# app/services/image_saver.py
#
# Handles saving uploaded images from FastAPI UploadFile objects to disk.
#
# WHY A SEPARATE MODULE?
#   The route handlers receive FastAPI UploadFile objects (async file-like
#   objects). The Celery pipelines work with file paths (strings), because
#   Celery tasks are separate processes that cannot share in-memory objects.
#
#   This module bridges that gap: routes call save_uploads() to persist all
#   UploadFile bytes to disk, then pass the resulting file paths to the
#   Celery task via the message broker.
#
# SOLID note (Single Responsibility):
#   This module owns only "UploadFile → disk path". It knows nothing about
#   detection, OCR, or pipelines.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from app.core.logging import get_logger

logger = get_logger(__name__)


async def save_uploads(
    images: list[UploadFile],
    upload_dir: str,
    job_id: str,
) -> list[str]:
    """
    Persist a list of uploaded images to disk and return their file paths.

    Files are saved under {upload_dir}/originals/{job_id}/ with their
    original filenames (sanitised). Each call is async so it doesn't block
    the FastAPI event loop while writing potentially large image files.

    Args:
        images:     List of FastAPI UploadFile objects from the request.
        upload_dir: Root directory for all uploads (from Settings.upload_dir).
        job_id:     Job identifier used to namespace the saved files.

    Returns:
        List of absolute file path strings, one per uploaded image,
        in the same order as the input list.
    """
    # Create per-job directory for original uploaded images
    job_dir = Path(upload_dir) / "originals" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []

    for idx, image in enumerate(images):
        # Sanitise filename: strip path components, fall back to index-based name
        safe_name = _safe_filename(image.filename, idx)
        dest = job_dir / safe_name                         # Image location on saving

        # Read the uploaded bytes and write them to disk asynchronously
        content = await image.read()
        async with aiofiles.open(dest, "wb") as f:
            await f.write(content)

        saved_paths.append(str(dest))

        logger.debug(
            "upload_saved",
            job_id=job_id,
            filename=safe_name,
            size_bytes=len(content),
            path=str(dest),
        )

    logger.info(
        "uploads_saved",
        job_id=job_id,
        count=len(saved_paths),
    )

    return saved_paths


def _safe_filename(filename: str | None, fallback_index: int) -> str:
    """
    Return a safe filename by stripping directory components and special chars.

    Args:
        filename:       Original filename from UploadFile (may be None).
        fallback_index: Used to generate a unique name if filename is absent.

    Returns:
        A sanitised filename string (no path separators, no leading dots).
    """
    if not filename:
        return f"image_{fallback_index}_{uuid.uuid4().hex[:8]}.jpg"

    # Strip any directory components the client might have included
    name = Path(filename).name

    # Replace characters that are problematic in filenames
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)

    # Prevent hidden files (leading dot)
    if safe.startswith("."):
        safe = f"img_{safe}"

    return safe or f"image_{fallback_index}.jpg"
