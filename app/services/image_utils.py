# ─────────────────────────────────────────────────────────────────────────────
# app/services/image_utils.py
#
# Pure image manipulation utilities used by the detection pipeline.
#
# SOLID note (Single Responsibility):
#   This module owns exactly one concern: PIL image transformations.
#   It has zero knowledge of ML models, file paths, job IDs, or services.
#   Every function is a pure transformation: PIL Image in → PIL Image out.
#   This makes every function trivially unit-testable with synthetic images.
#
# SOLID note (Open/Closed):
#   New preprocessing steps (e.g. histogram equalisation, sharpening) can be
#   added as new functions without modifying existing ones.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import io
from typing import Tuple

from PIL import Image as PILImage
from PIL import ImageOps, ImageFilter


# ── Type alias for readability ─────────────────────────────────────────────────
# A bounding box expressed as (x_min, y_min, x_max, y_max) pixel coordinates.
PixelBox = Tuple[int, int, int, int]


def load_image_from_bytes(data: bytes) -> PILImage.Image:
    """
    Decode raw bytes (from an HTTP upload) into a PIL Image in RGB mode.

    Always converts to RGB so downstream code never has to handle RGBA,
    palette, or greyscale modes – Faster-RCNN expects 3-channel tensors.

    Args:
        data: Raw image bytes (JPEG, PNG, WEBP, TIFF all supported).

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If the bytes cannot be decoded as a valid image.
    """
    try:
        image = PILImage.open(io.BytesIO(data))
        # Convert palette / RGBA / grayscale → RGB in one step
        return image.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not decode image bytes: {exc}") from exc


def resize_for_inference(
    image: PILImage.Image,
    max_side: int = 1333,
    min_side: int = 800,
) -> PILImage.Image:
    """
    Resize a shelf image to the size range Faster-RCNN was trained on
    (800–1333 px on the shorter/longer side) while preserving aspect ratio.

    Torchvision's GeneralizedRCNN internally resizes again, but pre-resizing
    very large images (e.g. 4K phone photos) here keeps memory usage bounded
    and speeds up preprocessing.

    Args:
        image:    Input PIL Image.
        max_side: Maximum length of the longer side after resizing.
        min_side: Minimum length of the shorter side after resizing.

    Returns:
        Resized PIL Image. Returns the original unchanged if it already fits.
    """
    w, h = image.size
    longer  = max(w, h)
    shorter = min(w, h)

    # If the image already fits within the target range, return as-is
    if longer <= max_side and shorter >= min_side:
        return image

    # Scale so that the longer side equals max_side
    scale = max_side / longer
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Use LANCZOS (high-quality downscaling) when shrinking
    return image.resize((new_w, new_h), PILImage.LANCZOS)


def crop_bounding_box(
    image: PILImage.Image,
    box: PixelBox,
    padding_px: int = 2,
) -> PILImage.Image:
    """
    Crop a single bounding box region from `image` with optional padding.

    The padding argument adds a small border around the tight bounding box so
    the OCR engine has context right to the edge of the spine text.

    Args:
        image:      The full shelf PIL Image.
        box:        (x_min, y_min, x_max, y_max) in pixel coordinates.
        padding_px: Extra pixels to add on each side (clamped to image bounds).

    Returns:
        A new PIL Image containing only the cropped region.
    """
    x_min, y_min, x_max, y_max = box
    img_w, img_h = image.size

    # Apply padding while keeping coordinates within image bounds
    x_min_padded = max(0,     x_min - padding_px)
    y_min_padded = max(0,     y_min - padding_px)
    x_max_padded = min(img_w, x_max + padding_px)
    y_max_padded = min(img_h, y_max + padding_px)

    return image.crop((x_min_padded, y_min_padded, x_max_padded, y_max_padded))


def preprocess_crop_for_ocr(crop: PILImage.Image) -> PILImage.Image:
    """
    Apply lightweight preprocessing to a book spine crop to improve OCR accuracy.

    Book spines are often:
      - Low contrast (faded covers, dark spines with dark text)
      - Slightly rotated (books not perfectly vertical on the shelf)
      - Small and blurry (from phone camera distortion)

    This function applies:
      1. Auto-contrast stretch: maps the darkest pixel to 0 and lightest to 255,
         dramatically improving text legibility on low-contrast spines.
      2. Mild unsharp mask: sharpens edges to help the OCR model find character
         boundaries. The parameters are conservative to avoid noise amplification.

    Args:
        crop: A single book-spine crop PIL Image.

    Returns:
        A preprocessed PIL Image ready for OCR. Always RGB.
    """
    # Ensure RGB (crop() returns the same mode as the source, which is always
    # RGB after load_image_from_bytes, but we guard defensively here)
    if crop.mode != "RGB":
        crop = crop.convert("RGB")

    # Step 1: Auto-contrast per-channel to maximise text/background separation
    # ImageOps.autocontrast operates on a single-channel image, so we split,
    # process each channel, then merge back.
    r, g, b = crop.split()
    r = ImageOps.autocontrast(r, cutoff=1)  # cutoff=1: ignore top/bottom 1% outliers
    g = ImageOps.autocontrast(g, cutoff=1)
    b = ImageOps.autocontrast(b, cutoff=1)
    enhanced = PILImage.merge("RGB", (r, g, b))

    # Step 2: Unsharp mask – sharpens fine details without boosting large-scale noise
    # radius=1: kernel size (small → targets fine edges like character strokes)
    # percent=120: strength (120% → subtle; 200%+ would be too aggressive)
    # threshold=3: only sharpen edges with contrast > 3 grey levels (ignores noise)
    sharpened = enhanced.filter(
        ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3)
    )

    return sharpened


def rotate_spine_crop(
    crop: PILImage.Image,
    angle_degrees: float,
) -> PILImage.Image:
    """
    Rotate a spine crop by the given angle, expanding the canvas to avoid clipping.

    Used when book spines are tilted (common on messy shelves). The rotation
    expands the canvas size so no content is clipped. Background fill is white
    (matches what OCR engines expect for padding).

    Args:
        crop:          The book spine crop to rotate.
        angle_degrees: Clockwise rotation in degrees. Positive = clockwise.

    Returns:
        Rotated PIL Image with expanded canvas.
    """
    # PIL's rotate() uses counter-clockwise angles; negate for clockwise
    return crop.rotate(
        -angle_degrees,
        expand=True,            # Expand canvas to fit the rotated content
        fillcolor=(255, 255, 255),  # White background for padding areas
        resample=PILImage.BICUBIC,  # High-quality interpolation
    )


def image_to_bytes(image: PILImage.Image, fmt: str = "JPEG", quality: int = 92) -> bytes:
    """
    Serialise a PIL Image to bytes in the given format.

    Args:
        image:   PIL Image to serialise.
        fmt:     Output format: "JPEG" (default), "PNG", "WEBP".
        quality: JPEG/WEBP quality (1–95). Ignored for PNG.

    Returns:
        Image encoded as bytes.
    """
    buf = io.BytesIO()
    save_kwargs: dict = {"format": fmt}
    if fmt in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    image.save(buf, **save_kwargs)
    return buf.getvalue()
