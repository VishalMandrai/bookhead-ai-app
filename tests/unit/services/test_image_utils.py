# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_image_utils.py
#
# Unit tests for app/services/image_utils.py
#
# All tests use synthetic PIL Images created in-memory — no real images needed.
# These tests are extremely fast (no model inference, no disk I/O).
# ─────────────────────────────────────────────────────────────────────────────

import io
import pytest
from PIL import Image as PILImage

from app.services.image_utils import (
    crop_bounding_box,
    image_to_bytes,
    load_image_from_bytes,
    preprocess_crop_for_ocr,
    resize_for_inference,
    rotate_spine_crop,
)


def _make_image(width: int, height: int, color=(128, 64, 32)) -> PILImage.Image:
    """Create a solid-colour RGB test image."""
    return PILImage.new("RGB", (width, height), color=color)


def _image_to_png_bytes(image: PILImage.Image) -> bytes:
    """Encode a PIL Image as PNG bytes (lossless – for round-trip tests)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class TestLoadImageFromBytes:

    def test_loads_valid_png(self):
        """A valid PNG should decode to an RGB PIL Image."""
        img = _make_image(100, 200)
        raw = _image_to_png_bytes(img)

        result = load_image_from_bytes(raw)

        assert isinstance(result, PILImage.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 200)

    def test_converts_rgba_to_rgb(self):
        """An RGBA image (e.g. PNG with transparency) must be converted to RGB."""
        img = PILImage.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = load_image_from_bytes(buf.getvalue())

        assert result.mode == "RGB"

    def test_converts_grayscale_to_rgb(self):
        """A grayscale image must be converted to RGB (3-channel) for the model."""
        img = PILImage.new("L", (80, 80), color=128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = load_image_from_bytes(buf.getvalue())

        assert result.mode == "RGB"

    def test_raises_on_invalid_bytes(self):
        """Garbage bytes should raise ValueError."""
        with pytest.raises(ValueError, match="Could not decode"):
            load_image_from_bytes(b"this is not an image")


class TestResizeForInference:

    def test_small_image_unchanged(self):
        """
        An image already within the target range should not be resized.
        (width=1000, height=800 → longer=1000 ≤ 1333, shorter=800 ≥ 800)
        """
        img = _make_image(1000, 800)
        result = resize_for_inference(img, max_side=1333, min_side=800)
        assert result.size == (1000, 800)

    def test_large_image_downscaled(self):
        """A 4K image (3840×2160) should be scaled down so longer side = 1333."""
        img = _make_image(3840, 2160)
        result = resize_for_inference(img, max_side=1333, min_side=800)

        longer = max(result.size)
        assert longer <= 1333

    def test_aspect_ratio_preserved(self):
        """After resizing, the aspect ratio should be preserved within 1%."""
        img = _make_image(4000, 2000)   # 2:1 ratio
        result = resize_for_inference(img)

        w, h = result.size
        original_ratio = 4000 / 2000
        result_ratio   = w / h
        assert abs(result_ratio - original_ratio) / original_ratio < 0.01

    def test_output_is_pil_image(self):
        img = _make_image(2000, 1500)
        result = resize_for_inference(img)
        assert isinstance(result, PILImage.Image)


class TestCropBoundingBox:

    def test_crops_correct_region(self):
        """
        Cropping a 100×100 region from a 500×500 image should give a
        (100+2*padding)×(100+2*padding) image (clamped to image bounds).
        """
        img = _make_image(500, 500)
        crop = crop_bounding_box(img, box=(50, 50, 150, 150), padding_px=0)
        assert crop.size == (100, 100)

    def test_padding_applied(self):
        """Padding should expand the crop by 2*padding on each axis."""
        img = _make_image(500, 500)
        crop = crop_bounding_box(img, box=(50, 50, 150, 150), padding_px=10)
        # Width: 150-50 + 2*10 = 120; Height: same
        assert crop.size == (120, 120)

    def test_padding_clamped_at_edges(self):
        """
        When the bounding box is near the image edge, padding should be
        clamped so we don't go out of bounds.
        """
        img = _make_image(200, 200)
        # Box touching the top-left corner
        crop = crop_bounding_box(img, box=(0, 0, 50, 50), padding_px=20)
        # x_min clamped to 0, y_min clamped to 0; only right/bottom padded
        assert crop.size[0] <= 70   # 50 + 20, but left already at 0
        assert crop.size[1] <= 70

    def test_returns_pil_image(self):
        img = _make_image(300, 300)
        crop = crop_bounding_box(img, box=(10, 10, 100, 100), padding_px=0)
        assert isinstance(crop, PILImage.Image)


class TestPreprocessCropForOcr:

    def test_output_is_rgb(self):
        """Preprocessing should always return an RGB image."""
        img = _make_image(80, 200)
        result = preprocess_crop_for_ocr(img)
        assert result.mode == "RGB"

    def test_output_same_size_as_input(self):
        """Preprocessing must not change the image dimensions."""
        img = _make_image(60, 180)
        result = preprocess_crop_for_ocr(img)
        assert result.size == img.size

    def test_handles_grayscale_input(self):
        """A greyscale crop should be converted to RGB without errors."""
        img = PILImage.new("L", (60, 180), color=100)
        result = preprocess_crop_for_ocr(img)
        assert result.mode == "RGB"

    def test_returns_pil_image(self):
        img = _make_image(60, 200)
        result = preprocess_crop_for_ocr(img)
        assert isinstance(result, PILImage.Image)


class TestRotateSpineCrop:

    def test_90_degree_rotation_swaps_dimensions(self):
        """
        A 90° rotation of a non-square image should swap width and height
        (because expand=True is used).
        """
        img = _make_image(60, 200)   # portrait (narrow & tall)
        rotated = rotate_spine_crop(img, angle_degrees=90)
        # After 90° clockwise rotation with expand, dimensions swap
        assert rotated.size[0] > rotated.size[1]   # now landscape

    def test_zero_degree_rotation_unchanged(self):
        """A 0° rotation should return an image with the same dimensions."""
        img = _make_image(80, 200)
        rotated = rotate_spine_crop(img, angle_degrees=0)
        assert rotated.size == img.size

    def test_returns_pil_image(self):
        img = _make_image(60, 200)
        result = rotate_spine_crop(img, angle_degrees=15)
        assert isinstance(result, PILImage.Image)


class TestImageToBytes:

    def test_jpeg_output_is_valid_image(self):
        """Bytes from image_to_bytes should re-decode as a valid PIL Image."""
        img = _make_image(100, 100)
        raw = image_to_bytes(img, fmt="JPEG", quality=85)

        recovered = PILImage.open(io.BytesIO(raw))
        assert recovered.format == "JPEG"

    def test_png_output_is_valid_image(self):
        img = _make_image(100, 100)
        raw = image_to_bytes(img, fmt="PNG")

        recovered = PILImage.open(io.BytesIO(raw))
        assert recovered.format == "PNG"

    def test_jpeg_smaller_than_png(self):
        """
        For a photographic image, JPEG should produce smaller output than PNG.
        (Both encode the same content, so the size difference is purely format.)
        """
        # Use a gradient image that compresses differently per format
        img = PILImage.new("RGB", (200, 200))
        pixels = img.load()
        for x in range(200):
            for y in range(200):
                pixels[x, y] = (x, y, (x + y) % 256)

        jpeg_bytes = image_to_bytes(img, fmt="JPEG", quality=85)
        png_bytes  = image_to_bytes(img, fmt="PNG")

        assert len(jpeg_bytes) < len(png_bytes)
