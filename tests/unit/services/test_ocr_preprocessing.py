# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_ocr_preprocessing.py
#
# Unit tests for app/services/ocr_preprocessing.py
# All tests use synthetic PIL Images — no real book images needed.
# ─────────────────────────────────────────────────────────────────────────────

import pytest
from PIL import Image as PILImage

from app.services.ocr_preprocessing import (
    MIN_OCR_WIDTH_PX,
    TARGET_OCR_WIDTH_PX,
    OCRSpinePreprocessor,
    apply_adaptive_threshold,
    normalise_orientation,
    to_rgb_for_easyocr,
    upscale_if_narrow,
)


def _spine(w: int = 40, h: int = 300, color=(80, 80, 80)) -> PILImage.Image:
    """Create a synthetic spine crop."""
    return PILImage.new("RGB", (w, h), color=color)


class TestUpscaleIfNarrow:

    def test_narrow_crop_is_upscaled(self):
        """A crop narrower than MIN_OCR_WIDTH_PX should be upscaled."""
        img = _spine(w=30, h=200)
        result = upscale_if_narrow(img)
        assert result.width == TARGET_OCR_WIDTH_PX
        assert result.width > img.width

    def test_wide_crop_is_unchanged(self):
        """A crop wider than MIN_OCR_WIDTH_PX should not be resized."""
        img = _spine(w=200, h=400)
        result = upscale_if_narrow(img)
        assert result.size == img.size

    def test_exactly_at_threshold_is_unchanged(self):
        """A crop exactly at MIN_OCR_WIDTH_PX should not be resized."""
        img = _spine(w=MIN_OCR_WIDTH_PX, h=300)
        result = upscale_if_narrow(img)
        assert result.size == img.size

    def test_aspect_ratio_preserved_after_upscale(self):
        """After upscaling, the aspect ratio should be preserved within 2%."""
        img = _spine(w=20, h=300)
        result = upscale_if_narrow(img)
        original_ratio = img.height / img.width
        result_ratio   = result.height / result.width
        assert abs(result_ratio - original_ratio) / original_ratio < 0.02

    def test_output_is_pil_image(self):
        img = _spine(w=30, h=200)
        assert isinstance(upscale_if_narrow(img), PILImage.Image)


class TestNormaliseOrientation:

    def test_already_horizontal_image_unchanged_size(self):
        """
        A clearly horizontal image (text already runs left-to-right) should
        be returned at the same size (no rotation, or 0° / 360° rotation).
        """
        # Horizontal stripes simulate horizontal text rows
        img = PILImage.new("RGB", (300, 60), color=(255, 255, 255))
        # Draw dark horizontal bands to simulate text rows
        pixels = img.load()
        for y in range(0, 60, 10):
            for x in range(300):
                pixels[x, y] = (0, 0, 0)

        result = normalise_orientation(img)
        # After normalisation, the image should still be wider than tall
        assert result.width >= result.height

    def test_returns_pil_image(self):
        img = _spine(w=40, h=300)
        result = normalise_orientation(img)
        assert isinstance(result, PILImage.Image)

    def test_output_mode_is_rgb(self):
        img = _spine(w=40, h=300)
        result = normalise_orientation(img)
        assert result.mode == "RGB"


class TestApplyAdaptiveThreshold:

    def test_output_is_greyscale(self):
        """Adaptive threshold converts RGB to greyscale ('L' mode)."""
        img = _spine(w=60, h=200)
        result = apply_adaptive_threshold(img)
        assert result.mode == "L"

    def test_output_same_dimensions(self):
        """The size should not change after thresholding."""
        img = _spine(w=60, h=200)
        result = apply_adaptive_threshold(img)
        assert result.size == img.size

    def test_pixel_values_are_binary(self):
        """Every pixel in the output should be either 0 or 255."""
        img = _spine(w=60, h=200)
        result = apply_adaptive_threshold(img)
        unique_values = set(result.getdata())
        assert unique_values.issubset({0, 255})

    def test_dark_background_image_produces_output(self):
        """A dark-background image should produce a non-empty binary image."""
        # Dark navy spine
        img = PILImage.new("RGB", (60, 200), color=(20, 20, 80))
        result = apply_adaptive_threshold(img)
        assert result.size == (60, 200)


class TestToRgbForEasyocr:

    def test_rgb_image_returned_unchanged(self):
        img = PILImage.new("RGB", (100, 100))
        result = to_rgb_for_easyocr(img)
        assert result.mode == "RGB"
        assert result is img   # Should be the exact same object (no copy made)

    def test_greyscale_converted_to_rgb(self):
        img = PILImage.new("L", (100, 100))
        result = to_rgb_for_easyocr(img)
        assert result.mode == "RGB"

    def test_rgba_converted_to_rgb(self):
        img = PILImage.new("RGBA", (100, 100))
        result = to_rgb_for_easyocr(img)
        assert result.mode == "RGB"


class TestOCRSpinePreprocessor:

    def test_prepare_returns_rgb_image(self):
        """The full pipeline must always output an RGB PIL Image."""
        preprocessor = OCRSpinePreprocessor()
        img = _spine(w=40, h=300)
        result = preprocessor.prepare(img)
        assert isinstance(result, PILImage.Image)
        assert result.mode == "RGB"

    def test_prepare_upscales_narrow_crops(self):
        """A narrow crop (20px wide) should be wider after prepare()."""
        preprocessor = OCRSpinePreprocessor(
            upscale=True,
            fix_orientation=False,
            adaptive_threshold=False,
        )
        img = _spine(w=20, h=300)
        result = preprocessor.prepare(img)
        assert result.width > img.width

    def test_prepare_with_upscale_disabled_does_not_resize(self):
        """With upscale=False, a narrow crop should not be resized."""
        preprocessor = OCRSpinePreprocessor(
            upscale=False,
            fix_orientation=False,
            adaptive_threshold=False,
        )
        img = _spine(w=20, h=300)
        result = preprocessor.prepare(img)
        assert result.width == 20

    def test_all_steps_disabled_returns_rgb_unchanged_size(self):
        """With all steps disabled, prepare() should still return RGB at original size."""
        preprocessor = OCRSpinePreprocessor(
            upscale=False,
            fix_orientation=False,
            adaptive_threshold=False,
        )
        img = _spine(w=100, h=300)
        result = preprocessor.prepare(img)
        assert result.mode == "RGB"
        assert result.size == img.size

    def test_prepare_accepts_greyscale_input(self):
        """The preprocessor should handle greyscale inputs without crashing."""
        preprocessor = OCRSpinePreprocessor()
        img = PILImage.new("L", (40, 300), color=128)
        result = preprocessor.prepare(img)
        assert result.mode == "RGB"
