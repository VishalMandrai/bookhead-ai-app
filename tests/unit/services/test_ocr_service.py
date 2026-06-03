# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_ocr_service.py
#
# Unit tests for EasyOCRService.
#
# The actual EasyOCR model is NEVER loaded here.
# Instead, we:
#   1. Inject a mock _reader whose readtext() returns synthetic boxes
#   2. Write real crop images to a temp directory so _load_crop() succeeds
#   3. Inject a real SpineTextParser so parsing logic is exercised
#
# This tests the EasyOCRService orchestration logic (loading, preprocessing,
# parsing, error handling) without ever running the 100 MB OCR model.
# ─────────────────────────────────────────────────────────────────────────────

import io
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from app.core.exceptions import OCRError
from app.models.book import OCRResult
from app.services.ocr import EasyOCRService
from app.services.ocr_preprocessing import OCRSpinePreprocessor
from app.services.text_parser import SpineTextParser


def _write_crop(tmp_path: Path, filename: str = "crop.jpg") -> str:
    """Write a small synthetic crop JPEG to tmp_path and return its path string."""
    img = PILImage.new("RGB", (60, 200), color=(180, 150, 120))
    path = tmp_path / filename
    img.save(str(path), format="JPEG")
    return str(path)


def _make_ocr_box(text: str, confidence: float, y: float = 10.0) -> tuple:
    """Build a synthetic EasyOCR-format detection box."""
    return (
        [[0, y], [100, y], [100, y + 20], [0, y + 20]],
        text,
        confidence,
    )


def _make_service(mock_reader_boxes: list | None = None) -> EasyOCRService:
    """
    Build an EasyOCRService with:
      - A mock _reader pre-injected (bypasses model loading)
      - Real preprocessor (with all steps disabled to avoid image size changes
        that would complicate test assertions)
      - Real SpineTextParser (so parsing is tested end-to-end)
    """
    svc = EasyOCRService(
        languages=["en"],
        gpu=False,
        preprocessor=OCRSpinePreprocessor(
            upscale=False,
            fix_orientation=False,
            adaptive_threshold=False,
        ),
        parser=SpineTextParser(),
    )

    # Inject a mock reader whose readtext() returns the given boxes
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = mock_reader_boxes or []
    svc._reader = mock_reader   # Bypass lazy loading

    return svc


class TestEasyOCRServiceExtract:

    def test_returns_ocr_result_type(self, tmp_path):
        """extract() must always return an OCRResult instance."""
        svc = _make_service([_make_ocr_box("Dune", 0.90)])
        crop_path = _write_crop(tmp_path)
        book_id = str(uuid.uuid4())

        result = svc.extract(crop_path=crop_path, book_id=book_id)

        assert isinstance(result, OCRResult)

    def test_book_id_preserved_in_result(self, tmp_path):
        """The book_id passed to extract() must appear in the returned OCRResult."""
        svc = _make_service([_make_ocr_box("Some Book", 0.88)])
        crop_path = _write_crop(tmp_path)
        book_id = "test-book-id-abc"

        result = svc.extract(crop_path=crop_path, book_id=book_id)

        assert result.book_id == book_id

    def test_crop_path_preserved_in_result(self, tmp_path):
        """The crop_image_path in the result must match the input crop_path."""
        svc = _make_service([_make_ocr_box("Some Book", 0.88)])
        crop_path = _write_crop(tmp_path)
        book_id = str(uuid.uuid4())

        result = svc.extract(crop_path=crop_path, book_id=book_id)

        assert result.crop_image_path == crop_path

    def test_title_extracted_from_ocr_boxes(self, tmp_path):
        """The raw_title field should reflect what SpineTextParser extracted."""
        svc = _make_service([
            _make_ocr_box("The Great Gatsby", 0.92, y=10),
            _make_ocr_box("F. Scott Fitzgerald", 0.87, y=80),
        ])
        crop_path = _write_crop(tmp_path)

        result = svc.extract(crop_path=crop_path, book_id=str(uuid.uuid4()))

        assert result.raw_title == "The Great Gatsby"

    def test_author_extracted_from_ocr_boxes(self, tmp_path):
        """The raw_author field should reflect what SpineTextParser extracted."""
        svc = _make_service([
            _make_ocr_box("Dune", 0.93, y=10),
            _make_ocr_box("Frank Herbert", 0.89, y=80),
        ])
        crop_path = _write_crop(tmp_path)

        result = svc.extract(crop_path=crop_path, book_id=str(uuid.uuid4()))

        assert result.raw_author == "Frank Herbert"

    def test_empty_ocr_output_gives_empty_title_author(self, tmp_path):
        """If the OCR model finds no text, title and author should be empty strings."""
        svc = _make_service([])   # No detected text
        crop_path = _write_crop(tmp_path)

        result = svc.extract(crop_path=crop_path, book_id=str(uuid.uuid4()))

        assert result.raw_title == ""
        assert result.raw_author == ""
        assert result.confidence == 0.0

    def test_confidence_is_float_between_zero_and_one(self, tmp_path):
        """Confidence score must always be in [0.0, 1.0]."""
        svc = _make_service([_make_ocr_box("Clear Text Here", 0.85)])
        crop_path = _write_crop(tmp_path)

        result = svc.extract(crop_path=crop_path, book_id=str(uuid.uuid4()))

        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_flagged_for_review_always_false(self, tmp_path):
        """
        extract() must NEVER set flagged_for_review = True.
        That is exclusively the ConfidenceRouter's responsibility.
        """
        svc = _make_service([_make_ocr_box("A Book", 0.10)])  # Very low confidence
        crop_path = _write_crop(tmp_path)

        result = svc.extract(crop_path=crop_path, book_id=str(uuid.uuid4()))

        assert result.flagged_for_review is False

    def test_raises_ocr_error_when_file_missing(self):
        """If the crop file does not exist, OCRError must be raised."""
        svc = _make_service()

        with pytest.raises(OCRError, match="Crop image not found"):
            svc.extract(
                crop_path="/nonexistent/path/book_0.jpg",
                book_id=str(uuid.uuid4()),
            )

    def test_raises_ocr_error_when_reader_throws(self, tmp_path):
        """If EasyOCR's readtext() raises, it must be wrapped in OCRError."""
        svc = _make_service()
        svc._reader.readtext.side_effect = RuntimeError("GPU out of memory")
        crop_path = _write_crop(tmp_path)

        with pytest.raises(OCRError, match="GPU out of memory"):
            svc.extract(crop_path=crop_path, book_id=str(uuid.uuid4()))

    def test_reader_called_with_numpy_array(self, tmp_path):
        """
        EasyOCR's readtext() must receive a numpy array (not a PIL Image or path).
        """
        import numpy as np

        svc = _make_service([_make_ocr_box("Test", 0.90)])
        crop_path = _write_crop(tmp_path)

        svc.extract(crop_path=crop_path, book_id=str(uuid.uuid4()))

        # Verify readtext was called with a numpy array
        call_args = svc._reader.readtext.call_args
        first_arg = call_args[0][0]
        assert isinstance(first_arg, np.ndarray)

    def test_reader_called_with_detail_1(self, tmp_path):
        """readtext() must be called with detail=1 to get bounding boxes."""
        svc = _make_service([_make_ocr_box("Test", 0.90)])
        crop_path = _write_crop(tmp_path)

        svc.extract(crop_path=crop_path, book_id=str(uuid.uuid4()))

        call_kwargs = svc._reader.readtext.call_args[1]
        assert call_kwargs.get("detail") == 1


class TestEasyOCRServiceResultMerger:
    """Tests for the result_merger integration via review queue flow."""

    def test_multiple_crops_all_processed(self, tmp_path):
        """
        Calling extract() multiple times (one per crop) should produce
        independent OCRResult objects with different book_ids.
        """
        svc = _make_service([_make_ocr_box("Book One", 0.90)])

        ids = [str(uuid.uuid4()) for _ in range(3)]
        results = []
        for book_id in ids:
            path = _write_crop(tmp_path, f"crop_{book_id[:8]}.jpg")
            results.append(svc.extract(crop_path=path, book_id=book_id))

        returned_ids = [r.book_id for r in results]
        assert returned_ids == ids
