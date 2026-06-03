# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_detection.py
#
# Unit tests for FasterRCNNDetectionService.
#
# The actual Faster-RCNN model is NEVER loaded in these tests.
# Instead, we mock out `_run_inference` to return synthetic prediction tensors.
# This means:
#   - Tests run in milliseconds (no 10-second model load)
#   - Tests are deterministic (no stochastic model output)
#   - CI/CD doesn't need a GPU or the ~300 MB model weights
#
# We use MemoryCropStore so no files are written to disk.
# ─────────────────────────────────────────────────────────────────────────────

import uuid
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from app.core.exceptions import DetectionError, NoBooksDetectedError
from app.models.book import BoundingBox
from app.services.crop_store import MemoryCropStore
from app.services.detection import FasterRCNNDetectionService


def _make_shelf_image(width=800, height=400) -> PILImage.Image:
    """Synthetic shelf image for tests."""
    return PILImage.new("RGB", (width, height), color=(180, 160, 140))


def _make_service(
    detection_threshold: float = 0.5,
    nms_iou_threshold: float = 0.3,
) -> FasterRCNNDetectionService:
    """
    Build a FasterRCNNDetectionService with:
      - MemoryCropStore (no disk I/O)
      - A pre-set _model attribute (so no actual model is loaded)
      - Configurable thresholds
    """
    svc = FasterRCNNDetectionService(
        model_weights_path=None,
        upload_dir="/tmp/test",
        crop_store=MemoryCropStore(),
        detection_threshold=detection_threshold,
        nms_iou_threshold=nms_iou_threshold,
        device="cpu",
    )
    # Inject a fake model object so _model is not None (skips load_detection_model)
    svc._model = MagicMock()
    return svc


def _make_tensor_predictions(boxes, scores, labels):
    """
    Build a fake predictions dict that mirrors torchvision's output format.
    `boxes` is a list of [x1, y1, x2, y2] lists.
    """
    import torch
    return {
        "boxes":  torch.tensor(boxes,  dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


class TestFasterRCNNDetectionService:

    def test_detect_returns_correct_number_of_books(self):
        """
        When inference returns 3 high-confidence book detections, detect()
        should return 3 BoundingBoxes and 3 crop paths.
        """
        svc = _make_service()
        image = _make_shelf_image()
        job_id = str(uuid.uuid4())

        # 3 high-confidence books (COCO book class = 84)
        fake_preds = _make_tensor_predictions(
            boxes=[[10, 10, 80, 300], [100, 10, 170, 300], [200, 10, 270, 300]],
            scores=[0.95, 0.88, 0.76],
            labels=[84, 84, 84],
        )

        with patch.object(svc, "_run_inference", return_value=fake_preds):
            boxes, paths = svc.detect(image, job_id)

        assert len(boxes) == 3
        assert len(paths) == 3

    def test_detect_returns_bounding_box_objects(self):
        """detect() should return proper BoundingBox model instances."""
        svc = _make_service()
        image = _make_shelf_image()
        job_id = str(uuid.uuid4())

        fake_preds = _make_tensor_predictions(
            boxes=[[10, 10, 100, 350]],
            scores=[0.92],
            labels=[84],
        )

        with patch.object(svc, "_run_inference", return_value=fake_preds):
            boxes, _ = svc.detect(image, job_id)

        assert isinstance(boxes[0], BoundingBox)
        # Box coordinates must be non-negative
        assert boxes[0].x_min >= 0
        assert boxes[0].y_min >= 0
        assert boxes[0].x_max > boxes[0].x_min
        assert boxes[0].y_max > boxes[0].y_min

    def test_detect_saves_crop_paths_to_store(self):
        """Each detected book should produce a saved crop in the MemoryCropStore."""
        crop_store = MemoryCropStore()
        svc = FasterRCNNDetectionService(
            model_weights_path=None,
            upload_dir="/tmp/test",
            crop_store=crop_store,
            detection_threshold=0.5,
            device="cpu",
        )
        svc._model = MagicMock()
        image = _make_shelf_image()
        job_id = str(uuid.uuid4())

        fake_preds = _make_tensor_predictions(
            boxes=[[10, 10, 80, 300], [100, 10, 170, 300]],
            scores=[0.90, 0.85],
            labels=[84, 84],
        )

        with patch.object(svc, "_run_inference", return_value=fake_preds):
            _, paths = svc.detect(image, job_id)

        # Both crops should be in the memory store
        assert len(crop_store.saved_paths) == 2
        for path in paths:
            assert crop_store.get_bytes(path) is not None

    def test_detect_filters_low_confidence_predictions(self):
        """
        Predictions below detection_threshold must be excluded from results.
        """
        svc = _make_service(detection_threshold=0.8)  # High threshold
        image = _make_shelf_image()
        job_id = str(uuid.uuid4())

        fake_preds = _make_tensor_predictions(
            boxes=[[10, 10, 80, 300], [100, 10, 170, 300], [200, 10, 270, 300]],
            scores=[0.95, 0.79, 0.60],   # Only first is above 0.8
            labels=[84, 84, 84],
        )

        with patch.object(svc, "_run_inference", return_value=fake_preds):
            boxes, paths = svc.detect(image, job_id)

        assert len(boxes) == 1
        assert boxes[0].confidence >= 0.8

    def test_detect_filters_non_book_classes(self):
        """
        Predictions with class IDs that are not book (84) must be ignored.
        """
        svc = _make_service()
        image = _make_shelf_image()
        job_id = str(uuid.uuid4())

        fake_preds = _make_tensor_predictions(
            boxes=[[10, 10, 80, 300], [100, 10, 170, 300]],
            scores=[0.95, 0.90],
            labels=[84, 1],    # First is book (84), second is person (1)
        )

        with patch.object(svc, "_run_inference", return_value=fake_preds):
            boxes, paths = svc.detect(image, job_id)

        assert len(boxes) == 1

    def test_detect_raises_no_books_when_no_detections(self):
        """
        If no predictions survive filtering, NoBooksDetectedError must be raised.
        """
        svc = _make_service()
        image = _make_shelf_image()
        job_id = str(uuid.uuid4())

        # All predictions are of wrong class
        fake_preds = _make_tensor_predictions(
            boxes=[[10, 10, 80, 300]],
            scores=[0.95],
            labels=[1],    # person, not book
        )

        with patch.object(svc, "_run_inference", return_value=fake_preds):
            with pytest.raises(NoBooksDetectedError):
                svc.detect(image, job_id)

    def test_detect_raises_no_books_when_all_below_threshold(self):
        """If all books are below the confidence threshold, raise NoBooksDetectedError."""
        svc = _make_service(detection_threshold=0.9)
        image = _make_shelf_image()
        job_id = str(uuid.uuid4())

        fake_preds = _make_tensor_predictions(
            boxes=[[10, 10, 80, 300], [100, 10, 170, 300]],
            scores=[0.70, 0.80],   # Both below 0.9 threshold
            labels=[84, 84],
        )

        with patch.object(svc, "_run_inference", return_value=fake_preds):
            with pytest.raises(NoBooksDetectedError):
                svc.detect(image, job_id)

    def test_detect_raises_detection_error_on_inference_failure(self):
        """If the model raises an exception, it should be wrapped in DetectionError."""
        svc = _make_service()
        image = _make_shelf_image()
        job_id = str(uuid.uuid4())

        with patch.object(svc, "_run_inference", side_effect=RuntimeError("CUDA OOM")):
            with pytest.raises(DetectionError, match="CUDA OOM"):
                svc.detect(image, job_id)

    def test_detect_nms_removes_overlapping_boxes(self):
        """
        Two heavily overlapping boxes (IoU > nms_iou_threshold) should be
        reduced to one after NMS.
        """
        # Very low nms_iou_threshold → aggressive suppression
        svc = _make_service(nms_iou_threshold=0.1)
        image = _make_shelf_image()
        job_id = str(uuid.uuid4())

        # Two nearly identical boxes (one book detected twice)
        fake_preds = _make_tensor_predictions(
            boxes=[[10, 10, 80, 300], [12, 11, 82, 302]],   # Almost identical
            scores=[0.95, 0.90],
            labels=[84, 84],
        )

        with patch.object(svc, "_run_inference", return_value=fake_preds):
            boxes, _ = svc.detect(image, job_id)

        # Should keep only the higher-confidence box
        assert len(boxes) == 1

    def test_crop_paths_contain_job_id(self):
        """Returned crop paths should be namespaced under the job_id."""
        svc = _make_service()
        image = _make_shelf_image()
        job_id = "fixed-test-job-id-123"

        fake_preds = _make_tensor_predictions(
            boxes=[[10, 10, 80, 300]],
            scores=[0.92],
            labels=[84],
        )

        with patch.object(svc, "_run_inference", return_value=fake_preds):
            _, paths = svc.detect(image, job_id)

        assert job_id in paths[0]

    def test_bounding_box_confidence_matches_model_score(self):
        """The BoundingBox.confidence should match the model's prediction score."""
        svc = _make_service()
        image = _make_shelf_image()
        job_id = str(uuid.uuid4())

        expected_score = 0.87
        fake_preds = _make_tensor_predictions(
            boxes=[[10, 10, 80, 300]],
            scores=[expected_score],
            labels=[84],
        )

        with patch.object(svc, "_run_inference", return_value=fake_preds):
            boxes, _ = svc.detect(image, job_id)

        assert abs(boxes[0].confidence - expected_score) < 0.001
