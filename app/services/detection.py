# ─────────────────────────────────────────────────────────────────────────────
# app/services/detection.py
#
# FasterRCNNDetectionService – concrete implementation of BaseDetectionService.
#
# Responsibility:
#   Given a PIL Image of a bookshelf, detect all visible book spines using
#   a Faster-RCNN model, crop each detected spine from the image, and
#   persist the crops via the CropStore.
#
# Pipeline inside detect():
#   1. Preprocess – resize the image to inference dimensions
#   2. Transform  – convert PIL → normalised PyTorch tensor
#   3. Inference  – run Faster-RCNN forward pass (no_grad for speed)
#   4. Filter     – discard low-confidence detections
#   5. NMS        – remove duplicate overlapping boxes (non-maximum suppression)
#   6. Crop       – extract and preprocess each spine region
#   7. Store      – save crops via CropStore, collect relative paths
#
# SOLID notes:
#   Single Responsibility: Only does detection + crop extraction. File I/O is
#     delegated to CropStore; model loading is delegated to model_loader.
#   Dependency Inversion: Depends on BaseCropStore (not DiskCropStore directly).
#     Tests inject MemoryCropStore to avoid disk I/O.
#   Open/Closed: Adding a new model backend means creating a new subclass of
#     BaseDetectionService, not modifying this file.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
from typing import Any

from PIL import Image as PILImage

from app.core.exceptions import DetectionError, NoBooksDetectedError
from app.core.logging import get_logger
from app.models.book import BoundingBox
from app.services.base import BaseDetectionService
from app.services.crop_store import BaseCropStore, DiskCropStore
from app.services.image_utils import (
    crop_bounding_box,
    preprocess_crop_for_ocr,
    resize_for_inference,
)
from app.services.model_loader import load_detection_model

logger = get_logger(__name__)


class FasterRCNNDetectionService(BaseDetectionService):
    """
    Book spine detection using torchvision's Faster-RCNN ResNet-50 FPN v2.

    This is the production implementation of BaseDetectionService.
    For tests, use the StubDetectionService from conftest.py instead.
    """

    # ── COCO class ID for "book" ───────────────────────────────────────────────
    # In the COCO dataset, "book" is class 84 (1-indexed).
    # When using pretrained COCO weights, we filter predictions to this class.
    # When using fine-tuned weights (2-class: background + book), the book
    # class ID is 1.
    
    COCO_BOOK_CLASS_ID = 84
    FINETUNED_BOOK_CLASS_ID = 1

    def __init__(
        self,
        model_weights_path: str | None = None,
        num_detection_classes: int = None,
        upload_dir: str = "/app/uploads",
        crop_store: BaseCropStore | None = None,
        detection_threshold: float = 0.70,
        nms_iou_threshold: float = 0.40,
        width_to_height_diff_of_book: int = 200,
        padding_around_book_crop: int = 2,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_weights_path:  Path to custom .pth weights, or None for pretrained.
            upload_dir:          Root directory for saving crop files (used to build
                                 the default DiskCropStore if crop_store is None).
            crop_store:          Injectable crop storage backend. Defaults to
                                 DiskCropStore(upload_dir). Pass MemoryCropStore
                                 in tests.
            detection_threshold: Minimum Faster-RCNN confidence score to keep a
                                 detection. Lower → more detections (more noise).
                                 Higher → fewer, more reliable detections.
            nms_iou_threshold:   IoU threshold for Non-Maximum Suppression.
                                 Boxes with IoU > this value are merged (duplicate
                                 removed). Lower → more aggressive merging.
            device:              PyTorch device ("cpu" or "cuda").
        """
        self._weights_path = model_weights_path
        self._num_detection_classes = num_detection_classes
        self._detection_threshold = detection_threshold
        self._nms_iou_threshold = nms_iou_threshold
        self._device = device
        self._width_to_height_diff_of_book = width_to_height_diff_of_book
        self._padding_around_book_crop = padding_around_book_crop

        # Inject or default the crop store
        self._crop_store: BaseCropStore = crop_store or DiskCropStore(upload_dir)

        # Determine which class ID to filter for
        # Fine-tuned (2-class) models use class 1; COCO pretrained uses 84
        self._book_class_id = (
            self.COCO_BOOK_CLASS_ID
            if num_detection_classes == 0
            else self.FINETUNED_BOOK_CLASS_ID
        )

        # The model is loaded lazily on first call to detect() so the
        # service can be instantiated quickly during app startup without
        # immediately loading hundreds of MB of weights.
        self._model: Any | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(
        self,
        image: PILImage.Image,
        job_id: str,
        img_idx: int = 0) -> tuple[list[BoundingBox], list[str]]:
        """
        Run full detection pipeline on a shelf image.

        See BaseDetectionService.detect() for the contract.
        """
        # ── Lazy model load ────────────────────────────────────────────────
        if self._model is None:
            self._model = load_detection_model(
                weights_path=self._weights_path,
                num_detection_classes=self._num_detection_classes,
                device=self._device,
            )

        # ── Step 1: Resize image to inference dimensions ───────────────────
        # Keeps memory usage bounded for high-res phone photos
        resized = resize_for_inference(image)

        logger.info("detection_started", job_id=job_id, 
                    original_size=image.size, resized_size=resized.size,
                    )

        # ── Step 2: Convert PIL Image → normalised PyTorch tensor ─────────
        tensor = self._image_to_tensor(resized)

        # ── Step 3: Run Faster-RCNN inference ─────────────────────────────
        try:
            raw_predictions = self._run_inference(tensor)
        except Exception as exc:
            raise DetectionError(
                f"Faster-RCNN inference failed: {exc}",
                detail={"job_id": job_id},
            ) from exc


        # ── Step 4: Filter predictions by class and confidence ─────────────
        filtered = self._filter_predictions(raw_predictions)

        logger.info(
            "detection_raw_predictions",
            job_id=job_id,
            total_objects=len(raw_predictions["boxes"]),
            after_filter_total_books=len(filtered["boxes"]),
            conf_threshold=self._detection_threshold,
        )

        if len(filtered["boxes"]) == 0:
            raise NoBooksDetectedError(
                "No books detected in the provided image. "
                "Try a clearer photo with the spines facing the camera.",
                detail={"job_id": job_id},
            )

        # ── Step 5: Non-Maximum Suppression (NMS) ─────────────────────────
        # NMS removes duplicate detections when two boxes overlap significantly.
        # This is especially common when the model fires multiple boxes for a
        # single wide book spine.
        kept_indices = self._apply_nms(filtered["boxes"], filtered["scores"])
        
        logger.info(
            "detection_after_nms",
            job_id=job_id,
            before_nms_total_books=len(filtered["boxes"]),
            after_nms_total_books=len(kept_indices),
            iou_threshold=self._nms_iou_threshold,
        )

        # ── Step 6 & 7: Crop each detected spine and save via CropStore ────
        bounding_boxes: list[BoundingBox] = []
        crop_paths: list[str] = []

        # Scale factor needed to map resized-image coordinates back to the
        # original image's pixel space (we crop from the ORIGINAL, not resized)
        scale_x = image.width  / resized.width
        scale_y = image.height / resized.height

        for book_index, pred_idx in enumerate(kept_indices):
            # Extract raw box coordinates from the tensor (still in resized space)
            x1, y1, x2, y2 = [
                float(c) for c in filtered["boxes"][pred_idx]
            ]
            score = float(filtered["scores"][pred_idx])

            # Scale coordinates back to the original image's pixel space
            orig_x1 = int(x1 * scale_x)
            orig_y1 = int(y1 * scale_y)
            orig_x2 = int(x2 * scale_x)
            orig_y2 = int(y2 * scale_y)
            
            ## Bounding Box must not be Squares and must be rectangles
            x_len = orig_x2 - orig_x1
            y_len = orig_y2 - orig_y1
            
            # Remove near-square boxes which are unlikely to be book spines 
            # (e.g. books photographed head-on, or false positives on objects like vases). 
            if max(x_len, y_len) - min(x_len, y_len) < self._width_to_height_diff_of_book: 
                logger.info(
                    "skipping_near_square_box",
                    job_id=job_id,
                    book_index=book_index,
                    box=(orig_x1, orig_y1, orig_x2, orig_y2),
                )
                continue

            # Build the BoundingBox model (validates coordinates)
            box = BoundingBox(
                x_min=orig_x1,
                y_min=orig_y1,
                x_max=orig_x2,
                y_max=orig_y2,
                confidence=score,
            )
            bounding_boxes.append(box)

            # Crop the original (full-resolution) image for best OCR quality
            raw_crop = crop_bounding_box(
                image,
                box=(orig_x1, orig_y1, orig_x2, orig_y2),
                padding_px=self._padding_around_book_crop,    # Small border to capture edge characters
            )

            # Apply OCR preprocessing (contrast enhancement + sharpening)
            # processed_crop = preprocess_crop_for_ocr(raw_crop)           # NOTE: Not applying this preprocessing cause OCR working fine on simple RGB image.

            if raw_crop.mode != "RGB":
                processed_crop = raw_crop.convert("RGB")
            else:
                processed_crop = raw_crop
                     
                     
            # Persist via CropStore and collect the relative path
            relative_path = self._crop_store.save(
                crop=processed_crop,
                job_id=job_id,
                book_index=book_index,
                image_idx=img_idx,
            )
            crop_paths.append(relative_path)

        logger.info("detection_complete", job_id=job_id, num_books=len(bounding_boxes))
        
        # Also saving the Original Resized image along with book crops
        self._crop_store.save(
                crop=resized,
                job_id=job_id,
                book_index="original",
            )

        return bounding_boxes, crop_paths



    # ── Private helpers ────────────────────────────────────────────────────────

    def _image_to_tensor(self, image: PILImage.Image):
        """
        Convert a PIL Image to the normalised float32 tensor expected by
        torchvision's Faster-RCNN.

        torchvision's detection models expect:
          - A list of float32 tensors
          - Values in range [0.0, 1.0]
          - Shape: (C, H, W) — channels-first

        We use torchvision.transforms.functional for the conversion rather
        than torchvision.transforms.ToTensor because the latter is deprecated
        in recent versions of torchvision.
        """
        # Lazy import for the same reason as model_loader.py
        import torch
        from torchvision.transforms.functional import to_tensor

        # to_tensor: PIL (H, W, C) uint8 → torch.Tensor (C, H, W) float32 [0,1]
        tensor = to_tensor(image)

        # Move to the same device as the model
        return tensor.to(self._device)

    def _run_inference(self, tensor) -> dict:
        """
        Run a single Faster-RCNN forward pass.

        Returns the raw predictions dict:
          {
            "boxes":  Tensor(N, 4)  – (x1, y1, x2, y2) in pixel coordinates
            "labels": Tensor(N,)    – class IDs (int)
            "scores": Tensor(N,)    – confidence scores (float, 0–1)
          }

        torch.no_grad() disables gradient tracking, which:
          - Reduces memory consumption (no computation graph stored)
          - Speeds up inference (no backward pass overhead)
        """
        import torch

        with torch.no_grad():
            # The model expects a list of tensors (one per image in the batch).
            # We process one image at a time (batch_size=1) for simplicity.
            predictions = self._model([tensor])

        # predictions is a list with one element per input image
        return {
            "boxes":  predictions[0]["boxes"].cpu(),
            "labels": predictions[0]["labels"].cpu(),
            "scores": predictions[0]["scores"].cpu(),
        }

    def _filter_predictions(self, predictions: dict) -> dict:
        """
        Keep only predictions where:
          - The class label matches our book class ID
          - The confidence score meets the detection threshold

        Args:
            predictions: Raw output from _run_inference().

        Returns:
            Filtered dict with the same structure as predictions.
        """
        import torch

        boxes  = predictions["boxes"]
        labels = predictions["labels"]
        scores = predictions["scores"]

        # Boolean mask: True for predictions we want to keep
        # Condition 1: correct class (book)
        is_book = labels == self._book_class_id
        # Condition 2: confidence above threshold
        is_confident = scores >= self._detection_threshold

        keep_mask = is_book & is_confident

        return {
            "boxes":  boxes[keep_mask],
            "labels": labels[keep_mask],
            "scores": scores[keep_mask],
        }

    def _apply_nms(self, boxes, scores) -> list[int]:
        """
        Apply Non-Maximum Suppression to remove redundant overlapping boxes.

        NMS works by:
          1. Sort boxes by descending confidence score.
          2. Keep the highest-scoring box.
          3. Remove any remaining boxes that overlap the kept box by more
             than `nms_iou_threshold` (IoU = Intersection over Union).
          4. Repeat from step 2 with remaining boxes.

        Args:
            boxes:  Tensor(N, 4) of filtered bounding boxes.
            scores: Tensor(N,)   of corresponding confidence scores.

        Returns:
            List of integer indices (into boxes/scores) that survived NMS.
            Sorted by descending score (highest confidence first).
        """
        from torchvision.ops import nms

        # torchvision.ops.nms returns a 1D tensor of kept indices
        kept_indices_tensor = nms(
            boxes=boxes.float(),        # nms requires float32
            scores=scores.float(),
            iou_threshold=self._nms_iou_threshold,
        )

        # Convert tensor to a plain Python list of ints for easy iteration
        return kept_indices_tensor.tolist()

