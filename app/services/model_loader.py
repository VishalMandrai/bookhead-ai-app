# ─────────────────────────────────────────────────────────────────────────────
# app/services/model_loader.py
#
# Faster-RCNN model loading with lazy singleton pattern.
#
# WHY A SEPARATE MODULE?
#   Loading a PyTorch model is expensive: reading weights from disk takes
#   several seconds and allocates hundreds of MB of RAM/VRAM. We want to
#   pay this cost ONCE per worker process, not on every request.
#
#   Keeping model loading here (rather than inside FasterRCNNDetectionService)
#   enforces the Single Responsibility Principle: the detection service's
#   job is INFERENCE, not model lifecycle management.
#
# SINGLETON PATTERN:
#   The module-level `_model_cache` dict acts as a process-scoped singleton.
#   The first call to `load_detection_model()` loads the weights; subsequent
#   calls return the already-loaded model instantly.
#
# CUSTOM WEIGHTS SUPPORT:
#   Pass a path to a `.pth` file to use fine-tuned weights (e.g. a model
#   trained specifically on book-spine images). Pass None to use torchvision's
#   COCO-pretrained weights as a starting point.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# Module-level cache: maps weights_path (or "pretrained") → loaded model
# Using a dict (not a bare global) makes it easy to support multiple model
# variants in the same process if needed in future.
_model_cache: dict[str, Any] = {}


def load_detection_model(
    weights_path: str | None = None,
    num_detection_classes: int = None,
    device: str = "cpu",
) -> Any:
    """
    Load (or return cached) the Faster-RCNN model for book spine detection.

    Args:
        weights_path: Path to a custom .pth weights file, or None to use
                      torchvision's COCO pretrained weights.
                      
        num_detection_classes: Number of classes that fine-tuned model detects. Need to be set 
                               when using own fine-tuned model for book detection.
        
        device:       PyTorch device string: "cpu", "cuda", "cuda:0", etc.
                      Defaults to "cpu" for broad compatibility; GPU is used
                      automatically when DETECTION_DEVICE=cuda is set in config.

    Returns:
        A torchvision FasterRCNN model in eval() mode, moved to `device`.

    Raises:
        FileNotFoundError: If `weights_path` is given but the file doesn't exist.
        RuntimeError:      If the weights file is incompatible with the model.
    """
    # Use the weights path (or the string "pretrained") as the cache key
    cache_key = weights_path or "pretrained"

    if cache_key in _model_cache:
        logger.debug("detection_model_cache_hit", cache_key=cache_key)
        return _model_cache[cache_key]

    # ── Lazy import: only import torch/torchvision when this function is called ──
    # This keeps startup time fast when running tests or CLI tools that don't
    # need the detection model.
    import torch
    import torchvision
    from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2,
                                              FasterRCNN_ResNet50_FPN_V2_Weights
                                              )

    logger.info("detection_model_loading", weights=cache_key, device=device)

    if weights_path:
        # ── Custom fine-tuned weights & Pre-downloaded models for faster loading  ──────
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Detection model weights not found at: {weights_path}"
            )

        # Build the model architecture WITHOUT pretrained weights
        # (we're about to load our own), then replace the classifier head
        # to match the number of classes our fine-tuned model expects.
        #
        # NOTE: The number of classes (including background) must match
        # the checkpoint. For model fine-tuned purely on book-spine detection
        # it will be 2 (background + "book").
        
        model = fasterrcnn_resnet50_fpn_v2(weights=None)

        # Swap the box predictor head to match our class count
        # Only if it is provided during function call
        if num_detection_classes >= 2:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = _build_box_predictor(in_features, 
                                                                 num_classes=num_detection_classes)

        # Load the saved state dict
        state_dict = torch.load(path, 
                                map_location=device)
        model.load_state_dict(state_dict)

        logger.info("detection_model_custom_weights_loaded", path=str(path))

    else:
        # ── COCO pretrained weights - Download and Load ─────────────────────────
        # FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT are the best available
        # COCO pretrained weights in torchvision (trained on 80 object classes).
        # "book" is one of those 80 classes (COCO class ID 84), so this works
        # as an out-of-the-box baseline for book detection without any
        # fine-tuning. Performance improves significantly with fine-tuned weights.
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        logger.info("detection_model_pretrained_loaded")

    # Move model to the requested device and switch to inference mode
    model.to(device)
    model.eval()   # eval() disables dropout and batchnorm training behaviour

    # Cache and return
    _model_cache[cache_key] = model
    logger.info("detection_model_ready", device=device)
    return model


def _build_box_predictor(in_features: int, num_classes: int):
    """
    Build a new FastRCNNPredictor head with the given number of output classes.

    This replaces the pretrained COCO head (80 classes) with a smaller head
    matching our fine-tuned dataset.

    Args:
        in_features:  Number of input features from the ROI pooling layer.
        num_classes:  Total classes INCLUDING background (so book-only = 2).
    """
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    return FastRCNNPredictor(in_features, num_classes)


def clear_model_cache() -> None:
    """
    Evict all cached models from memory.

    Called in tests to ensure a clean state between test cases, or
    in production to force a model reload (e.g. after hot-swapping weights).
    """
    _model_cache.clear()
    logger.info("detection_model_cache_cleared")
