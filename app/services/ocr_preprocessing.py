# ─────────────────────────────────────────────────────────────────────────────
# app/services/ocr_preprocessing.py
#
# Advanced preprocessing pipeline specifically for book spine OCR.
#
# Why a separate preprocessing module from image_utils.py?
#   image_utils.py contains GENERAL image transformations (resize, crop, rotate)
#   used by both the detection and OCR stages.
#   THIS module contains SPINE-SPECIFIC logic whose only purpose is maximising
#   OCR accuracy on the unique challenges of book spine images:
#
#   Challenge 1 – Spine orientation
#     Books on a shelf are read left-to-right when upright, but the spine text
#     is printed vertically (rotated 90° either way). EasyOCR handles rotated
#     text, but accuracy improves significantly when we normalise orientation
#     first. We detect which rotation gives the most horizontal text runs.
#
#   Challenge 2 – Low resolution
#     Spine crops from a shelf photo are often 30–80 px wide. EasyOCR's
#     accuracy drops sharply below ~32 px character height. We upscale narrow
#     crops before inference.
#
#   Challenge 3 – Dark/coloured backgrounds
#     Many book covers have dark backgrounds. Standard OCR expects light
#     background / dark text. We apply adaptive thresholding to normalise this.
#
# SOLID (Single Responsibility):
#   Each function does exactly one thing. Compose them in OCRSpinePreprocessor.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from PIL import Image as PILImage, ImageFilter
import numpy as np
import cv2


# ── Constants ──────────────────────────────────────────────────────────────────

MIN_OCR_WIDTH_PX = 64 
# Minimum width (px) at which we consider the crop large enough for OCR.
# If below this, then upscaling significantly improves accuracy. 
 
TARGET_OCR_WIDTH_PX = 256
# Target width for upscaled narrow crops.

# Candidate rotations to try during orientation detection (degrees clockwise)
ORIENTATION_CANDIDATES = [0, 90, 180, 270]


# ── Individual preprocessing steps ────────────────────────────────────────────

def upscale_if_narrow(image: PILImage.Image, 
                      MIN_OCR_WIDTH_PX: int = 64, 
                      TARGET_OCR_WIDTH_PX: int = 256) -> PILImage.Image:
    """
    Upscale a narrow crop so that character details are large enough for OCR.

    Book spines are tall and narrow. A spine crop of 40×300 px has characters
    that are only ~20 px tall — too small for reliable OCR. Upscaling to
    256 px wide keeps aspect ratio and brings character height to ~120 px.

    Uses LANCZOS (high-quality) when upscaling. No-ops for wide images.

    Args:
        image: Input PIL Image (any size).

    Returns:
        Upscaled PIL Image if narrow, otherwise the original unchanged.
    """
    
    w, h = image.size
    print(f"Original image size: {w}x{h}")
    
    if w > h:
        print("Image is in landscape orientation (wider than tall)")
    else:
        print("Image is in portrait orientation (taller than wide) → Rotating to landscape")
        image = image.rotate(90, expand=True)
    
    # Only upscale if the image is narrower than our minimum threshold
    if w >= MIN_OCR_WIDTH_PX:
        return image

    # Scale factor that brings width to TARGET_OCR_WIDTH_PX
    scale = TARGET_OCR_WIDTH_PX / w
    new_w = TARGET_OCR_WIDTH_PX
    new_h = int(h * scale)

    return image.resize((new_w, new_h), PILImage.LANCZOS)



def normalise_orientation(image: PILImage.Image) -> PILImage.Image:
    """
    Rotate the crop to the orientation most likely to yield accurate OCR.

    Strategy: try all four 90° rotations, score each by the "horizontalness"
    of its text regions using a simple projection profile heuristic, and
    return the rotation with the highest score.

    The projection profile heuristic:
      - Convert to greyscale and binarise (text = black on white)
      - For each row, count black pixels (horizontal projection)
      - Text rows have concentrated black pixels → the variance of the
        row-wise black-pixel counts is HIGH when text runs horizontally
      - The rotation with the HIGHEST row-projection variance has the most
        horizontal text → most OCR-friendly orientation

    This is much faster than running OCR four times and picking the best result.

    Args:
        image: RGB PIL Image of a book spine crop.

    Returns:
        PIL Image rotated to the best orientation.
    """
    best_rotation = 0
    best_score = -1.0

    for angle in ORIENTATION_CANDIDATES:
        rotated = image.rotate(-angle, expand=True)   # PIL uses CCW, we want CW
        score = _horizontal_text_score(rotated)

        if score > best_score:
            best_score = score
            best_rotation = angle

    if best_rotation == 0:
        return image

    return image.rotate(-best_rotation, expand=True)


def _horizontal_text_score(image: PILImage.Image) -> float:
    """
    Score an image by how horizontally-oriented its text regions are.

    Higher score = text runs left-to-right = good for OCR.
    Lower score = text runs top-to-bottom = needs rotation.

    Implementation:
      1. Convert to greyscale
      2. Binarise with a fixed threshold (127)
      3. For each row, count dark pixels
      4. Return the variance of the row counts — high variance means rows
         alternate between "text row" (many dark pixels) and "gap row"
         (few dark pixels), which is the signature of horizontal text.
    """
    grey = image.convert("L")
    # Binarise: pixels darker than 127 → 0 (text), lighter → 255 (background)
    binary = grey.point(lambda p: 0 if p < 127 else 255)

    width, height = binary.size
    if height == 0:
        return 0.0

    # Count dark (text) pixels in each row
    pixels = list(binary.getdata())
    row_dark_counts = []
    for row_idx in range(height):
        row_start = row_idx * width
        row_pixels = pixels[row_start : row_start + width]
        # Dark pixel = value 0
        dark_count = sum(1 for p in row_pixels if p == 0)
        row_dark_counts.append(dark_count)

    if not row_dark_counts:
        return 0.0

    # Variance of row dark-pixel counts
    mean = sum(row_dark_counts) / len(row_dark_counts)
    variance = sum((c - mean) ** 2 for c in row_dark_counts) / len(row_dark_counts)
    return variance


def apply_adaptive_threshold(image: PILImage.Image) -> PILImage.Image:
    """
    Convert a colour spine crop to high-contrast greyscale via adaptive
    thresholding to normalise varied background colours.

    Why adaptive (not global) thresholding?
      Books have wildly different spine colours — a dark navy spine has text
      near value 200, while a yellow spine has text near value 80. A single
      global threshold (e.g. 127) would misclassify one or the other.
      Adaptive thresholding computes the threshold locally per small region,
      making it robust to uneven illumination and colour variation.

    We simulate local adaptive thresholding using PIL's built-in tools:
      1. Convert to greyscale
      2. Create a blurred version (acts as the local mean)
      3. Subtract: pixels brighter than their local mean → white (background)
                   pixels darker than their local mean → black (text)

    Args:
        image: RGB PIL Image.

    Returns:
        Greyscale PIL Image with text as dark pixels on light background.
        NOTE: Returns greyscale ("L" mode), not RGB. EasyOCR accepts both.
    """
    grey = image.convert("L")

    # Gaussian blur to compute local mean (radius 15 → captures ~30px context)
    blurred = grey.filter(ImageFilter.GaussianBlur(radius=15))

    # Pixel-wise: if original > blurred (brighter than local mean) → background
    #             if original ≤ blurred (darker than local mean) → text
    width, height = grey.size
    orig_pixels  = list(grey.getdata())
    blur_pixels  = list(blurred.getdata())

    # C = 5: small constant subtracted from local mean to reduce noise
    # Pixels more than 5 grey levels below the local mean are classified as text
    C = 5
    result_pixels = [
        255 if (orig - blur) > -C else 0
        for orig, blur in zip(orig_pixels, blur_pixels)
    ]

    result = PILImage.new("L", (width, height))
    result.putdata(result_pixels)
    return result


def to_rgb_for_easyocr(image: PILImage.Image) -> PILImage.Image:
    """
    Convert any PIL Image mode back to RGB for EasyOCR.

    EasyOCR's Python API accepts both RGB and greyscale numpy arrays, but
    accepting only RGB here keeps the pipeline's type contract consistent
    and avoids subtle mode-related bugs in EasyOCR's internal preprocessing.

    Args:
        image: PIL Image in any mode.

    Returns:
        RGB PIL Image.
    """
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


# ── Composed preprocessing pipeline ───────────────────────────────────────────
class OCRSpinePreprocessor:
    """
    Composes the individual preprocessing steps into a single callable pipeline.

    Usage:
        preprocessor = OCRSpinePreprocessor()
        ready_image = preprocessor.prepare(crop)

    The pipeline steps are ordered deliberately:
      1. upscale_if_narrow   → ensures characters are large enough to detect
      2. normalise_orientation → rotates to horizontal-text orientation
      3. apply_adaptive_threshold → normalises contrast regardless of spine colour
      4. to_rgb_for_easyocr  → ensures EasyOCR gets an RGB tensor

    SOLID (Open/Closed):
      New preprocessing steps can be injected by subclassing and overriding
      `prepare()`, or by adding steps to `_steps` without modifying callers.
    """

    def __init__(
        self,
        upscale: bool = True,
        fix_orientation: bool = False,
        adaptive_threshold: bool = False,
    ) -> None:
        """
        Args:
            upscale:            Enable upscaling of narrow crops (default True).
            fix_orientation:    Enable orientation detection/correction (default True).
            adaptive_threshold: Enable adaptive contrast normalisation (default True).
        """
        self._upscale = upscale
        self._fix_orientation = fix_orientation
        self._adaptive_threshold = adaptive_threshold

    def prepare(self, image: PILImage.Image) -> PILImage.Image:
        """
        Run the full preprocessing pipeline on a single spine crop.

        Args:
            image: Raw PIL Image crop from the detection stage (RGB).

        Returns:
            Preprocessed PIL Image ready for EasyOCR (RGB).
        """
        # Step 1: Upscale narrow crops before any other processing
        # (orientation detection is more accurate on larger images)
        if self._upscale:
            image = upscale_if_narrow(image)

        # Step 2: Rotate to horizontal orientation
        if self._fix_orientation:
            image = normalise_orientation(image)

        # Step 3: Adaptive contrast normalisation
        if self._adaptive_threshold:
            image = apply_adaptive_threshold(image)

        # Step 4: Ensure RGB mode for EasyOCR
        image = to_rgb_for_easyocr(image)

        return image



## ─────────────────────────────────────────────────────────────────────────────────────────────

class TextRegionExtraction:
    """
    Composes the text region extraction process into a single callable pipeline.
    
    Extracting out Detected Text Regions from book spine as independent images 
    using Perspective Transform to feed into PaddleOCR TextRecognition:

    Usage:
        extracter = TextRegionExtraction()
        ready_list_of_ordered_images = extracter.prepare(original_image, text_detection_result)
    """

    # def __init__(
    #     self,
    #     original_image: np.array,
    #     text_detection_result: dict,
    #     ) -> None:
    #     self._image_arr = original_image
    #     self._det_results = text_detection_result
        
    #     self.crop_text_regions(self._image_arr, self._det_results)
    
    
    def crop_text_regions(self, image_arr: np.array, text_det_result: dict) -> list[tuple]:
        crops = []

        for polygon, rank in zip(text_det_result['dt_polys'], text_det_result['position_rank_on_x_axis']):
            warped = self.four_point_transform(image_arr, polygon)
            crops.append((warped, rank))
            
        crops = sorted(crops, key=lambda x: x[1])   # Sort by rank (position on x-axis)
        return crops
    
    
    def four_point_transform(self, image: np.array, pts: list) -> np.array:
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute width
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        # Compute height
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped


    def order_points(self, pts: list[tuple]) -> list[tuple]:
        # Ensure consistent order: top-left, top-right, bottom-right, bottom-left
        pts = np.array(pts, dtype="float32")

        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        ordered = np.zeros((4, 2), dtype="float32")
        ordered[0] = pts[np.argmin(s)]       # top-left
        ordered[2] = pts[np.argmax(s)]       # bottom-right
        ordered[1] = pts[np.argmin(diff)]    # top-right
        ordered[3] = pts[np.argmax(diff)]    # bottom-left

        return ordered
    







    