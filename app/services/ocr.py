# ─────────────────────────────────────────────────────────────────────────────
# app/services/ocr.py
#
# EasyOCRService – concrete implementation of BaseOCRService.
#
# Responsibility:
#   Given the file path of a saved book-spine crop image, run EasyOCR to
#   extract text, parse it into title/author fields, and return an OCRResult.
#
# Pipeline inside extract():
#   1. Load   – read the crop image from disk into a PIL Image
#   2. Preprocess – run OCRSpinePreprocessor (upscale, orientation, contrast)
#   3. Infer  – call EasyOCR Reader.readtext() to get raw text boxes
#   4. Parse  – SpineTextParser converts boxes → (title, author, confidence)
#   5. Return – build and return an OCRResult model
#
# EasyOCR model loading is LAZY (happens on first call to extract()).
# The Reader object is cached as an instance attribute so it is loaded only
# once per worker process.
#
# SOLID note (Single Responsibility):
#   EasyOCRService only drives the EasyOCR model. Text interpretation is
#   delegated to SpineTextParser. Preprocessing is OCRSpinePreprocessor.
#   Each has its own tests.
#
# SOLID note (Dependency Inversion):
#   SpineTextParser and OCRSpinePreprocessor are injected at construction
#   time, making them swappable in tests without subclassing EasyOCRService.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Tuple

from PIL import Image as PILImage

from app.core.exceptions import OCRError
from app.core.logging import get_logger
from app.models.book import OCRValidatedResult
from app.services.base import BaseOCRService
from app.services.ocr_preprocessing import OCRSpinePreprocessor, TextRegionExtraction
from app.services.text_parser import SpineTextCleaner, SpineTextValidator 

logger = get_logger(__name__)

## NOTE: As we know we'll be utilising PaddleOCR (for text detection in images + text recognition 
##       from images) and TrOCR (small model; as fallback if PaddleOCR underperforms).
##       PaddleOCR Text Detection would take Book Spine image as input. It will return boxed regions on
##       image where text is detected. Then we crop out these boxes as new set of images.
##       We need to preprocesss these new set of images and then feed into PaddleOCR Recognition for text
##       recognition. Once we get the text, we look up the Google Books records through a Public API
##       and get the most related books. Then we apply fuzzy logic which matches our OCR read text string 
##       with Google Book records. For the most related record, we scrap out Book Title and Author Name.

class EasyOCRService(BaseOCRService):
    """
    Book spine OCR using the EasyOCR library.

    EasyOCR supports 80+ languages and handles rotated/curved text well,
    making it a good fit for book spines that come in many orientations
    and languages.

    For production use, the Reader is initialised with `gpu=False` by default
    for broad CPU-only compatibility. Set `gpu=True` for faster inference on
    GPU-enabled machines.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
        preprocessor: OCRSpinePreprocessor | None = None,
        book_validator: SpineTextValidator | None = None,
        
        device: str = 'cpu',
        
        detector_threshold: float = 0.1,
        detector_box_threshold: float = 0.3,
        detector_unclip_ratio: float = 1.8,
        
        min_box_distance_px: int = 5,
        
        min_conf_thresh_text_rec: float = 0.7,
    ) -> None:
        """
        Args:
            languages:    List of language codes EasyOCR will detect.
                          Defaults to ["en"]. Adding languages (e.g. ["en", "fr"])
                          increases recall but slows inference.
            gpu:          Whether to use GPU for EasyOCR inference.
                          Defaults to False for broad CPU compatibility.
            preprocessor: OCR preprocessing pipeline. Defaults to
                          OCRSpinePreprocessor() with all steps enabled.
            parser:       Text parsing strategy. Defaults to SpineTextParser().
        """
        self._languages = languages or ["en"]
        self._gpu = gpu
        self._device = device

        # Injected collaborators — default to production implementations
        self._preprocessor: OCRSpinePreprocessor = preprocessor or OCRSpinePreprocessor()
        self._text_reg_extractor = TextRegionExtraction()
        self._spine_txt_cleaner = SpineTextCleaner()
        self._validator: SpineTextValidator = book_validator or SpineTextValidator()

        # Lazy-loaded PaddleOCR Text region detector & Text recognizer (loaded on first call to extract())
        # Stored as an instance attribute so it is reused across all calls.
        self._detector: Any | None = None
        self._reader: Any | None = None
        
        ## PaddleOCR - Text region detector params:
        self._detector_threshold = detector_threshold
        self._detector_box_threshold = detector_box_threshold
        self._detector_unclip_ratio = detector_unclip_ratio
        
        ## Param for Filtering Text Boxes from Text Detector:
        self._min_box_distance_px = min_box_distance_px
        
        ## Param for filtering recognised texts from text recognition
        ## keeping text which is recognised with good confidence score
        self._min_conf_thresh_text_rec = min_conf_thresh_text_rec 
        
        ## Path to root directory:
        self._root = Path("/app/uploads")
        
    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self, crop_path: str, book_id: str) -> Tuple[str, float]:
        """
        Run OCR on the crop at `crop_path` and return structured results.

        See BaseOCRService.extract() for the full contract.
        """
        # ── Lazy-load the PaddleyOCR text region detector ───────────────────────────────────
        # First call triggers model download (~100 MB).
        # All subsequent calls reuse the cached detector.
        if self._detector is None:
            self._detector = self._load_text_region_detector()
        
        if self._reader is None:
            self._reader = self._load_text_recognizer()

        # ── Step 1: Load the crop image from disk ──────────────────────────
        crop_image = self._load_crop(crop_path)

        logger.info(
            "book_spine_ocr_started",
            book_id=book_id,
            crop_path=crop_path,
            image_size=crop_image.size,
        )

        # ── Step 2: Preprocess the crop for OCR ───────────────────────────
        # Upscale narrow spines, fix orientation, normalise contrast
        # NOTE: We only upscaling the narrown spines as PaddleOCR is great with unprocessed images
        prepared_image = self._preprocessor.prepare(crop_image)

        
        # PaddleOCR TextDetection's predict() accepts numpy arrays (H, W, C) in uint8 RGB.
        image_array = np.array(prepared_image)

        # ── Step 3: Run PaddleOCR inference - Detection + Recognition ─────────────────────────────
        # -----------------------------
        # Run Text Detection
        # -----------------------------
        det_result = self.get_book_text_detection(image_array, book_id, crop_path)
        
        ## ──────────────────────────────────────────────────────────────────────────────
        ## Filtering boxes that are close to image border and arranging them in readable order
        filt_det_result = self.filter_boxes_by_min_distance_and_arranging(det_result, 
                                                                          self._min_box_distance_px)
        
        ## Extracting all detected text regions as cropped images
        cropped_text_reg = self._text_reg_extractor.crop_text_regions(filt_det_result["input_img"], 
                                                                      filt_det_result)
        
        
        # -----------------------------
        # Run Text Recognition
        # -----------------------------
        text_rec_result: list[dict] = self.get_book_text_recognition(cropped_text_reg, 
                                                                     book_id, crop_path)
             
        ## Keeping text which is detected with atleast 70% (default) cenfidence score 
        book_spine_text = []
        for item in text_rec_result:
            if item['rec_score'] >= self._min_conf_thresh_text_rec: 
                book_spine_text.append(item['rec_text'])
            
            continue
        
        final_book_spine_text = " ".join(book_spine_text)
        
        
        # ── Step 4: Removing Noisy words from OCR obtained book spine text ──────────────────────
        # -----------------------------  
        final_book_spine_text_cln = self._spine_txt_cleaner.clean_ocr_text(final_book_spine_text)
        
        
        # ── Step 5: Computing a composite OCR score for obtained book spine text ────────────────
        # -----------------------------
        text_scroes = [(item['rec_text'], item['rec_score']) for item in text_rec_result]
                
        comp_score: float = self._compute_confidence(text_scroes)
        
        # ── Finally: Returning the book spine text & OCR Confidence Score ───────────────────────
        return final_book_spine_text_cln, comp_score
    


    def validate(self, 
                 book_spine_text: str,
                 crop_path: str,
                 book_id: str) -> OCRValidatedResult:
    
        # ──  Validating OCR obtained book spine text from Google Book records ────────────
        # -----------------------------      
        if book_spine_text:
            val_ocr_res: OCRValidatedResult = self._validator.validate(book_spine_text,
                                                                       book_id, 
                                                                       crop_path,)
            logger.info(
                "ocr_completed",
                book_id=book_id,
                title=val_ocr_res.title,
                author=val_ocr_res.author,
            )
            
            return val_ocr_res
            
        else:
            logger.info("No OCR text detected from book spine. Provide better picture.")
            # For Reader Pipeline, we omit books without any OCR text.
            # BUT for Librarian Pipeline, we'll flag them for review later while routing
            # Hence we keep a OCRValidatedResult object for such books.
            
            return OCRValidatedResult(
                book_id=book_id,
                crop_image_path = crop_path,
                ori_ocr_ext_spine_txt = book_spine_text,
            )
               
            
            

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_text_region_detector(self):
        """
        Instantiate the PaddleOCR text region detector.

        This is separated from __init__ so the expensive import and model
        download happens lazily (on first extract() call, not at service
        construction). This keeps app startup fast and lets tests construct
        EasyOCRService instances without triggering model downloads.
        """
        from paddleocr import TextDetection 
        
        logger.info("paddleocr_text_region_detector_loading", gpu=self._gpu)

        # -----------------------------
        # Initialize the Text Detector
        # -----------------------------
        try:
            text_detector = TextDetection(
                model_name="PP-OCRv5_mobile_det",
                model_dir="/app/models/paddleocr/PP-OCRv5_mobile_det",
                device=self._device,
                thresh=self._detector_threshold,                     
                                                # Lower value → more pixels considered text
                                                # Higher value → only very confident pixels kept
                
                box_thresh=self._detector_box_threshold,                 
                                                # Lower value → more boxes (including low-confidence ones)
                                                # Higher value → fewer boxes (only high-confidence ones)
                
                unclip_ratio=self._detector_unclip_ratio,               
                                                # Lower value → tighter boxes around text
                                                # Higher value → looser boxes that may include more background
                )
        except Exception as exc:
                logger.info("paddleocr_text_region_detector_model_loading_failed")      

                raise OCRError(
                    f"failed loading PaddleOCR TextDetection for text region detection: {exc}",
                    detail={"model_name": "PP-OCRv5_mobile_det"},
                ) from exc

        logger.info("paddleocr_text_region_detector_ready")      
        return text_detector



    def _load_text_recognizer(self):
        """
        Instantiate the EasyOCR Reader with the configured languages.

        This is separated from __init__ so the expensive import and model
        download happens lazily (on first extract() call, not at service
        construction). This keeps app startup fast and lets tests construct
        EasyOCRService instances without triggering model downloads.
        """
        from paddleocr import TextRecognition
        
        logger.info("paddleocr_text_recognizer_loading", gpu=self._gpu)

        # -----------------------------
        # Initialize the Text Detector
        # -----------------------------
        try:
            text_reader = TextRecognition(
                model_name="PP-OCRv3_mobile_rec",
                model_dir="/app/models/paddleocr/PP-OCRv3_mobile_rec",
                device=self._device,
                )
        except Exception as exc:
                logger.info("paddleocr_text_recognition_model_loading_failed")      

                raise OCRError(
                    f"Failed loading PaddleOCR TextRecognition for text recognition: {exc}",
                    detail={"model_name": "PP-OCRv3_mobile_rec"},
                ) from exc


        logger.info("paddleocr_text_recognizer_ready")      
        return text_reader
    
    

    def _load_crop(self, crop_path: str) -> PILImage.Image:
        """
        Load the crop image from the given path.

        Converts to RGB immediately to guarantee the mode expected by the
        preprocessor and by numpy array conversion downstream.

        Raises:
            OCRError: If the file does not exist or cannot be opened.
        """
        path = self._root / crop_path

        if not path.exists():
            raise OCRError(
                f"Crop image not found: {str(crop_path)}",
                detail={"crop_path": str(crop_path)},
            )

        try:
            image = PILImage.open(path).convert("RGB")
            return image
        except Exception as exc:
            raise OCRError(
                f"Could not open crop image at {str(crop_path)}: {exc}",
                detail={"crop_path": str(crop_path)},
            ) from exc


    def get_book_text_detection(self, img_array: np.array, book_id: str, crop_path: str) -> dict:
        """
        We'll run text detection for two image orientations - at 0 deg and at 180 deg
        We keep the one where most text regions are detected.  
        """
        final_result = None
        det_results = []
        img = PILImage.fromarray(img_array)
        
        for angle in [0, 180]:
            rotated_img = img.rotate(angle, expand=True)
            
            # Convert PIL → numpy array for PaddleOCR text region detection
            img_arr = np.array(rotated_img)
            
            try:
                result = self._detector.predict(img_arr)
            except Exception as exc:
                raise OCRError(
                    f"Text Detection failed for book {book_id}: {exc}",
                    detail={"book_id": book_id, "crop_path": crop_path},
                ) from exc
                
            # Appening results to det_results
            det_results.append((angle, result[0]))


        ## ──────────────────────────────────────────────────────────────────────────────
        ## Selecting the best result based on number of detected boxes and confidence scores
        result_0, result_180 = [det[1] for det in det_results]
        
        if len(result_0['dt_scores']) == len(result_180['dt_scores']):           
            avg_conf1 = np.mean(result_0['dt_scores'])
            avg_conf2 = np.mean(result_180['dt_scores'])
            
            if avg_conf1 >= avg_conf2:
                final_result = result_0
            else:
                final_result = result_180

        elif len(result_0['dt_scores']) > len(result_180['dt_scores']):
            final_result = result_0
            
        else:
            final_result = result_180
            
        return final_result



    def get_book_text_recognition(self, 
                                  cropped_text_reg: list[tuple], 
                                  book_id: str, 
                                  crop_path: str,
                                  ) -> list[dict]:
        """
        Run text recognition for two image orientations - at 0 deg and at 180 deg
        NOTE: Recognition confidence score for correctly oriented text is always high.
        So, we'll keep the text recognition result where average confidence score is highest. 

        Raises:
            OCRError: error raised if text recognition fails
        """
        final_rec_result = None
        
        processed_0 = [item[0] for item in cropped_text_reg]     # list of all cropped text region images at 0 deg rotation
        processed_180 = []                                       # list of all cropped text region images at 180 deg rotation

        for img, pos in cropped_text_reg:
            img_180 = np.array(PILImage.fromarray(img).rotate(180, expand=True))
            processed_180.append(img_180)

        ## Recognising text for images at 0 deg and 180 deg both:
        try:
            result_0 = self._reader.predict(processed_0)
            result_180 = self._reader.predict(processed_180)
        except Exception as exc:
                raise OCRError(
                    f"Text Recognition failed for book {book_id}: {exc}",
                    detail={"book_id": book_id, "crop_path": crop_path},
                ) from exc
            
        ## Computing average confidence scores in both the orientations:
        avg_conf_0 = np.mean([res['rec_score'] for res in result_0])
        avg_conf_180 = np.mean([res['rec_score'] for res in result_180])
        
        if avg_conf_0 > 0.5 and avg_conf_0 > avg_conf_180:
            final_rec_result = result_0
        else:
            result_180.reverse()
            final_rec_result = result_180
            
        return final_rec_result


    # ── Filtering boxes touching book crop borders ───────────────────────────────────────────────

    def filter_boxes_by_min_distance_and_arranging(self, text_det_result: dict, min_dist_px: int = 5):
        """
        We need to filter and keep only those Detected Text Regions which are away from Book spine borders.
        Detected Text Regions which are closer to book spine borders are likely be text regions on 
        adjacent books rather than the target book spine.
        
        We also need to arrange the text regions in a sentence like fashion, so they make a meaningful text.
        For arrangment we'll compute the centroids of each detected box and find centroids position along on x-axis.
        Then arranging the boxes in order of their centroids position on x-axis to maintain the reading order from left to right.

        Args:
            text_det_result (dict): Final Text Detection Result
            min_dist_px (int, optional): Minimum distance that boxes need to maintain from image border. 
                                         Defaults to 5.

        Returns:
            dict: Filtered and Enhanced Text Detection Result
        """
        
        text_det_result = text_det_result.copy()                # Avoid modifying original
        H, W = text_det_result['input_img'].shape[:2]

        filtered_boxes = []
        box_position_on_x_axis = []

        ## Filtering boxes based on minimum distance from borders and computing centroids:
        for pts, scores in zip(text_det_result['dt_polys'], text_det_result['dt_scores']):
            
            ## Minimum distance of each point to borders
            distances = []

            for x, y in pts:
                d_left, d_right, d_top, d_bottom = x, W - x, y, H - y
                distances.append(min(d_left, d_right, d_top, d_bottom))

            min_dist = min(distances)
            
            if min_dist < min_dist_px:
                continue

            filtered_boxes.append((pts, scores))

            ## Now computing the centroid of the box and its position on x-axis:
            x_coords = [x for x, y in pts]
            centroid_x = sum(x_coords) / len(x_coords)
            box_position_on_x_axis.append(centroid_x)

        ## Sorting the filtered boxes based on their centroids position on x-axis:  
        position_on_x_axis = box_position_on_x_axis.copy()
        for i, rank in zip(sorted(box_position_on_x_axis), range(len(box_position_on_x_axis))): 
            position_on_x_axis[box_position_on_x_axis.index(i)] = rank
            
        # Update the text detection result with filtered boxes
        text_det_result['dt_polys'] = [item[0] for item in filtered_boxes]
        text_det_result['dt_scores'] = [item[1] for item in filtered_boxes]
        text_det_result['box_position_on_x_axis'] = box_position_on_x_axis
        text_det_result['position_rank_on_x_axis'] = position_on_x_axis

        return text_det_result



    # ── Confidence scoring ─────────────────────────────────────────────────────

    def _compute_confidence(self, cleaned_texts: list[tuple[str, float]]) -> float:
        """
        Compute a composite confidence score for the full spine OCR result.

        Strategy: weighted average of individual block confidence scores,
        where longer text blocks get higher weight (a 10-char detection is
        more informative than a 2-char one).

        Returns a float in [0.0, 1.0].
        """
        if not cleaned_texts:
            return 0.0

        # Weight each block's confidence by its text length
        total_weight = 0.0
        weighted_sum = 0.0

        for text, confidence in cleaned_texts:
            weight = max(len(text), 1)   # Minimum weight of 1
            weighted_sum += confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        raw_confidence = weighted_sum / total_weight

        # Apply a length penalty: very short total text (< 5 chars) is suspect
        total_text_length = sum(len(t) for t, _ in cleaned_texts)
        if total_text_length < 5:
            raw_confidence *= 0.5   # Halve confidence for very short text

        # Clamp to [0, 1] to guard against any floating-point edge cases
        return round(max(0.0, min(1.0, raw_confidence)), 3)