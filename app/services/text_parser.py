# ─────────────────────────────────────────────────────────────────────────────
# app/services/text_parser.py
#
# Parses raw EasyOCR output (a list of text bounding boxes) into structured
# (title, author) fields, and computes a composite confidence score.
#
# WHY A SEPARATE PARSER MODULE?
#   EasyOCR returns a flat list of detected text regions with positions:
#     [([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "Some text", 0.85), ...]
#   This raw output needs to be:
#     1. Sorted by spatial position (top → bottom or left → right)
#     2. Cleaned of noise characters and common OCR artefacts
#     3. Heuristically split into title vs author
#     4. Confidence-scored as a single composite float
#
#   Keeping this logic here (not inside EasyOCRService) means:
#     - It's independently unit-testable with synthetic OCR output
#     - EasyOCRService stays focused on model inference only
#     - A different parser strategy (e.g. LLM-based) can be swapped in
#       without touching the OCR model code
#
# SOLID (Single Responsibility + Open/Closed):
#   SpineTextParser owns text interpretation. A new parsing strategy
#   (e.g. SpineTextLLMParser) can implement the same interface without
#   modifying this file.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
import string
from app.core.logging import get_logger
from dataclasses import dataclass
from app.models.book import OCRValidatedResult

logger = get_logger(__name__)



# ── Types ─────────────────────────────────────────────────────────────────────

# Raw EasyOCR result for a single detected text region:
#   (bounding_box_coords, text_string, confidence_score)
# bounding_box_coords = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]  (4 corners)
EasyOCRBox = tuple[list[list[float]], str, float]


@dataclass
class ParsedSpineText:
    """
    Structured result of parsing the OCR output for a single book spine.

    Attributes:
        title:      Best-guess book title (may be empty if not detected).
        author:     Best-guess author name (may be empty if not detected).
        confidence: Composite confidence score (0.0–1.0).
        raw_blocks: The original EasyOCR boxes, preserved for debugging.
    """
    title: str
    author: str
    confidence: float
    raw_blocks: list[EasyOCRBox]


# ── Noise patterns to strip from OCR text ─────────────────────────────────────

# Characters that are almost always OCR noise on book spines
_NOISE_CHARS = re.compile(r"[|\\^~`_<>{}【】「」『』《》〔〕]")

# ISBN / price / barcode patterns — not useful as title or author
_ISBN_PATTERN   = re.compile(r"\b(?:isbn[-:]?\s*)?97[89][\d\-]{10,}\b", re.IGNORECASE)
_PRICE_PATTERN  = re.compile(r"\b(?:rs\.?|₹|\$|€|£)\s*\d+(?:\.\d{2})?\b", re.IGNORECASE)
_BARCODE_DIGITS = re.compile(r"^\d{8,}$")   # 8+ consecutive digits = barcode

# Strings shorter than this are likely OCR fragments, not real words
_MIN_WORD_LENGTH = 2


# ── SpineTextParser ───────────────────────────────────────────────────────────

class SpineTextParser:
    """
    Converts EasyOCR's raw box output into a ParsedSpineText.

    Parsing strategy:
      1. Filter out noise/barcode/ISBN blocks
      2. Sort remaining blocks by their vertical centroid (top → bottom)
      3. Concatenate spatially-grouped lines into candidate strings
      4. Apply title/author heuristics:
           - The longest meaningful text block is the title
           - Shorter text with recognisable author patterns (e.g. "J. Smith",
             "by Smith") is the author
           - If only one block remains, it is treated as the title
      5. Compute composite confidence as a weighted average of block scores

    This is intentionally heuristic — it will not be perfect for every spine.
    Low-confidence results will be routed to the human review queue, where
    a librarian provides the ground truth.
    """

    def parse(self, ocr_boxes: list[EasyOCRBox]) -> ParsedSpineText:
        """
        Parse a list of EasyOCR detection boxes into structured title + author.

        Args:
            ocr_boxes: Raw EasyOCR output for one spine crop.
                       Each element is (coords, text, confidence).

        Returns:
            ParsedSpineText with title, author, confidence, and raw_blocks.
        """
        # ── Edge case: no text detected at all ────────────────────────────
        if not ocr_boxes:
            return ParsedSpineText(
                title="",
                author="",
                confidence=0.0,
                raw_blocks=[],
            )

        # ── Step 1: Filter out noise blocks ───────────────────────────────
        clean_boxes = [box for box in ocr_boxes if self._is_meaningful(box)]

        if not clean_boxes:
            # All blocks were noise — return zero-confidence result
            return ParsedSpineText(
                title="",
                author="",
                confidence=0.0,
                raw_blocks=ocr_boxes,
            )

        # ── Step 2: Sort by vertical position (top → bottom) ──────────────
        # The vertical centroid of each box is the average y-coordinate of its
        # four corners. Sorting by this places title text (usually at top of
        # spine) before author text (usually at bottom).
        sorted_boxes = sorted(clean_boxes, key=lambda b: self._vertical_centroid(b))

        # ── Step 3: Clean the text in each box ────────────────────────────
        cleaned_texts = [
            (self._clean_text(box[1]), box[2])   # (text, confidence)
            for box in sorted_boxes
        ]
        # Remove any that became empty after cleaning
        cleaned_texts = [(t, c) for t, c in cleaned_texts if t.strip()]

        if not cleaned_texts:
            return ParsedSpineText(
                title="", author="", confidence=0.0, raw_blocks=ocr_boxes
            )

        # ── Step 4: Assign title and author ───────────────────────────────
        title, author = self._assign_title_author(cleaned_texts)

        # ── Step 5: Compute composite confidence ──────────────────────────
        confidence = self._compute_confidence(cleaned_texts)

        return ParsedSpineText(
            title=title,
            author=author,
            confidence=confidence,
            raw_blocks=ocr_boxes,
        )

    # ── Filtering helpers ──────────────────────────────────────────────────────

    def _is_meaningful(self, box: EasyOCRBox) -> bool:
        """
        Return True if this OCR box likely contains real title/author text.
        Filters out barcodes, ISBNs, prices, and very short fragments.
        """
        _, text, confidence = box

        # Discard very low-confidence detections outright
        if confidence < 0.1:
            return False

        stripped = text.strip()

        # Empty or single character
        if len(stripped) < _MIN_WORD_LENGTH:
            return False

        # Pure barcode digits
        if _BARCODE_DIGITS.match(stripped):
            return False

        # ISBN numbers
        if _ISBN_PATTERN.search(stripped):
            return False

        # Price tags
        if _PRICE_PATTERN.search(stripped):
            return False

        return True

    # ── Geometry helpers ───────────────────────────────────────────────────────

    def _vertical_centroid(self, box: EasyOCRBox) -> float:
        """
        Return the average y-coordinate of the four bounding-box corners.

        EasyOCR's bounding box format:
          [[x_top_left, y_top_left],
           [x_top_right, y_top_right],
           [x_bottom_right, y_bottom_right],
           [x_bottom_left, y_bottom_left]]
        """
        coords = box[0]   # List of 4 [x, y] pairs
        return sum(corner[1] for corner in coords) / 4.0

    # ── Text cleaning ──────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """
        Remove noise characters and normalise whitespace.

        Steps:
          1. Strip noise characters (pipes, brackets, etc.)
          2. Collapse multiple spaces
          3. Strip leading/trailing whitespace
          4. Title-case the result for consistency
        """
        # Remove known noise characters
        cleaned = _NOISE_CHARS.sub(" ", text)
        # Collapse multiple consecutive spaces into one
        cleaned = re.sub(r"\s+", " ", cleaned)
        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()
        return cleaned

    # ── Title / author assignment ──────────────────────────────────────────────

    def _assign_title_author(
        self,
        cleaned_texts: list[tuple[str, float]],
    ) -> tuple[str, str]:
        """
        Heuristically assign cleaned text blocks to title and author.

        Rules (in priority order):
          1. If only one block: it's the title.
          2. If a block matches an author pattern (see _looks_like_author):
             it's the author; the longest remaining block is the title.
          3. Fallback: first block = title, second block = author.

        Returns:
            (title, author) — either may be an empty string.
        """
        texts_only = [t for t, _ in cleaned_texts]

        if len(texts_only) == 1:
            return texts_only[0], ""

        # Look for a block that matches author-name patterns
        author_candidates = [t for t in texts_only if self._looks_like_author(t)]

        if author_candidates:
            # Use the first author candidate
            author = author_candidates[0]
            # Title = longest non-author block
            non_author = [t for t in texts_only if t != author]
            title = max(non_author, key=len) if non_author else ""
            return title, author

        # Fallback: longest block = title, second longest = author
        sorted_by_len = sorted(texts_only, key=len, reverse=True)
        title  = sorted_by_len[0]
        author = sorted_by_len[1] if len(sorted_by_len) > 1 else ""
        return title, author

    def _looks_like_author(self, text: str) -> bool:
        """
        Return True if `text` looks like an author name rather than a title.

        Heuristics:
          - Contains "by " prefix (e.g. "by J.K. Rowling")
          - Matches "Firstname Lastname" with 2–4 words, each word capitalised
          - Contains initials pattern (e.g. "J.K." or "R.R.")
          - Shorter than 5 words (titles tend to be longer or more varied)
        """
        stripped = text.strip()
        lower = stripped.lower()

        # "by <name>" is an explicit author marker
        if lower.startswith("by "):
            return True

        words = stripped.split()
        word_count = len(words)

        # Too long to be an author name
        if word_count > 5:
            return False

        # Too short (single word could be either)
        if word_count < 2:
            return False

        # All words are capitalised (typical of "First Last" author format)
        all_capitalised = all(
            w[0].isupper() for w in words if len(w) > 1
        )

        # Contains initialised first name (e.g. "J.K. Rowling", "R.R. Martin")
        has_initials = any(re.match(r"^[A-Z]\.$", w) for w in words)

        return all_capitalised or has_initials

    # ── Confidence scoring ─────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        cleaned_texts: list[tuple[str, float]],
    ) -> float:
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
        return max(0.0, min(1.0, raw_confidence))

# ─────────────────────────────────────────────────────────────────────────────────────────────

class SpineTextCleaner:
    """
    Removes all unwanted words from OCR Spine Text.
    Helps in better look up into Google Book records.
    """

    UNWANTED_BOOK_TEXT = [
        # Bestseller / Marketing
        "bestseller", "best seller", "best-selling", "bestselling", "international bestseller",
        "new york times bestseller", "usa today bestseller", "usa today bestselling", "award-winning",
        "award winning", "must-read", "must read", 
        # Edition / Format
        "edition", "revised edition", "updated edition", "expanded edition", "paperback", "hardcover",
        "softcover", "mass market paperback", 
        # Metadata
        "isbn", "isbn-10", "isbn-13", "ean", "barcode", "published by", "publisher", 
        # Author roles
        "author", "edited by", "editor", "translated by", "illustrated by", "foreword", "afterword", 
        "introduction",
        # Series / volume
        "volume", "vol", "book",
        # OCR garbage
        "™", "®", "©",
    ]


    def clean_ocr_text(self, text: str) -> str:
        """
        Removes unwanted book-related noise words
        from OCR extracted text.
        """
        
        # Lowercase normalization
        cleaned = text.lower()

        # Sort by length descending
        # Prevent partial matching issues
        unwanted_words = sorted(self.UNWANTED_BOOK_TEXT, key=len, reverse=True)

        for word in unwanted_words:
            escaped_word = re.escape(word)          # Escape regex special chars
            pattern = rf"\b{escaped_word}\b"        # Word boundary matching
            cleaned = re.sub( pattern, " ", cleaned, flags=re.IGNORECASE)

        # Collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned.upper()





# ─────────────────────────────────────────────────────────────────────────────────────────────

class SpineTextValidator:
    """
    Takes PaddleOCR recognised text from book spines and looks up Google Book records
    to get correct book title and author name. 
    Also bring in other information around that book like - book lenght, published data, genre etc.

    Enrichment strategy:
      1. Make an API call to Google Book API
      2. Parse the API response JSON to get necessary information.
      3. Apply Fuzzy Logic to select best suitable book record from API response
      4. Compute Composite confidence score for OCR results as a weighted average of text recognition scores
      5. Create and return OCRResult pydantic object
      
    This is combination of coherent validation from clean book records and heuristic — 
    it may not be perfect for every spine but largely robust.
    
    In Librarian Pipeline: Low-confidence results will be routed to the human review queue, where
    a librarian provides the ground truth.
    """
        
    def __init__(self,
                 api_key: str) -> None:
        self._api_key = api_key
        return None
    
    def validate(self,
                 book_spine_text: str,
                 book_id: str,
                 crop_path: str,
                 ) -> None:
        
        # --------- Step 1: Fetch book records from Google Books API --------------------
        logger.info("Calling Google Books API to get book records", spine_text =book_spine_text,
                    book_id=book_id, crop_path=crop_path)
        book_data = self.google_api_resp(book_spine_text)
                        
        if not book_data:
            logger.info("No book records returned from Google Books API", 
                        book_id=book_id, crop_path=crop_path)
    
            return OCRValidatedResult(
                book_id = book_id,
                crop_image_path = crop_path,
                ori_ocr_ext_spine_txt=book_spine_text,
                book_info = {"pageCount":None, "genre":None, "maturityRating":None, 
                             "publisher":None, "publishedDate":None, "isbn":None, 
                             "averageRating":1},
            )
        
        # --------- Step 2: Apply Fuzzy matching to find the most similar book record ---
        book_title_author: list[tuple] = []
        
        for idx, book in enumerate(book_data):
            try:
                book_title = book['volumeInfo']['title']
                if book_title.lower().find("summary") != -1:
                    continue
            except:
                continue
            try:
                book_authors: list = book['volumeInfo']['authors']
            except:
                book_authors: list = []
                pass
            book_rec = book_title + " " + " ".join(book_authors)
            logger.info(f"{book_rec}")
            book_title_author.append((idx, book_rec))
        
        matching_result: dict = self.fuzzy_matching(book_spine_text, book_title_author)
        
        best_book_ind = matching_result['best_match'][0]
        best_book: dict = book_data.pop(best_book_ind)                 ## ---> BEST Matched Book!
        
        ## There could be more than one record for best matched book:
        books_to_extract_info: list[dict] = []
        books_to_extract_info.append(best_book)
        
        for title, book in [(book['volumeInfo']['title'], book) for book in book_data]:
            if title == best_book['volumeInfo']['title']:
                books_to_extract_info.append(book)
        
        
        # --------- Step 3: Parse out all necessary information from Book record ---
        
        final_result: OCRValidatedResult = self.records_parser(books_to_extract_info, 
                                                               book_spine_text,
                                                               book_id, 
                                                               crop_path)
        
        return final_result
          
    
    def google_api_resp(self, book_spine_text: str, retries: int = 5) -> list:
        import time, random
        import requests
        url = "https://www.googleapis.com/books/v1/volumes"
    
        params = {"q": book_spine_text,
                  "maxResults": 5,
                  "key": self._api_key}
        
        backoff = 1                     # Wait-time before next API call
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    try:
                        if not data["items"]: return {}
                    except:
                        return {}
                    
                    return data['items']
                
                elif response.status_code in [429, 500, 502, 503, 504]:
                    print(f"Retryable error: {response.status_code}")

                else:
                    logger.error("Some error has occured in fetching Book Records from Google API",
                                  status_code = response.status_code)
                    break


            except (requests.ConnectionError, requests.Timeout) as e:
                print(f"Network error: {e}")
            
            if attempt == retries - 1:
                logger.info("Max retries exceeded")

            sleep_time = (backoff + random.uniform(0, 1))
            print(f"Retrying in {sleep_time:.2f} sec")

            time.sleep(sleep_time)
            backoff *= 2
        
        return {}      # Executes only if some problem exists in code...



    def fuzzy_matching(self, book_spine_text: str, books: list[tuple]) -> dict:
        """
        Using fuzzy matching to calculate similarity score between 
        OCR text and API returned Book title + author
        """    
        best_book: tuple = None
        best_score: float = -1
        books_and_score: list[tuple] = []

        for idx, book in books:
            score = self.compute_similarity_score(book_spine_text, book)

            books_and_score.append({"score": score, 
                                    "book": (idx, book)
                                    })
            if score > best_score:
                best_score = score
                best_book = (idx, book)

        ## Sorting in descending order
        books_and_score.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return {"best_match": best_book,
                "best_score": best_score,
                "all_scores": books_and_score
                }


    def records_parser(self, book_records: list[dict], book_spine_text: str,
                       book_id: str, crop_path: str
                       ) -> OCRValidatedResult:
        
        best_book = book_records[0]
        title = best_book['volumeInfo']['title']
        subtitle = best_book['volumeInfo']['subtitle'] if "subtitle" in best_book['volumeInfo'].keys() else ""
        authors = best_book['volumeInfo']['authors'] if "authors" in best_book['volumeInfo'].keys() else []
        
        authors = ", ".join(authors) if authors else ""   # Append all authors into one string
        
        ## Getting all details from book records:        
        book_info: dict = {}      # NOTE: book_info = {pageCount, genre, maturityRating, publisher, 
                                  #                    publishedDate, isbn, averageRating}
        description: str = ""
        
        
        
        for book in book_records:
            details = book['volumeInfo']
            
            ## 1. Adding info of number of pages in book:
            if "pageCount" in details.keys() and details['pageCount'] > 0:
                book_info['pageCount'] = details['pageCount']
            elif "printedPageCount" in details.keys() and details['printedPageCount'] > 0:
                book_info['pageCount'] = details['printedPageCount']
            elif "pageCount" not in book_info:
                book_info['pageCount'] = None
            
            ## 2. Adding book genre category:
            ## NOTE: 1. Later we'll apply fuzzy matching on this genre with our own list
            ##          and then identify the final genre and its genre code.
            ##       2. If no genre catgory is found. Then we'll use LLM feed it book  
            ##          description and ask it choose one.
            if "categories" in details.keys():
                book_info['genre'] = details['categories']
            elif "genre" not in book_info:
                book_info['genre'] = None
            
            ## 3. Adding Adult Rating info of book:
            if "maturityRating" in details.keys():
                book_info['maturityRating'] = details['maturityRating']
            elif "maturityRating" not in book_info:
                book_info['maturityRating'] = None
            
            ## 4. Adding Publisher info of book:
            if "publisher" in details.keys():
                book_info['publisher'] = details['publisher']
            elif "publisher" not in book_info:
                book_info['publisher'] = None
        
            ## 5. Adding Date of Publish of book:
            if "publishedDate" in details.keys():
                book_info['publishedDate'] = details['publishedDate']
            elif "publishedDate" not in book_info:
                book_info['publishedDate'] = None
        
            ## 6. Adding ISBN details of book:
            if "industryIdentifiers" in details.keys():
                isbn_det = details["industryIdentifiers"]
                res = [list(det.values())[0]  + "/" + list(det.values())[1] for det in isbn_det]
                book_info['isbn'] = ", ".join(res)
            elif "isbn" not in book_info:
                book_info['isbn'] = None
        
            ## 7. Adding Book ratings:
            if "averageRating" in details.keys():
                book_info['averageRating'] = details['averageRating']
            elif "averageRating" not in book_info:
                book_info['averageRating'] = 1
                
            ## 8. Getting Book description:
            if description and "description" in details.keys():
                len_curr = len(description.split(" "))
                len_new = len(details['description'].split(" "))
                
                if len_new > len_curr:
                    description = details['description']
    
            elif not description and "description" in details.keys():
                description = details['description']
                
                
            return OCRValidatedResult(book_id = book_id,
                                      crop_image_path = crop_path,
                                      ori_ocr_ext_spine_txt = book_spine_text,
                                      title = title,
                                      subtitle = subtitle, 
                                      author = authors,
                                      description = description,
                                      about_book = book_info)
        
        

    ## ------------------------------- Helper Functions ----------------------------------------
    def normalize_text(self, text: str) -> str:
            """
            TEXT NORMALIZATION: Clean OCR text for better fuzzy matching.
            """
            from unidecode import unidecode
            import re
            
            if not text: return ""

            text = unidecode(text)
            text = text.lower()                        # lowercase
            text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove punctuation
            text = re.sub(r"\s+", " ", text).strip()   # remove multiple spaces

            return text
        
    

    def compute_similarity_score(self, ocr_text: str, book: str) -> float:
        """
        Calculating Book record and OCR Text similarity score.
        """
        from rapidfuzz import fuzz

        ocr_text = self.normalize_text(ocr_text)
        book = self.normalize_text(book)

        # Different fuzzy metrics
        ratio_score = fuzz.ratio(ocr_text, book)
        partial_score = fuzz.partial_ratio(ocr_text, book)
        token_sort_score = fuzz.token_sort_ratio(ocr_text, book)
        token_set_score = fuzz.token_set_ratio(ocr_text, book)

        # Weighted score
        final_score = (
            ratio_score * 0.15            # ratio_score - Good for full-string similarity
            + partial_score * 0.30        # partial_score - Very useful when OCR captures only part of the title.
            + token_sort_score * 0.25     # token_sort_score - Handles word-order problems.
            + token_set_score * 0.20      # token_set_score - Excellent when OCR introduces extra garbage tokens.
        )

        return round(final_score, 2)