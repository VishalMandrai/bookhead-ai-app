# ─────────────────────────────────────────────────────────────────────────────
# app/services/llm_parser.py
#
# Parses raw LLM text responses into typed Pydantic models.
#
# WHY A SEPARATE PARSER MODULE?
#   The LLM is instructed to return JSON, but in practice it may:
#     - Wrap the JSON in markdown fences (```json ... ```)
#     - Return extra prose before or after the JSON
#     - Emit slightly different key names than instructed
#     - Include null fields, missing fields, or wrong types
#
#   Centralising this messy boundary logic here keeps the service classes
#   clean and makes the parsing independently testable with synthetic strings.
#
# DESIGN: fail-soft parsing
#   Rather than crashing on every minor LLM deviation, we:
#     1. Strip markdown fences and surrounding prose
#     2. Parse the JSON
#     3. Build Pydantic models with field-level defaults for missing keys
#     4. Log warnings for unexpected shapes rather than raising immediately
#   A hard LLMError is raised only when the response cannot be parsed as
#   JSON at all — that indicates a prompt / model failure worth surfacing.
#
# SOLID note (Single Responsibility):
#   This module owns the LLM output → Python types boundary.
#   Service classes are kept ignorant of raw string manipulation.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import re
from typing import Any

from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.models.book import OCRValidatedResult, BookRecord, BookRecommendation

logger = get_logger(__name__)

# Regex to strip markdown code fences that Claude sometimes wraps JSON in.
# Matches ```json ... ``` or ``` ... ``` with any amount of whitespace.
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


# ── Low-level helpers ──────────────────────────────────────────────────────────

def extract_json(raw_text: str) -> dict:
    """
    Extract and parse a JSON object from raw LLM response text.

    Handles three common LLM response shapes:
      1. Pure JSON:        {"key": "value"}
      2. Fenced JSON:      ```json\n{"key": "value"}\n```
      3. Prose + JSON:     "Here is the result:\n{"key": "value"}"

    Args:
        raw_text: Raw string returned by AnthropicLLMClient.complete().

    Returns:
        Parsed dict.

    Raises:
        LLMError: If no valid JSON object can be extracted.
    """
    # Step 1: Try stripping markdown fences first
    fence_match = _FENCE_RE.search(raw_text)
    candidate = fence_match.group(1) if fence_match else raw_text

    # Step 2: Find the first '{' and last '}' to isolate the JSON object
    # This strips any prose prefix/suffix that the LLM emitted
    start = candidate.find("{")
    end   = candidate.rfind("}")

    if start == -1 or end == -1:
        raise LLMError(
            "LLM response contained no JSON object.",
            detail={"raw_preview": raw_text[:200]},
        )

    json_str = candidate[start : end + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"LLM response contained malformed JSON: {exc}",
            detail={"json_preview": json_str[:300]},
        ) from exc


# ── Heavy LLM Recommendation response parser ───────────────────────────────────────

class RecommendationResponseParser:
    """
    Parses the LLM's recommendation JSON into a list of BookRecommendation.

    Expected input shape:
    {
      "recommendations": [
        {
          "book_id": "...",
          "title": "...",
          "author": "...",
          "rank": 1,
          "summary": "...",
          "match_reason": "...",
          "genre": "...",
          "genre_code": "...",
          "year_published": 1965
        },
        ...
      ]
    }
    """

    def parse(
        self,
        raw_text: str,
        original_books: list[BookRecord],
    ) -> list[BookRecommendation]:
        """
        Parse the LLM response and return a list of BookRecommendation objects.

        The `original_books` list is used to:
          - Validate that returned book_ids match what was sent to the LLM
          - Fall back to the OCR title/author if the LLM omits a field
          - Recover the crop_image_path (not sent to LLM, recovered here)

        Args:
            raw_text:       Raw string from AnthropicLLMClient.complete().
            original_books: The books that were sent to the LLM.

        Returns:
            List of BookRecommendation objects, sorted by rank ascending.

        Raises:
            LLMError: If the response cannot be parsed as JSON.
        """
        data = extract_json(raw_text)

        # Build a lookup from book_id → BookRecord for quick validation
        books_by_id: dict[str, BookRecord] = {b.book_id: b for b in original_books}

        raw_recs = data.get("recommendations", [])
        if not raw_recs:
            logger.warning("llm_recommendation_parser_empty_list", raw_preview=raw_text[:200])
            return []

        recommendations: list[BookRecommendation] = []

        for item in raw_recs:
            book_id = item.get("book_id", "")        # Take book_id of Book item returned in recommendation
            original = books_by_id.get(book_id)      # Check if it exist in original book records 

            if not original:                         # If it don't raise a warning.
                # LLM returned a book_id which don't match with any existing book_ids, 
                # So, we don''t send it as recommendation — skip it with a warning
                logger.warning(
                    "llm_recommendation_unknown_book_id",
                    book_id=book_id,
                )
                continue

            try:
                rec = BookRecommendation(
                    book_id=book_id,
                    # LLM may have corrected garbled OCR text — prefer its version
                    title=item.get("title") or original.title,
                    author=item.get("author") or original.author,
                    # crop_image_path is never sent to the LLM; recover it here
                    crop_image_path=original.crop_image_path,
                    summary=item.get("summary", "No summary available."),
                    rank=int(item.get("rank", len(recommendations) + 1)),
                    match_reason=item.get("match_reason", ""),
                )
                recommendations.append(rec)
            except Exception as exc:
                # Log a warning and skip malformed entries rather than crashing
                logger.warning(
                    "llm_recommendation_parse_error",
                    book_id=book_id,
                    error=str(exc),
                )

        # Sort by rank so the best recommendation is always first
        recommendations.sort(key=lambda r: r.rank)

        logger.info(
            "llm_recommendation_parsed",
            total_returned=len(raw_recs),
            total_valid=len(recommendations),
        )

        return recommendations

    def enrich_book_records(
        self,
        raw_text: str,
        original_books: list[BookRecord],
    ) -> list[BookRecord]:
        """
        Update original BookRecord objects with LLM-enriched metadata.

        After parsing recommendations, we can back-fill genre/year/isbn into
        the BookRecord objects before writing them to the vector store.

        Args:
            raw_text:       Raw LLM response text.
            original_books: BookRecord objects to update in-place.

        Returns:
            Updated list of BookRecord objects.
        """
        try:
            data = extract_json(raw_text)
        except LLMError:
            return original_books   # Best-effort: return unchanged on parse failure

        enrichment_by_id: dict[str, dict] = {
            item["book_id"]: item
            for item in data.get("recommendations", [])
            if "book_id" in item
        }

        for book in original_books:
            enrichment = enrichment_by_id.get(book.book_id, {})
            if enrichment:
                book.genre         = enrichment.get("genre") or book.genre
                book.genre_code    = enrichment.get("genre_code") or book.genre_code
                book.summary       = enrichment.get("summary") or book.summary
                book.year_published = enrichment.get("year_published") or book.year_published

        return original_books

# ── Gemma-2B Summary parser and Info Enricher ─────────────────────────────────


class GemmaSummaryParser_InfoEnricher:
    """
    Parses the Gemma's Text Summarization JSON response into clean summary text.
    Uses about_book field within OCRValidatedResult of given book to create and 
    enrich BookRecord instance of given book.
    
    """
    
    def extract_summary(self, llm_resp: dict) -> str:
        """
        Extract and parse book summary from raw LLM response text.

        Args:
            llm_resp: Raw response body returned by Gemma-2B LLM

        Returns:
            summary text

        Raises:
            LLMError: If no summary text is found.
        """
        # STEP 1: Get useful part of LLM reponse
        text = llm_resp['choices'][0]['text']

        # STEP 2: Remove all the noisy characters
        _NOISE_CHARS = re.compile(r"[|\\^~`_<>{}【】「」『』《》〔〕\n *]")
        
        text = _NOISE_CHARS.sub(" ", text)
        cleaned = re.sub(r"\s+", " ", text)
        
        # STEP 3: Find summary start    
        for search in ["summary: ", "summary is: ", "summary - ", "summary is - "]:
            start = cleaned.lower().find(search)
            if start != -1 and len(cleaned[start + len(search) : ]) > 15:
                final_txt_summ = cleaned[start + len(search) : ]           # Index out full summary text
                break
        
        try:
            if final_txt_summ:
                pass
        except:
            final_txt_summ = cleaned
            
            logger.warning("Summary Text don't start with - ('summary: ', 'summary is: ', 'summary - ', 'summary is - '). Look for better strings to parse out actual summary text only.")
       
        
        # STEP 3: Check if it contains proper sentences and have sentence casing
        tot_sent = len(re.findall(f"\\.", final_txt_summ))

        if tot_sent >= 2 and final_txt_summ[0].isupper() and final_txt_summ.endswith("."):
            return final_txt_summ
        
        else:
            raise LLMError(f"LLM response contained no useful summary.")


    def enrich_book_records(
        self,
        book_summary: str,
        original_book: OCRValidatedResult,
    ) -> BookRecord:
        """
        Create BookRecord and enrich it with info from OCRValidatedResult
        instance of original book.

        After enrichment we can insert the book to Vector DB.

        Args:
            book_summary:   Book summary generated by Gemma LLM.
            original_book:  OCRValidatedResult instance of original book.

        Returns:
            New BookRecord object.
        """
        book = BookRecord(
            book_id = original_book.book_id,
            title = original_book.title,
            subtitle = original_book.subtitle,
            author = original_book.author,
            crop_image_path = original_book.crop_image_path,
            ocr_confidence = original_book.ocr_confidence,
            ori_ocr_ext_spine_txt = original_book.ori_ocr_ext_spine_txt,

            pagecount = original_book.about_book['pageCount'],
            maturityRating = original_book.about_book['maturityRating'],
            ratings = original_book.about_book['averageRating'],
            description = original_book.description,
            summary = book_summary,
            isbn = original_book.about_book['isbn'],
            publisher = original_book.about_book['publisher'],
            year_published = original_book.about_book['publishedDate'],
        )
        
        return book




# ── Catalog response parser ───────────────────────────────────────────────────

class CatalogResponseParser:
    """
    Parses the LLM's catalog JSON into enriched BookRecord objects.

    Expected input shape:
    {
      "catalog": [
        {
          "book_id": "...",
          "title": "...",
          "author": "...",
          "genre": "...",
          "genre_code": "...",
          "summary": "...",
          "year_published": 1965,
          "isbn": "978-..."
        },
        ...
      ]
    }
    """

    def parse(
        self,
        raw_text: str,
        original_books: list[BookRecord],
    ) -> list[BookRecord]:
        """
        Parse the LLM catalog response and return enriched BookRecord objects.

        For each entry in the LLM response, we find the matching BookRecord by
        book_id and update it with the LLM-provided metadata. Books the LLM
        omits are returned with their original (unenriched) data so no books
        are silently dropped.

        Args:
            raw_text:       Raw string from AnthropicLLMClient.complete().
            original_books: The BookRecord objects sent to the LLM.

        Returns:
            Updated BookRecord list (same length as original_books).

        Raises:
            LLMError: If the response cannot be parsed as JSON.
        """
        data = extract_json(raw_text)

        catalog_items = data.get("catalog", [])
        if not catalog_items:
            logger.warning("llm_catalog_parser_empty_list")
            return original_books

        # Build lookup: book_id → LLM enrichment dict
        enrichment_map: dict[str, dict] = {
            item["book_id"]: item
            for item in catalog_items
            if "book_id" in item
        }

        enriched_records: list[BookRecord] = []
        for book in original_books:
            enrichment = enrichment_map.get(book.book_id, {})

            if enrichment:
                # Update the existing BookRecord with LLM-provided metadata
                # LLM may have corrected OCR errors in title/author
                book.title          = enrichment.get("title") or book.title
                book.author         = enrichment.get("author") or book.author
                book.genre          = enrichment.get("genre") or book.genre
                book.genre_code     = enrichment.get("genre_code") or book.genre_code
                book.summary        = enrichment.get("summary") or book.summary
                book.year_published = enrichment.get("year_published") or book.year_published
                book.isbn           = enrichment.get("isbn") or book.isbn
            else:
                # LLM skipped this book — log a warning, keep original data
                logger.warning(
                    "llm_catalog_missing_book",
                    book_id=book.book_id,
                    title=book.title,
                )

            enriched_records.append(book)

        logger.info(
            "llm_catalog_parsed",
            total_input=len(original_books),
            total_enriched=len(enrichment_map),
        )

        return enriched_records
