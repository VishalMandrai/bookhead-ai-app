# ─────────────────────────────────────────────────────────────────────────────
# app/services/catalog.py
#
# LLMCatalogService – concrete implementation of BaseCatalogService.
# PlainCatalogService - concrete implementation of BaseCatalogService.
#
# RESPONSIBILITY LLMCatalogService:
#   Given a list of BookRecord objects from the merged OCR pipeline, use the
#   LLM (with web search) to enrich each book with genre codes,
#   summaries, ISBNs, and publication years. Also provides CSV export.
# 
# RESPONSIBILITY - PlainCatalogService:
#   Given a list of BookRecords / OCRValidatedResult objects from the merged 
#   OCR pipeline, use "about_info" dict of OCRValidatedResult to create a valid
#   enriched BookRecord. Also provides CSV export.
#
#
# BATCHING STRATEGY:
#   The Anthropic API has a context-window limit. For large library catalogs
#   (potentially 100+ books from many shelf images), we batch books into
#   groups of BATCH_SIZE (default 25) and make one LLM call per batch.
#   Results are merged back into a single list preserving original order.
#
# PIPELINE inside LLMCatalogService - generate_catalog():
#   1. Split books into batches of BATCH_SIZE
#   2. For each batch:
#      a. Build catalog prompt via CatalogPromptBuilder
#      b. Call AnthropicLLMClient (web search enabled)
#      c. Parse response via CatalogResponseParser
#   3. Concatenate all enriched batches
#   4. Return full enriched catalog
#
# SOLID notes:
#   Single Responsibility  – catalog generation + CSV export only.
#   Open/Closed            – batch size is configurable; new export formats
#                            (e.g. JSON, MARC21) can be added as new methods.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Tuple

from app.core.logging import get_logger
from app.models.book import BookRecord, OCRValidatedResult
from app.services.base import BaseCatalogService
from app.services.llm_client import AnthropicLLMClient
from app.services.llm_parser import CatalogResponseParser
from app.services.prompt_builder import CatalogPromptBuilder
from app.services.llm_parser import GemmaSummaryParser_InfoEnricher


logger = get_logger(__name__)

# Maximum books per LLM call — keeps token usage predictable and avoids
# context window overflows for large library batches.
DEFAULT_BATCH_SIZE = 25

# CSV column order for the catalog export
CSV_COLUMNS = [
    "title",
    "subtitle",
    "author",
    "genre",
    "genre_code",
    "pagecount",
    "ratings",
    "publisher",
    "year_published",
    "isbn",
    "description",
    "source",
    "ocr_confidence",
    "book_id",
]


class LLMCatalogService(BaseCatalogService):
    """
    Enriches a book list with genre codes and metadata via the Anthropic LLM,
    and provides CSV export of the resulting catalog.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048,
        batch_size: int = DEFAULT_BATCH_SIZE,
        llm_client: AnthropicLLMClient | None = None,
        prompt_builder: CatalogPromptBuilder | None = None,
        parser: CatalogResponseParser | None = None,
    ) -> None:
        """
        Args:
            api_key:        Anthropic API key.
            model:          Anthropic model identifier.
            max_tokens:     Max tokens per LLM call.
            batch_size:     Books per LLM batch call.
            llm_client:     Injectable LLM client (MockAnthropicLLMClient in tests).
            prompt_builder: Injectable prompt builder.
            parser:         Injectable response parser.
        """
        self._batch_size = batch_size
        self._llm_client = llm_client or AnthropicLLMClient(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            enable_web_search=True,   # Verify titles/authors via web
        )
        self._prompt_builder = prompt_builder or CatalogPromptBuilder()
        self._parser = parser or CatalogResponseParser()

    # ── Public API (implements BaseCatalogService) ─────────────────────────────

    def generate_catalog(self, books: list[BookRecord]) -> list[BookRecord]:
        """
        Enrich books with genre codes and metadata via batched LLM calls.

        Books are processed in batches of `batch_size` to avoid hitting the
        LLM context window. Each batch is enriched independently and merged
        back in original order.

        Returns:
            The same list of BookRecord objects with genre/summary/isbn/year
            fields populated where the LLM could determine them.
        """
        if not books:
            return []

        logger.info(
            "catalog_generation_started",
            total_books=len(books),
            batch_size=self._batch_size,
            num_batches=_ceil_div(len(books), self._batch_size),
        )

        enriched: list[BookRecord] = []
        batches = _chunk(books, self._batch_size)

        for batch_num, batch in enumerate(batches, start=1):
            logger.info(
                "catalog_batch_start",
                batch_num=batch_num,
                batch_size=len(batch),
            )

            # Build prompt for this batch
            system_prompt, user_prompt = self._prompt_builder.build(batch)

            # Call LLM (web search enabled — verifies titles, finds ISBNs)
            raw_response = self._llm_client.complete(system_prompt, user_prompt)

            # Parse enriched records for this batch
            enriched_batch = self._parser.parse(raw_response, batch)
            enriched.extend(enriched_batch)

            logger.info(
                "catalog_batch_complete",
                batch_num=batch_num,
                enriched=len(enriched_batch),
            )

        logger.info(
            "catalog_generation_complete",
            total_enriched=len(enriched),
        )

        return enriched

    def to_csv(self, books: list[BookRecord]) -> str:
        """
        Serialise the enriched catalog to a UTF-8 CSV string.

        The CSV includes a header row and one data row per book.
        Empty/None fields are written as empty strings.

        The output is a plain string — the route handler writes it to a
        temp file and serves it as a download.

        Args:
            books: Enriched BookRecord objects (output of generate_catalog).

        Returns:
            CSV string with header row.
        """
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",   # Silently ignore fields not in CSV_COLUMNS
            lineterminator="\r\n",   # RFC 4180 line endings
        )
        writer.writeheader()

        for book in books:
            row = {
                "title":          book.title or "",
                "author":         book.author or "",
                "genre":          book.genre or "",
                "genre_code":     book.genre_code or "",
                "year_published": book.year_published or "",
                "isbn":           book.isbn or "",
                "summary":        book.summary or "",
                "source":         book.source,
                "ocr_confidence": f"{book.ocr_confidence:.2f}",
                "book_id":        book.book_id,
            }
            writer.writerow(row)

        return buf.getvalue()


# ── Private helpers ────────────────────────────────────────────────────────────

def _chunk(items: list, size: int) -> list[list]:
    """
    Split `items` into consecutive non-overlapping chunks of `size`.

    Example: _chunk([1,2,3,4,5], 2) → [[1,2], [3,4], [5]]
    """
    return [items[i : i + size] for i in range(0, len(items), size)]


def _ceil_div(n: int, d: int) -> int:
    """Integer ceiling division: ceil(n/d) without importing math."""
    return (n + d - 1) // d


# ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────


class PlainCatalogService(BaseCatalogService):
    """
    Enriches a book list with genre codes and metadata via the Anthropic LLM,
    and provides CSV export of the resulting catalog.
    """

    def __init__(
        self,
        emb_model,
        genre_emb_file_path: str,
    ) -> None:
        """
        Args:
            emb_model:        Embedding Model to be used for Genre matching.
            genre_emb_file_path: NumPy file containing list of embeddings for standard genres.
        """
        self._emb_model = emb_model
        self._parser = GemmaSummaryParser_InfoEnricher()
        
        self._genre_emb_file_path = genre_emb_file_path
        

    # ── Public API (implements BaseCatalogService) ─────────────────────────────

    def generate_catalog(self, all_books: list) -> list[BookRecord]:
        """
        Enrich books with genre codes and metadata via "about_info" dict
        of OCRValidatedResult object. 
        
        For genre and genre code consistency across records we'll use Embedding
        Model for Fuzzy matching.

        Returns:
            The list of BookRecord objects with genre/summary/isbn/year
            fields populated.
        """
        if not all_books:
            return []

        logger.info(
            "catalog_generation_started",
            total_books=len(all_books),)

        final_enriched_books: list[BookRecord] = []         # Final records to be returned
        
        
        # ── Step 1: Segregate OCRValidatedResult and BookRecord book objects ────────────
        book_uncached = [book for book in all_books if isinstance(book, OCRValidatedResult)]
        book_cached = [book for book in all_books if isinstance(book, BookRecord)]
        
        # ── Step 2: Parse metadata and get Bookrecord ────────────────────────────────────
        final_enriched_books.extend(book_cached)
        
        semi_enriched_books: list[BookRecord] = []   # Semi-enriched cause they still missing genre details
        
        # Need to get summary for books with OCRValidatedResult data model only
        for book in book_uncached:
            enriched_book = self._parser.enrich_book_records(book_summary="", 
                                                             original_book=book)
            
            enriched_book.source = "human_corrected"
            semi_enriched_books.append(enriched_book)
                

        # ── Step 3: Process raw genre category from OCRValidatedResult data model ────────
        for idx, book in enumerate(book_uncached):
            if book.about_book['genre']:
                # Get genre and genre code using raw genre text:
                code, genre = self.get_genre_via_embedding_model(book.about_book['genre'])
                
                # Update BookRecord:
                book = semi_enriched_books[idx]
                book.genre, book.genre_code = genre, code
                
            elif book.description:
                summary = semi_enriched_books[idx].summary
                
                # Identify genre and genre code using Book Summary:
                code, genre = self.get_genre_via_embedding_model(summary)
                
                # Update BookRecord:
                book = semi_enriched_books[idx]
                book.genre, book.genre_code = genre, code
            else:
                continue
                
        # ── Step 4: Getting final enriched book list having complete BookRecords ready ───
        final_enriched_books.extend(semi_enriched_books)
        
        logger.info(
            "catalog_generation_complete",
            total_enriched=len(final_enriched_books),)

        # ── Finally: Return final enriched book list ─────────────────────────────────────
        return final_enriched_books



    def get_genre_via_embedding_model(self,
                                      raw_genre: str,
                                      ) -> Tuple[str, str]:
        """
        Takes raw genre string as inputs.
        Use embedding model to generate embedding of raw genre. 
        Then compute similarity score between generated embedding and 
        pre-computed saved genre embeddings.
                
        Return:
                Top matched book genre and genre code.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        GENRE_CODES: dict[str, str] = {
            "FIC":  "Fiction (general)",
            "SCI":  "Science Fiction",
            "FAN":  "Fantasy",
            "MYS":  "Mystery / Thriller",
            "ROM":  "Romance",
            "HIS":  "Historical Fiction",
            "BIO":  "Biography / Memoir",
            "NF":   "Non-Fiction (general)",
            "SLF":  "Self-Help / Personal Development",
            "SCI2": "Popular Science",
            "HIS2": "History",
            "CRI":  "Crime / Mafia",
            "BUS":  "Business / Economics",
            "PHI":  "Philosophy",
            "POE":  "Poetry",
            "CHI":  "Children's / Young Adult",
            "GRA":  "Graphic Novel / Comics",
            "CLA":  "Classic Literature",
            "SPT":  "Sports",
            "TRV":  "Travel",
            "COO":  "Cooking / Food",
            "ART":  "Art / Design",
            "REL":  "Religion / Spirituality",
            "UNK":  "Unknown / Unclassifiable",
            }
        
        ## List of genres:
        genre_list = [(code, genre) for code, genre in GENRE_CODES.items()] 
        
        ## Generating embedding for raw genre:
        raw_genre_embeddings = self._emb_model.encode(
                        raw_genre,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
        
        # Reshape the array for smilarity check
        if raw_genre_embeddings.ndim == 1:
            raw_genre_embeddings = raw_genre_embeddings.reshape(1, -1)
            
        genre_embeddings = np.load(self._genre_emb_file_path)
        
        ## Calculating similarity score:
        similarities = cosine_similarity(raw_genre_embeddings, genre_embeddings)[0]
        
        ## Free up space:
        del genre_embeddings

        ## PAIR GENRES WITH SCORES
        results = list(zip(genre_list, similarities))

        ## Sort descending to get highest score:
        results.sort(key=lambda x: x[1], reverse=True)
        
        code, genre = results[0][0]
        return code, genre


    def to_csv(self, books: list[BookRecord]) -> str:
        """
        Serialise the enriched catalog to a UTF-8 CSV string.

        The CSV includes a header row and one data row per book.
        Empty/None fields are written as empty strings.

        The output is a plain string — the route handler writes it to a
        temp file and serves it as a download.

        Args:
            books: Enriched BookRecord objects (output of generate_catalog).

        Returns:
            CSV string with header row.
        """
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",   # Silently ignore fields not in CSV_COLUMNS
            lineterminator="\r\n",   # RFC 4180 line endings
        )
        writer.writeheader()

        for book in books:
            row = {
                "title":          book.title or "",
                "subtitle":       book.subtitle or "",
                "author":         book.author or "",
                "genre":          book.genre or "",
                "genre_code":     book.genre_code or "",
                "pagecount":      book.pagecount or "",
                "ratings":        book.ratings or "Nil",
                "publisher":      book.publisher or "",
                "year_published": book.year_published or "",
                "isbn":           book.isbn or "",
                "description":    book.description or "",
                "source":         book.source,
                "ocr_confidence": f"{book.ocr_confidence:.2f}",
                "book_id":        book.book_id,
            }
            writer.writerow(row)

        return buf.getvalue()

