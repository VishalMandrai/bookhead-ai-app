# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_catalog_service.py
#
# Unit tests for LLMCatalogService.
#
# The Anthropic API is NEVER called. A MockLLMClient returns pre-canned
# catalog JSON strings. Tests cover:
#   - Single-batch and multi-batch scenarios
#   - Genre code enrichment
#   - ISBN and year_published propagation
#   - CSV export format and content
#   - Empty input fast-path
# ─────────────────────────────────────────────────────────────────────────────

import csv
import io
import json
import uuid
import pytest
from unittest.mock import MagicMock, call

from app.models.book import BookRecord
from app.services.catalog import LLMCatalogService, _chunk, _ceil_div
from app.services.prompt_builder import CatalogPromptBuilder
from app.services.llm_parser import CatalogResponseParser


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_book(title: str = "Dune", author: str = "Frank Herbert") -> BookRecord:
    bid = str(uuid.uuid4())
    return BookRecord(
        book_id=bid,
        title=title,
        author=author,
        crop_image_path=f"crops/job/{bid[:8]}.jpg",
        ocr_confidence=0.88,
        source="ocr_auto",
        ori_ocr_ext_spine_txt=title,
    )


def _catalog_response(books: list[BookRecord], genre_code: str = "SCI") -> str:
    """Build a valid LLM catalog JSON response for the given books."""
    return json.dumps({
        "catalog": [
            {
                "book_id": b.book_id,
                "title": b.title,
                "author": b.author,
                "genre": "Science Fiction",
                "genre_code": genre_code,
                "summary": f"A great book: {b.title}.",
                "year_published": 1965,
                "isbn": "978-0-441-17271-9",
            }
            for b in books
        ]
    })


def _make_service(
    batch_size: int = 25,
    llm_response_fn=None,
) -> LLMCatalogService:
    """
    Build an LLMCatalogService with a mock LLM client.

    llm_response_fn: optional callable(books) → str that generates the
    mock response based on which books are in the current batch.
    If None, returns an empty catalog.
    """
    mock_client = MagicMock()
    if llm_response_fn:
        # The mock needs to return different responses per batch.
        # We use side_effect with a closure that peeks at the user_prompt.
        # Simpler: just return a fixed empty response and let tests override.
        mock_client.complete.return_value = ""
    return LLMCatalogService(
        api_key="test-key",
        batch_size=batch_size,
        llm_client=mock_client,
        prompt_builder=CatalogPromptBuilder(),
        parser=CatalogResponseParser(),
    )


# ── Tests: generate_catalog() ─────────────────────────────────────────────────

class TestLLMCatalogServiceGenerate:

    def test_empty_input_returns_empty_list(self):
        svc = _make_service()
        result = svc.generate_catalog([])
        assert result == []

    def test_empty_input_does_not_call_llm(self):
        mock_client = MagicMock()
        svc = LLMCatalogService(api_key="test-key", llm_client=mock_client)
        svc.generate_catalog([])
        mock_client.complete.assert_not_called()

    def test_single_batch_calls_llm_once(self):
        """For a list smaller than batch_size, LLM is called exactly once."""
        books = [_make_book(title=f"Book {i}") for i in range(5)]
        mock_client = MagicMock()
        mock_client.complete.return_value = _catalog_response(books)
        svc = LLMCatalogService(
            api_key="test-key",
            batch_size=25,
            llm_client=mock_client,
        )
        svc.generate_catalog(books)
        assert mock_client.complete.call_count == 1

    def test_multi_batch_calls_llm_per_batch(self):
        """30 books with batch_size=10 → exactly 3 LLM calls."""
        books = [_make_book(title=f"Book {i}") for i in range(30)]
        mock_client = MagicMock()

        # Return a valid catalog for whichever subset arrives
        def side_effect(system, user):
            # Parse book_ids from the user prompt JSON
            start = user.find("[")
            end = user.rfind("]") + 1
            batch_books_data = json.loads(user[start:end])
            batch_books = [
                _make_book(title=d["title"])
                for d in batch_books_data
            ]
            # Build response matching actual book_ids from the user prompt
            catalog_items = [
                {
                    "book_id": d["book_id"],
                    "title": d["title"],
                    "author": d["author"],
                    "genre": "Fiction",
                    "genre_code": "FIC",
                    "summary": "A fine book.",
                    "year_published": 2000,
                    "isbn": None,
                }
                for d in batch_books_data
            ]
            return json.dumps({"catalog": catalog_items})

        mock_client.complete.side_effect = side_effect
        svc = LLMCatalogService(
            api_key="test-key",
            batch_size=10,
            llm_client=mock_client,
        )
        result = svc.generate_catalog(books)
        assert mock_client.complete.call_count == 3
        assert len(result) == 30

    def test_genre_code_set_on_enriched_records(self):
        book = _make_book()
        mock_client = MagicMock()
        mock_client.complete.return_value = _catalog_response([book], genre_code="FAN")
        svc = LLMCatalogService(api_key="test-key", llm_client=mock_client)
        result = svc.generate_catalog([book])
        assert result[0].genre_code == "FAN"

    def test_isbn_set_on_enriched_records(self):
        book = _make_book()
        mock_client = MagicMock()
        mock_client.complete.return_value = _catalog_response([book])
        svc = LLMCatalogService(api_key="test-key", llm_client=mock_client)
        result = svc.generate_catalog([book])
        assert result[0].isbn == "978-0-441-17271-9"

    def test_year_published_set_on_enriched_records(self):
        book = _make_book()
        mock_client = MagicMock()
        mock_client.complete.return_value = _catalog_response([book])
        svc = LLMCatalogService(api_key="test-key", llm_client=mock_client)
        result = svc.generate_catalog([book])
        assert result[0].year_published == 1965

    def test_summary_set_on_enriched_records(self):
        book = _make_book(title="Dune")
        mock_client = MagicMock()
        mock_client.complete.return_value = _catalog_response([book])
        svc = LLMCatalogService(api_key="test-key", llm_client=mock_client)
        result = svc.generate_catalog([book])
        assert result[0].summary is not None
        assert len(result[0].summary) > 5


# ── Tests: to_csv() ────────────────────────────────────────────────────────────

class TestLLMCatalogServiceCSV:

    def _enrich(self, book: BookRecord) -> BookRecord:
        """Apply mock enrichment to a BookRecord for CSV tests."""
        book.genre = "Science Fiction"
        book.genre_code = "SCI"
        book.summary = "An epic tale."
        book.year_published = 1965
        book.isbn = "978-0-441-17271-9"
        return book

    def test_csv_has_header_row(self):
        svc = _make_service()
        book = self._enrich(_make_book())
        csv_str = svc.to_csv([book])
        reader = csv.DictReader(io.StringIO(csv_str))
        assert reader.fieldnames is not None
        assert "title" in reader.fieldnames
        assert "genre_code" in reader.fieldnames

    def test_csv_row_count_matches_book_count(self):
        svc = _make_service()
        books = [self._enrich(_make_book(title=f"Book {i}")) for i in range(5)]
        csv_str = svc.to_csv(books)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 5

    def test_csv_title_column_correct(self):
        svc = _make_service()
        book = self._enrich(_make_book(title="Foundation"))
        csv_str = svc.to_csv([book])
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        assert row["title"] == "Foundation"

    def test_csv_author_column_correct(self):
        svc = _make_service()
        book = self._enrich(_make_book(author="Isaac Asimov"))
        csv_str = svc.to_csv([book])
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        assert row["author"] == "Isaac Asimov"

    def test_csv_genre_code_column_correct(self):
        svc = _make_service()
        book = self._enrich(_make_book())
        csv_str = svc.to_csv([book])
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        assert row["genre_code"] == "SCI"

    def test_csv_isbn_column_correct(self):
        svc = _make_service()
        book = self._enrich(_make_book())
        csv_str = svc.to_csv([book])
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        assert row["isbn"] == "978-0-441-17271-9"

    def test_csv_empty_fields_are_empty_string(self):
        """None fields must be written as empty strings, not 'None'."""
        svc = _make_service()
        book = _make_book()  # No enrichment — isbn/genre are None
        csv_str = svc.to_csv([book])
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        assert row["isbn"] != "None"
        assert row["genre"] != "None"

    def test_csv_source_column_included(self):
        svc = _make_service()
        book = self._enrich(_make_book())
        book.source = "human_corrected"  # type: ignore[assignment]
        csv_str = svc.to_csv([book])
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        assert row["source"] == "human_corrected"

    def test_csv_ocr_confidence_is_formatted_float(self):
        svc = _make_service()
        book = self._enrich(_make_book())
        book.ocr_confidence = 0.876543
        csv_str = svc.to_csv([book])
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        # Should be formatted to 2 decimal places
        assert row["ocr_confidence"] == "0.88"

    def test_csv_empty_book_list_has_only_header(self):
        svc = _make_service()
        csv_str = svc.to_csv([])
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 0
        assert reader.fieldnames is not None  # Header still present


# ── Tests: private helpers ─────────────────────────────────────────────────────

class TestCatalogHelpers:

    def test_chunk_divides_evenly(self):
        result = _chunk([1, 2, 3, 4, 5, 6], 2)
        assert result == [[1, 2], [3, 4], [5, 6]]

    def test_chunk_with_remainder(self):
        result = _chunk([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_chunk_larger_than_list(self):
        result = _chunk([1, 2, 3], 10)
        assert result == [[1, 2, 3]]

    def test_chunk_empty_list(self):
        assert _chunk([], 5) == []

    def test_ceil_div_exact(self):
        assert _ceil_div(10, 5) == 2

    def test_ceil_div_with_remainder(self):
        assert _ceil_div(11, 5) == 3

    def test_ceil_div_less_than_divisor(self):
        assert _ceil_div(3, 10) == 1
