# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_llm_parser.py
#
# Unit tests for extract_json, RecommendationResponseParser,
# and CatalogResponseParser.
#
# No LLM calls are made — all tests feed synthetic response strings directly
# to the parser functions. This exercises the messy string-cleanup logic
# (markdown fences, prose prefix/suffix) exhaustively without network I/O.
# ─────────────────────────────────────────────────────────────────────────────

import json
import uuid
import pytest

from app.core.exceptions import LLMError
from app.models.book import BookRecord, BookRecommendation
from app.services.llm_parser import (
    CatalogResponseParser,
    RecommendationResponseParser,
    extract_json,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

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


def _rec_json(book: BookRecord, rank: int = 1) -> dict:
    """Build a syntactically correct recommendation dict for a book."""
    return {
        "book_id": book.book_id,
        "title": book.title,
        "author": book.author,
        "rank": rank,
        "summary": f"A wonderful book called {book.title}.",
        "match_reason": "Matches your preferred genre.",
        "genre": "Science Fiction",
        "genre_code": "SCI",
        "year_published": 1965,
    }


def _catalog_json(book: BookRecord) -> dict:
    """Build a syntactically correct catalog enrichment dict for a book."""
    return {
        "book_id": book.book_id,
        "title": book.title,
        "author": book.author,
        "genre": "Science Fiction",
        "genre_code": "SCI",
        "summary": "An epic science fiction novel.",
        "year_published": 1965,
        "isbn": "978-0-441-17271-9",
    }


# ── extract_json ───────────────────────────────────────────────────────────────

class TestExtractJson:

    def test_parses_pure_json(self):
        raw = '{"key": "value"}'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_strips_json_markdown_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_strips_plain_markdown_fence(self):
        raw = '```\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_strips_prose_prefix(self):
        raw = 'Here is the result:\n{"key": "value"}'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_strips_prose_suffix(self):
        raw = '{"key": "value"}\n\nLet me know if you need anything else!'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_handles_nested_json(self):
        raw = '{"outer": {"inner": [1, 2, 3]}}'
        result = extract_json(raw)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_raises_llm_error_for_no_json(self):
        with pytest.raises(LLMError, match="no JSON object"):
            extract_json("This response has no JSON in it at all.")

    def test_raises_llm_error_for_malformed_json(self):
        with pytest.raises(LLMError, match="malformed JSON"):
            extract_json('{"key": "value" BROKEN}')

    def test_raises_llm_error_for_empty_string(self):
        with pytest.raises(LLMError):
            extract_json("")

    def test_handles_fence_with_extra_whitespace(self):
        raw = '```json   \n  {"key": "value"}  \n```'
        result = extract_json(raw)
        assert result == {"key": "value"}


# ── RecommendationResponseParser ──────────────────────────────────────────────

class TestRecommendationResponseParser:

    def setup_method(self):
        self.parser = RecommendationResponseParser()

    def _build_response(self, recs: list[dict]) -> str:
        return json.dumps({"recommendations": recs})

    def test_returns_list_of_book_recommendations(self):
        book = _make_book()
        raw = self._build_response([_rec_json(book, rank=1)])
        results = self.parser.parse(raw, [book])
        assert all(isinstance(r, BookRecommendation) for r in results)

    def test_correct_count_returned(self):
        books = [_make_book(title=f"Book {i}") for i in range(3)]
        recs = [_rec_json(b, rank=i+1) for i, b in enumerate(books)]
        raw = self._build_response(recs)
        results = self.parser.parse(raw, books)
        assert len(results) == 3

    def test_sorted_by_rank_ascending(self):
        books = [_make_book(title=f"Book {i}") for i in range(3)]
        # Send in reverse rank order
        recs = [_rec_json(b, rank=3-i) for i, b in enumerate(books)]
        raw = self._build_response(recs)
        results = self.parser.parse(raw, books)
        ranks = [r.rank for r in results]
        assert ranks == sorted(ranks)

    def test_book_id_preserved(self):
        book = _make_book()
        raw = self._build_response([_rec_json(book)])
        results = self.parser.parse(raw, [book])
        assert results[0].book_id == book.book_id

    def test_crop_path_recovered_from_original(self):
        """crop_image_path is not sent to the LLM; parser must recover it."""
        book = _make_book()
        raw = self._build_response([_rec_json(book)])
        results = self.parser.parse(raw, [book])
        assert results[0].crop_image_path == book.crop_image_path

    def test_llm_corrected_title_used(self):
        """If LLM returns a corrected title, it should replace the OCR title."""
        book = _make_book(title="Grbld Ttl")
        rec = _rec_json(book)
        rec["title"] = "Correct Title"  # LLM corrected it
        raw = self._build_response([rec])
        results = self.parser.parse(raw, [book])
        assert results[0].title == "Correct Title"

    def test_unknown_book_id_skipped_with_warning(self):
        """If LLM returns an unknown book_id, it must be skipped."""
        book = _make_book()
        rec = _rec_json(book)
        rec["book_id"] = "does-not-exist"  # Unknown ID
        raw = self._build_response([rec])
        results = self.parser.parse(raw, [book])
        assert len(results) == 0

    def test_empty_recommendations_list_returns_empty(self):
        raw = json.dumps({"recommendations": []})
        results = self.parser.parse(raw, [_make_book()])
        assert results == []

    def test_markdown_fenced_response_parsed(self):
        book = _make_book()
        inner = json.dumps({"recommendations": [_rec_json(book)]})
        raw = f"```json\n{inner}\n```"
        results = self.parser.parse(raw, [book])
        assert len(results) == 1

    def test_summary_preserved(self):
        book = _make_book()
        rec = _rec_json(book)
        rec["summary"] = "A truly unique science fiction masterpiece."
        raw = self._build_response([rec])
        results = self.parser.parse(raw, [book])
        assert results[0].summary == "A truly unique science fiction masterpiece."

    def test_match_reason_preserved(self):
        book = _make_book()
        rec = _rec_json(book)
        rec["match_reason"] = "Matches your love of epic world-building."
        raw = self._build_response([rec])
        results = self.parser.parse(raw, [book])
        assert results[0].match_reason == "Matches your love of epic world-building."

    def test_enrich_book_records_updates_genre(self):
        """enrich_book_records must update BookRecord.genre from the LLM response."""
        book = _make_book()
        rec = _rec_json(book)
        rec["genre"] = "Classic Science Fiction"
        rec["genre_code"] = "SCI"
        raw = self._build_response([rec])
        updated = self.parser.enrich_book_records(raw, [book])
        assert updated[0].genre == "Classic Science Fiction"

    def test_enrich_book_records_updates_year(self):
        book = _make_book()
        rec = _rec_json(book)
        rec["year_published"] = 1965
        raw = self._build_response([rec])
        updated = self.parser.enrich_book_records(raw, [book])
        assert updated[0].year_published == 1965

    def test_enrich_book_records_on_parse_error_returns_originals(self):
        """If JSON parsing fails, enrich must return books unchanged."""
        books = [_make_book()]
        updated = self.parser.enrich_book_records("not json at all !!!", books)
        assert updated == books


# ── CatalogResponseParser ─────────────────────────────────────────────────────

class TestCatalogResponseParser:

    def setup_method(self):
        self.parser = CatalogResponseParser()

    def _build_response(self, items: list[dict]) -> str:
        return json.dumps({"catalog": items})

    def test_returns_same_length_as_input(self):
        """Output must have the same number of books as input."""
        books = [_make_book(title=f"Book {i}") for i in range(4)]
        items = [_catalog_json(b) for b in books]
        raw = self._build_response(items)
        result = self.parser.parse(raw, books)
        assert len(result) == 4

    def test_genre_code_applied(self):
        book = _make_book()
        item = _catalog_json(book)
        item["genre_code"] = "FAN"
        raw = self._build_response([item])
        result = self.parser.parse(raw, [book])
        assert result[0].genre_code == "FAN"

    def test_genre_applied(self):
        book = _make_book()
        item = _catalog_json(book)
        item["genre"] = "Fantasy"
        raw = self._build_response([item])
        result = self.parser.parse(raw, [book])
        assert result[0].genre == "Fantasy"

    def test_isbn_applied(self):
        book = _make_book()
        item = _catalog_json(book)
        item["isbn"] = "978-0-441-17271-9"
        raw = self._build_response([item])
        result = self.parser.parse(raw, [book])
        assert result[0].isbn == "978-0-441-17271-9"

    def test_year_published_applied(self):
        book = _make_book()
        item = _catalog_json(book)
        item["year_published"] = 1984
        raw = self._build_response([item])
        result = self.parser.parse(raw, [book])
        assert result[0].year_published == 1984

    def test_llm_corrected_title_used(self):
        """LLM may fix OCR-garbled titles; corrected title must replace original."""
        book = _make_book(title="Garbledd Ttitle")
        item = _catalog_json(book)
        item["title"] = "Garbled Title Fixed"
        raw = self._build_response([item])
        result = self.parser.parse(raw, [book])
        assert result[0].title == "Garbled Title Fixed"

    def test_missing_book_kept_with_original_data(self):
        """Books the LLM skips must still appear in output with original data."""
        book1 = _make_book(title="Book Present")
        book2 = _make_book(title="Book Missing")
        # Only include book1 in the LLM response
        raw = self._build_response([_catalog_json(book1)])
        result = self.parser.parse(raw, [book1, book2])
        # Both books must be in result
        assert len(result) == 2
        titles = {r.title for r in result}
        assert "Book Missing" in titles

    def test_order_preserved(self):
        """Output must be in the same order as the input books list."""
        books = [_make_book(title=f"Book {i}") for i in range(3)]
        items = [_catalog_json(b) for b in books]
        raw = self._build_response(items)
        result = self.parser.parse(raw, books)
        assert [r.book_id for r in result] == [b.book_id for b in books]

    def test_markdown_fenced_response_parsed(self):
        book = _make_book()
        inner = json.dumps({"catalog": [_catalog_json(book)]})
        raw = f"```json\n{inner}\n```"
        result = self.parser.parse(raw, [book])
        assert len(result) == 1

    def test_raises_on_completely_invalid_json(self):
        """A response that is not JSON at all must raise LLMError."""
        with pytest.raises(LLMError):
            self.parser.parse("I could not process your request.", [_make_book()])

    def test_empty_catalog_returns_original_books(self):
        """An empty catalog list must return the original books unchanged."""
        books = [_make_book()]
        raw = json.dumps({"catalog": []})
        result = self.parser.parse(raw, books)
        assert result == books
