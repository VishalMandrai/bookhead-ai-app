# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_recommendation_service.py
#
# Unit tests for LLMRecommendationService.
#
# The Anthropic API is NEVER called. A MockLLMClient is injected that returns
# pre-canned JSON strings, letting us test the full orchestration logic
# (prompt building → LLM call → parsing → enrichment) without network I/O.
# ─────────────────────────────────────────────────────────────────────────────

import json
import uuid
import pytest
from unittest.mock import MagicMock

from app.models.book import BookRecord, BookRecommendation
from app.models.request import ReaderPreferences
from app.services.recommendation import LLMRecommendationService
from app.services.prompt_builder import ReaderPromptBuilder
from app.services.llm_parser import RecommendationResponseParser


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_book(title: str = "Dune", author: str = "Frank Herbert") -> BookRecord:
    bid = str(uuid.uuid4())
    return BookRecord(
        book_id=bid,
        title=title,
        author=author,
        crop_image_path=f"crops/job/{bid[:8]}.jpg",
        ocr_confidence=0.90,
        source="ocr_auto",
        ori_ocr_ext_spine_txt=title,
    )


def _prefs(**kwargs) -> ReaderPreferences:
    return ReaderPreferences(**kwargs)


def _mock_llm_response(books: list[BookRecord]) -> str:
    """Build a valid LLM recommendation JSON response for the given books."""
    recs = [
        {
            "book_id": b.book_id,
            "title": b.title,
            "author": b.author,
            "rank": i + 1,
            "summary": f"Summary of {b.title}.",
            "match_reason": f"{b.title} matches your preferences.",
            "genre": "Science Fiction",
            "genre_code": "SCI",
            "year_published": 1965,
        }
        for i, b in enumerate(books)
    ]
    return json.dumps({"recommendations": recs})


def _make_service(mock_response: str | None = None) -> LLMRecommendationService:
    """Build a service with a mock LLM client pre-injected."""
    mock_client = MagicMock()
    mock_client.complete.return_value = mock_response or ""
    return LLMRecommendationService(
        api_key="test-key",
        llm_client=mock_client,
        prompt_builder=ReaderPromptBuilder(),
        parser=RecommendationResponseParser(),
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestLLMRecommendationService:

    def test_returns_list_of_book_recommendations(self):
        book = _make_book()
        svc = _make_service(_mock_llm_response([book]))
        results = svc.recommend([book], _prefs())
        assert all(isinstance(r, BookRecommendation) for r in results)

    def test_correct_count_of_recommendations(self):
        books = [_make_book(title=f"Book {i}") for i in range(5)]
        svc = _make_service(_mock_llm_response(books))
        results = svc.recommend(books, _prefs(max_recommendations=5))
        assert len(results) == 5

    def test_empty_book_list_returns_empty_without_llm_call(self):
        """No LLM call should be made when the book list is empty."""
        mock_client = MagicMock()
        svc = LLMRecommendationService(
            api_key="test-key",
            llm_client=mock_client,
        )
        results = svc.recommend([], _prefs())
        assert results == []
        mock_client.complete.assert_not_called()

    def test_llm_client_called_once(self):
        """The LLM client must be called exactly once per recommend() call."""
        book = _make_book()
        mock_client = MagicMock()
        mock_client.complete.return_value = _mock_llm_response([book])
        svc = LLMRecommendationService(
            api_key="test-key",
            llm_client=mock_client,
        )
        svc.recommend([book], _prefs())
        mock_client.complete.assert_called_once()

    def test_results_sorted_by_rank(self):
        books = [_make_book(title=f"Book {i}") for i in range(3)]
        svc = _make_service(_mock_llm_response(books))
        results = svc.recommend(books, _prefs())
        ranks = [r.rank for r in results]
        assert ranks == sorted(ranks)

    def test_book_ids_preserved(self):
        books = [_make_book(title=f"Book {i}") for i in range(3)]
        svc = _make_service(_mock_llm_response(books))
        results = svc.recommend(books, _prefs())
        result_ids = {r.book_id for r in results}
        original_ids = {b.book_id for b in books}
        assert result_ids == original_ids

    def test_crop_paths_recovered(self):
        """crop_image_path must be recovered from original books, not the LLM."""
        book = _make_book()
        svc = _make_service(_mock_llm_response([book]))
        results = svc.recommend([book], _prefs())
        assert results[0].crop_image_path == book.crop_image_path

    def test_book_records_enriched_with_genre(self):
        """After recommend(), the original BookRecord objects must have genre set."""
        book = _make_book()
        svc = _make_service(_mock_llm_response([book]))
        svc.recommend([book], _prefs())
        # The original book object should be mutated in-place with enrichment
        assert book.genre == "Science Fiction"

    def test_book_records_enriched_with_year(self):
        book = _make_book()
        svc = _make_service(_mock_llm_response([book]))
        svc.recommend([book], _prefs())
        assert book.year_published == 1965

    def test_prompts_passed_to_llm_client(self):
        """The LLM client must receive non-empty system and user prompts."""
        book = _make_book()
        mock_client = MagicMock()
        mock_client.complete.return_value = _mock_llm_response([book])
        svc = LLMRecommendationService(
            api_key="test-key",
            llm_client=mock_client,
        )
        svc.recommend([book], _prefs())
        call_args = mock_client.complete.call_args
        system_prompt = call_args[0][0]
        user_prompt   = call_args[0][1]
        assert len(system_prompt) > 50
        assert len(user_prompt) > 50

    def test_preferences_passed_into_prompt(self):
        """Genre preferences must reach the user prompt."""
        book = _make_book()
        mock_client = MagicMock()
        mock_client.complete.return_value = _mock_llm_response([book])
        svc = LLMRecommendationService(
            api_key="test-key",
            llm_client=mock_client,
        )
        svc.recommend([book], _prefs(preferred_genres=["horror"]))
        user_prompt = mock_client.complete.call_args[0][1]
        assert "horror" in user_prompt
