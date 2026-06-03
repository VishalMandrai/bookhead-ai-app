# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_prompt_builder.py
#
# Unit tests for ReaderPromptBuilder and CatalogPromptBuilder.
#
# No LLM calls are made — we only inspect the constructed prompt strings.
# Tests verify that all critical information (book IDs, titles, preferences,
# genre codes, output schema) is present in the prompt text.
# ─────────────────────────────────────────────────────────────────────────────

import json
import uuid
import pytest

from app.models.book import BookRecord
from app.models.request import ReaderPreferences
from app.services.prompt_builder import (
    CatalogPromptBuilder,
    ReaderPromptBuilder,
    GENRE_CODES,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_book(title: str = "Dune", author: str = "Frank Herbert") -> BookRecord:
    return BookRecord(
        book_id=str(uuid.uuid4()),
        title=title,
        author=author,
        crop_image_path="crops/job/book_0.jpg",
        ocr_confidence=0.88,
        source="ocr_auto",
        ori_ocr_ext_spine_txt=title,
    )


def _prefs(**kwargs) -> ReaderPreferences:
    return ReaderPreferences(**kwargs)


# ── ReaderPromptBuilder ────────────────────────────────────────────────────────

class TestReaderPromptBuilder:

    def setup_method(self):
        self.builder = ReaderPromptBuilder()

    def test_returns_two_strings(self):
        system, user = self.builder.build([_make_book()], _prefs())
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_prompt_not_empty(self):
        system, _ = self.builder.build([_make_book()], _prefs())
        assert len(system) > 100

    def test_system_prompt_contains_json_only_rule(self):
        """System prompt must instruct the LLM to return only JSON."""
        system, _ = self.builder.build([_make_book()], _prefs())
        assert "JSON" in system

    def test_system_prompt_contains_output_schema(self):
        """System prompt must contain the expected output schema keys."""
        system, _ = self.builder.build([_make_book()], _prefs())
        assert "recommendations" in system
        assert "book_id" in system
        assert "summary" in system
        assert "match_reason" in system

    def test_system_prompt_contains_web_search_instruction(self):
        """System prompt must instruct the model to use web search."""
        system, _ = self.builder.build([_make_book()], _prefs())
        assert "web search" in system.lower()

    def test_user_prompt_contains_all_book_ids(self):
        """All book IDs from the input list must appear in the user prompt."""
        books = [_make_book(title=f"Book {i}") for i in range(3)]
        _, user = self.builder.build(books, _prefs())
        for book in books:
            assert book.book_id in user

    def test_user_prompt_contains_titles(self):
        """Book titles must appear in the user prompt."""
        book = _make_book(title="The Great Gatsby")
        _, user = self.builder.build([book], _prefs())
        assert "The Great Gatsby" in user

    def test_user_prompt_contains_preferred_genres(self):
        """Preferred genres must be visible in the user prompt."""
        _, user = self.builder.build(
            [_make_book()],
            _prefs(preferred_genres=["science fiction", "mystery"]),
        )
        assert "science fiction" in user

    def test_user_prompt_contains_mood(self):
        """Mood preference must appear in the user prompt."""
        _, user = self.builder.build(
            [_make_book()],
            _prefs(mood="something dark and gritty"),
        )
        assert "dark and gritty" in user

    def test_user_prompt_contains_exclude_genres(self):
        """Excluded genres must appear in the user prompt."""
        _, user = self.builder.build(
            [_make_book()],
            _prefs(exclude_genres=["romance"]),
        )
        assert "romance" in user

    def test_user_prompt_contains_max_recommendations(self):
        """The max_recommendations count must appear in the user prompt."""
        _, user = self.builder.build(
            [_make_book()],
            _prefs(max_recommendations=7),
        )
        assert "7" in user

    def test_user_prompt_book_list_is_valid_json(self):
        """The book list embedded in the user prompt must be parseable JSON."""
        books = [_make_book(title="Dune"), _make_book(title="Foundation")]
        _, user = self.builder.build(books, _prefs())
        # Extract the JSON array from the prompt (starts at first '[')
        start = user.find("[")
        end = user.find("]", start) + 1
        book_list = json.loads(user[start:end])
        assert len(book_list) == 2

    def test_user_prompt_includes_ocr_confidence(self):
        """OCR confidence must be passed so the LLM knows which to scrutinise."""
        book = _make_book()
        _, user = self.builder.build([book], _prefs())
        assert "ocr_confidence" in user

    def test_user_prompt_includes_source_field(self):
        """Source (ocr_auto / human_corrected) must be visible to the LLM."""
        _, user = self.builder.build([_make_book()], _prefs())
        assert "source" in user

    def test_empty_book_list_does_not_crash(self):
        """Building a prompt with zero books must not raise."""
        system, user = self.builder.build([], _prefs())
        assert isinstance(system, str)
        assert isinstance(user, str)


# ── CatalogPromptBuilder ──────────────────────────────────────────────────────

class TestCatalogPromptBuilder:

    def setup_method(self):
        self.builder = CatalogPromptBuilder()

    def test_returns_two_strings(self):
        system, user = self.builder.build([_make_book()])
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_prompt_contains_json_only_rule(self):
        system, _ = self.builder.build([_make_book()])
        assert "JSON" in system

    def test_system_prompt_contains_catalog_schema(self):
        """Catalog schema keys must be present in the system prompt."""
        system, _ = self.builder.build([_make_book()])
        assert "catalog" in system
        assert "genre_code" in system
        assert "isbn" in system
        assert "year_published" in system

    def test_system_prompt_contains_genre_codes(self):
        """All genre codes from GENRE_CODES must be embedded in the system prompt."""
        system, _ = self.builder.build([_make_book()])
        for code in GENRE_CODES:
            assert code in system, f"Genre code '{code}' missing from system prompt"

    def test_system_prompt_contains_web_search_rule(self):
        system, _ = self.builder.build([_make_book()])
        assert "web search" in system.lower()

    def test_system_prompt_contains_unk_code(self):
        """UNK (Unknown) must be listed as a fallback genre code."""
        system, _ = self.builder.build([_make_book()])
        assert "UNK" in system

    def test_user_prompt_contains_book_count(self):
        """The user prompt must state how many books are being catalogued."""
        books = [_make_book(title=f"Book {i}") for i in range(5)]
        _, user = self.builder.build(books)
        assert "5" in user

    def test_user_prompt_contains_all_book_ids(self):
        books = [_make_book(title=f"Book {i}") for i in range(3)]
        _, user = self.builder.build(books)
        for book in books:
            assert book.book_id in user

    def test_user_prompt_contains_titles_and_authors(self):
        book = _make_book(title="Foundation", author="Isaac Asimov")
        _, user = self.builder.build([book])
        assert "Foundation" in user
        assert "Isaac Asimov" in user

    def test_user_prompt_book_list_is_valid_json(self):
        books = [_make_book(title="Dune"), _make_book(title="Neuromancer")]
        _, user = self.builder.build(books)
        start = user.find("[")
        end = user.find("]", start) + 1
        book_list = json.loads(user[start:end])
        assert len(book_list) == 2

    def test_empty_book_list_does_not_crash(self):
        system, user = self.builder.build([])
        assert isinstance(system, str)
        assert isinstance(user, str)
