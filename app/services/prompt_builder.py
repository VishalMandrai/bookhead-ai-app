# ─────────────────────────────────────────────────────────────────────────────
# app/services/prompt_builder.py
#
# Builds structured prompts for the two LLM tasks:
#   1. ReaderPromptBuilder  – recommendation + summary for a reader
#   2. CatalogPromptBuilder – genre-coding + other enrichment for a librarian
#
# WHY A SEPARATE PROMPT MODULE?
#   Prompts are business logic, not infrastructure. Keeping them here means:
#     - Prompts are version-controlled and reviewable like any other code
#     - LLMRecommendationService and LLMCatalogService stay thin
#     - Prompt changes don't require touching service or pipeline code
#     - Unit tests can inspect prompt text without making API calls
#
# PROMPT DESIGN PRINCIPLES USED:                                                   \\\NOTE: UPDATE THIS ---
#   1. Explicit output schema  – the LLM is shown the exact JSON it must return,
#      reducing hallucinated fields and missing keys.
#   2. Constraints in the system prompt – hard rules (max N results, JSON only)
#      live in the system prompt where they are hardest to ignore.
#   3. Per-book context – each book is presented with its OCR text and source
#      (ocr_auto vs human_corrected) so the LLM can weight confidence.
#   4. Web search instruction – the reader prompt explicitly asks the LLM to
#      use web search to fill gaps in its knowledge about specific titles.
#
# SOLID (Single Responsibility + Open/Closed):
#   Each builder owns exactly one prompt type for one specific task. 
#   A new task (e.g. "reading age suitability scoring") adds a new builder
#   class without touching these.
# ─────────────────────────────────────────────────────────────────────────────


from __future__ import annotations

import json

from app.models.book import BookRecord
from app.models.request import ReaderPreferences


# ── Genre code reference table (embedded in catalog prompt) ──────────────────

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

# Compact genre table for embedding in prompts
_GENRE_TABLE = "\n".join(f"  {code}: {label}" for code, label in GENRE_CODES.items())

# ── Gemma 2B LLM - Prompt builder for "Text Summarization" ────────────────────────────────────────────

# NOTE: Does Gemma 2B, support separate system_prompt and user_prompt?
# ANS: NO, not natively in the same way as GPT-4 or OpenAI APIs. 
# Gemma 2B (specifically the instruction-tuned gemma-2-2b-it variants) is designed to work with only two roles: user and model. It does not support a dedicated "system" role or a separate "system prompt" parameter in its native prompt structure.
# However, we can achieve the same behavior by concatenating the system instructions within the initial user prompt.
# How to Implement System Prompts (The "Gemma Way") ?
# SOLUTION: To simulate a system prompt, we must prepend the instructions directly to the user turn before sending it to the model.

# NOTE: The model relies on SPECIFIC formatting tokens. 
# Here is the correct structure for "gemma-2-2b-it":
#                   <start_of_turn>user
#                   {System Instructions/Role}

#                   {User Question}
#                   <end_of_turn>
#                   <start_of_turn>model

# Example Scenario: If we want it to act as a "helpful assistant" (System) and answer "Who are you?" (User), the format it like: 
#                               <start_of_turn>user
#                               You are a helpful assistant. Only answer in concise sentences.
#                               Who are you?
#                               <end_of_turn>
#                               <start_of_turn>model


class SummaryPromptGemma:
    """
    Builds the complete prompt for Text Summarization task.
    For better readability we write a system prompt, then build the user prompt
    and then format both togther in Gemma way for Text Summarization task.

    System Prompt establishes Gemma's role and strict output constraints.
    User Prompt provides the book description for summarization.
    """

    SYSTEM_PROMPT = """You are a book analysis API.
Write a 6-7 sentences summary of book description.
- Focus on:
      - core premise
      - genre/tone
      - what makes the book interesting
      - why readers may enjoy it

Rules:
- No markdown
"""

    def build(
        self,
        description: str,
    ) -> tuple[str, str]:
        """
        Build the prompt text summarization call.

        Args:
            description: book description obtained from Google Books API

        Returns: Final prompt to pass to the Gemma LLM.
        """
        user_prompt = f"""
        INPUT - Book Description:
        \"\"\"
        {description}
        \"\"\"
        """
        
        formatted_prompt = f"""
                    <start_of_turn>user
                    {self.SYSTEM_PROMPT + user_prompt}<end_of_turn>
                    <start_of_turn>model
                    """

        return formatted_prompt

    

# ─────────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────────


# ── Reader prompt builder - Heavy LLM with Web Search enabled ────────────────────────────────────────

class ReaderPromptBuilder:
    """
    Builds the system + user prompt pair for the reader recommendation task.

    System Prompt establishes Claude's role and strict output constraints.
    User Prompt provides the book list and reader preferences as structured
    context, then asks for a ranked JSON response.
    """

    SYSTEM_PROMPT = """You are an AI powered Book Shelf Engine, an expert literary advisor with deep
knowledge of books across all genres and eras.

Your task is to analyse a list of books detected from a bookshelf photo and
provide personalised reading recommendations for the user.

RULES:
- Use web search to look up any book you're not certain about (publication year,
  genre, plot, critical reception).
- Return ONLY a valid JSON object — no markdown fences, no prose outside the JSON.
- Rank books from best to worst fit for the user's stated preferences.
- Include only books from the provided list; do not invent titles.
- If a book title or author looks garbled (low OCR confidence), note that in
  match_reason and still attempt a recommendation if you can identify the book.
- Respect exclude_genres strictly.

OUTPUT SCHEMA (return exactly this structure):
{
  "recommendations": [
    {
      "book_id": "<exact book_id from input>",
      "title": "<confirmed/corrected title>",
      "author": "<confirmed/corrected author>",
      "rank": <integer starting at 1>,
      "summary": "<2-3 sentence plot/theme summary>",
      "match_reason": "<1 sentence: why this suits the user's preferences>",
      "genre": "<primary genre>",
      "genre_code": "<code from genre list>",
      "year_published": <integer or null>
    }
  ]
}"""

    def build(
        self,
        books: list[BookRecord],
        preferences: ReaderPreferences,
    ) -> tuple[str, str]:
        """
        Build the (system_prompt, user_prompt) pair for the recommendation call.

        Args:
            books:       All detected books from the shelf image.
            preferences: Reader's genre/mood/length preferences.

        Returns:
            (system_prompt, user_prompt) tuple ready to pass to the Anthropic API.
        """
        # Format the book list as a JSON array so the LLM gets structured input
        book_list_json = json.dumps(
            [
                {
                    "book_id": b.book_id,
                    "title": b.title,
                    "author": b.author,
                    "ocr_confidence": b.ocr_confidence,
                    "source": b.source,   # "ocr_auto" or "human_corrected"
                }
                for b in books
            ],
            indent=2,
        )

        # Format preferences clearly
        prefs_lines = [
            f"- Preferred genres: {preferences.preferred_genres or 'any'}",
            f"- Mood / vibe: {preferences.mood or 'not specified'}",
            f"- Preferred book length: {preferences.preferred_length}",
            f"- Exclude genres: {preferences.exclude_genres or 'none'}",
            f"- Max recommendations requested: {preferences.max_recommendations}",
        ]

        user_prompt = f"""Here are the books detected from the reader's bookshelf photo:

{book_list_json}

The reader's preferences are:
{chr(10).join(prefs_lines)}

Please:
1. Use web search to research any books you are not fully familiar with.
2. Rank the books best → worst fit for these preferences.
3. Return the top {preferences.max_recommendations} as a JSON object matching
   the output schema in your instructions.

Remember: JSON only, no surrounding text."""

        return self.SYSTEM_PROMPT, user_prompt


# ── Catalog prompt builder - for Heavy LLM with Web Search enabled ────────────────────────────────

class CatalogPromptBuilder:
    """
    Builds the system + user prompt pair for the librarian catalog task.

    The catalog prompt asks the LLM to enrich each book with:
      - Confirmed/corrected title and author (it may fix OCR errors via web search)
      - Genre and genre code from the standard GENRE_CODES table
      - A brief summary
      - Publication year and ISBN where known
    """

    SYSTEM_PROMPT = f"""You are BookLens AI, a library cataloguing assistant.

Your task is to enrich a list of books with accurate bibliographic metadata
and assign genre codes from the standard genre code table below.

GENRE CODE TABLE:
{_GENRE_TABLE}

RULES:
- Use web search to verify titles, authors, publication years, and ISBNs.
- Assign the MOST SPECIFIC genre code that applies; use "UNK" only as a last resort.
- If OCR text looks garbled, attempt to identify the book via web search using
  partial title/author clues.
- Return ONLY a valid JSON object — no markdown fences, no prose outside JSON.
- Process every book in the input list; do not skip or merge entries.

OUTPUT SCHEMA (return exactly this structure):
{{
  "catalog": [
    {{
      "book_id": "<exact book_id from input>",
      "title": "<verified/corrected title>",
      "author": "<verified/corrected author>",
      "genre": "<full genre name>",
      "genre_code": "<code from table above>",
      "summary": "<1-2 sentence description>",
      "year_published": <integer or null>,
      "isbn": "<ISBN-13 string or null>"
    }}
  ]
}}"""

    def build(self, books: list[BookRecord]) -> tuple[str, str]:
        """
        Build the (system_prompt, user_prompt) pair for the catalog call.

        For large catalogs (>20 books) the user prompt instructs the LLM to
        process them in a single batch rather than explaining each one, keeping
        token usage efficient.

        Args:
            books: All BookRecord objects to catalog.

        Returns:
            (system_prompt, user_prompt) tuple.
        """
        book_list_json = json.dumps(
            [
                {
                    "book_id": b.book_id,
                    "title": b.title,
                    "author": b.author,
                    "ocr_confidence": b.ocr_confidence,
                    "source": b.source,
                }
                for b in books
            ],
            indent=2,
        )

        user_prompt = f"""Please catalogue the following {len(books)} book(s) detected from library shelf photos.

{book_list_json}

Instructions:
1. Use web search to verify and correct any garbled OCR text.
2. Assign a genre code from the standard table.
3. Provide a brief summary and publication year for each book.
4. Return results as a JSON object matching the output schema exactly.

JSON only — no surrounding text."""

        return self.SYSTEM_PROMPT, user_prompt
