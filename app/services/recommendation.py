# ─────────────────────────────────────────────────────────────────────────────
# app/services/recommendation.py
#
# LLMRecommendationService – concrete implementation of BaseRecommendationService.
#
# RESPONSIBILITY:
#   Given a list of BookRecord objects and a reader's preferences, use the
#   Anthropic LLM (with web search) to produce a ranked, summarised list of
#   reading recommendations.
#
# PIPELINE inside recommend():
#   1. Build structured prompt via ReaderPromptBuilder
#   2. Call AnthropicLLMClient (web search enabled)
#   3. Parse response via RecommendationResponseParser
#   4. Enrich original BookRecord objects with LLM metadata
#   5. Return ranked BookRecommendation list
#
# SOLID notes:
#   Single Responsibility  – this service only orchestrates the reader LLM flow.
#   Dependency Inversion   – depends on AnthropicLLMClient and the parser
#                            via constructor injection (swappable in tests).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Tuple
from app.core.logging import get_logger
from app.models.book import OCRValidatedResult, BookRecord, BookRecommendation
from app.models.request import ReaderPreferences
from app.services.base import BaseRecommendationService
from app.services.llm_client import AnthropicLLMClient, GemmaLLMClient
from app.services.llm_parser import RecommendationResponseParser, GemmaSummaryParser_InfoEnricher
from app.services.prompt_builder import ReaderPromptBuilder, SummaryPromptGemma

import numpy as np

logger = get_logger(__name__)


class LLMRecommendationService(BaseRecommendationService):
    """
    Produces personalised reading recommendations via API call to heavy-weight LLM.

    Web search is enabled so the model can look up books it is not certain
    about, especially for older or less well-known titles that may not be
    in its training data.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048,
        llm_client: AnthropicLLMClient | None = None,
        prompt_builder: ReaderPromptBuilder | None = None,
        parser: RecommendationResponseParser | None = None,
    ) -> None:
        """
        Args:
            api_key:        LLM API key.
            model:          LLM model identifier.
            max_tokens:     Max tokens for the completion.
            llm_client:     Injectable LLM client (pass MockLLMClient
                            in tests to avoid real API calls).
            prompt_builder: Injectable prompt builder.
            parser:         Injectable response parser.
        """
        # Build the client only if not injected (allows test mocking)
        self._llm_client = llm_client or AnthropicLLMClient(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            enable_web_search=True,   # Readers benefit from current book info
        )
        self._prompt_builder = prompt_builder or ReaderPromptBuilder()
        self._parser = parser or RecommendationResponseParser()

    def recommend(
        self,
        books: list[BookRecord],
        preferences: ReaderPreferences,
    ) -> list[BookRecommendation]:
        """
        Rank and summarise books for a reader using the LLM.

        If the book list is empty, returns an empty list immediately without
        making any API call (fast-fail, no wasted tokens).

        See BaseRecommendationService.recommend() for full contract.
        """
        if not books:
            logger.warning("recommendation_service_empty_book_list")
            return []

        logger.info(
            "recommendation_started",
            num_books=len(books),
            max_recs=preferences.max_recommendations,
        )

        # ── Step 1: Build prompts ──────────────────────────────────────────
        system_prompt, user_prompt = self._prompt_builder.build(books, preferences)

        # ── Step 2: Call the LLM ──────────────────────────────────────────
        # AnthropicLLMClient handles retries and web search internally.
        raw_response = self._llm_client.complete(system_prompt, user_prompt)

        # ── Step 3: Parse recommendations ─────────────────────────────────
        recommendations = self._parser.parse(raw_response, books)

        # ── Step 4: Enrich original BookRecords with LLM metadata ─────────
        # This updates genre/year/summary on the BookRecord objects so they
        # can be written back to the vector store with full enrichment.
        self._parser.enrich_book_records(raw_response, books)

        logger.info(
            "recommendation_complete",
            num_recommendations=len(recommendations),
        )

        return recommendations

# ----------------------------------------------------------------------------------------------


class GemmaRecommendationService(BaseRecommendationService):
    """
    Produces personalised reading recommendations via Gemma-2B light-weight LLM
    and an Embedding Model.

    DO NOT use Web Search. Tries to make best recommendation based on available 
    Google Book Records information.
    
    Also enriches incomplete book records (OCRValidatedResult -> BookRecord) 
    """

    def __init__(
        self,
        llm_client: GemmaLLMClient,
        genre_emb_file_path: str,
        embedding_model_path: str,
        prompt_builder: SummaryPromptGemma | None = None,
        parser: GemmaSummaryParser_InfoEnricher | None = None,  
    ) -> None:
        """
        Args:
            llm_client:        Injectable Gemma LLM client.
            prompt_builder:    Injectable prompt builder.
            summary_parser:    Injectable json response parser to get summary.
        """
        # Build the client only if not injected (allows test mocking)
        self._llm_client = llm_client
        self._prompt_builder = prompt_builder or SummaryPromptGemma()
        self._parser = parser or GemmaSummaryParser_InfoEnricher()
        self._genre_emb_file_path = genre_emb_file_path
        
        # Loading Embedding Model - all-MiniLM-L6-v2
        
        self._embedding_model_path = embedding_model_path
        self._emb_model = self.load_embedding_model()

    def recommend(
        self,
        books: list,
        preferences: ReaderPreferences,
    ) -> Tuple[list[BookRecommendation], list[BookRecord]]:
        """
        Summarise books for a reader using the LLM.
        Rank them using Embedding Model.
        Return list of BookRecommendation for reader.
        
        Enrich incoming books data for upsert to Vector DB.
        For each incoming book with OCRValidatedResult enrich and convert to BookRecord.
        
        If the book list is empty, returns an empty list immediately without
        doing any computation (fast-fail, no waste).
        """
        if not books:
            logger.warning("recommendation_service_empty_book_list")
            return [], []

        logger.info(
            "recommendation_started",
            num_books=len(books),
            max_recs=preferences.max_recommendations,
        )

        # ── Step 1: Segregate OCRValidatedResult and BookRecord book objects ────────────
        book_uncached = [book for book in books if isinstance(book, OCRValidatedResult)]
        book_cached = [book for book in books if isinstance(book, BookRecord)]
        
        # ── Step 2: Get summary for all books with description ──────────────────────────
        final_books: list = book_cached
        
        semi_enriched_books: list[BookRecord] = []   # Semi-enriched cause they still missing genre details
        
        # Need to get summary for books with OCRValidatedResult data model only
        for book in book_uncached:
            desc = book.description
            if desc:
                prompt = self._prompt_builder.build(desc)                    # Get prompt
                raw_response = self._llm_client.generate_summary(prompt)     # Get Gemma raw response
                summary = self._parser.extract_summary(raw_response)         # Parse out summary text
                enriched_book = self._parser.enrich_book_records(book_summary=summary, 
                                                                 original_book=book)
            else:
                enriched_book = self._parser.enrich_book_records(book_summary="", 
                                                                 original_book=book)
                
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
        final_books.extend(semi_enriched_books)
        
        
        # ── Step 5: Using reader preferences and book records to make recommendations ────
        book_rec_score: list[Tuple] = []
        
        
        # Get individual embeddings of user preference - genre, mood, and preferred length:
        user_prefs = [" ".join(preferences.preferred_genres), 
                      preferences.mood, 
                      preferences.preferred_length]
        
        user_pref_emb = self._emb_model.encode(user_prefs,
                                               convert_to_numpy=True,
                                               normalize_embeddings=True,
                                               )
        
        # Get recommendation score for each book and append to scores list
        for idx, book in enumerate(final_books):
            # finding book length category
            book_len_cate: str = None
            if book.pagecount > 0 and book.pagecount < 250:
                book_len_cate = "short"
            elif book.pagecount >= 250 and book.pagecount < 450:
                book_len_cate = "medium"
            elif book.pagecount >= 450:
                book_len_cate = "long"
            
            book_info = [book.genre or "Unknown / Unclassifiable" , 
                         book.summary or "Summary not available",
                         book_len_cate or "book length not available",
                         ]
            
            if preferences.preferred_length == 'any':       # zero weightage to book length for rec score
                rec_score = self.get_book_rec_score_via_embedding_model(user_pref_emb, 
                                                                        book_info,
                                                                        weights = [0.5, 0.5, 0])
            else:
                rec_score = self.get_book_rec_score_via_embedding_model(user_pref_emb, 
                                                                        book_info,
                                                                        weights = [0.4, 0.5, 0.1])
                
            if book.genre and book.summary:
                priority = 1               # Book with both genre and summary available will have top
            elif book.genre:               # reliability and priority for final recommendation and so on.
                priority = 2
            else:
                priority = 3
            
            book_rec_score.append((idx, priority, rec_score))
        
        # Sorting "book_rec_score" by Book Score and Priority Score:
        # Priority Score -> ASC ; and wihtin Book Rec Score -> DESC
        ranked_books = sorted(book_rec_score, key=lambda x: (x[1], -x[2]), reverse=False)

        # Generating a BookRecommendation for books as per max requested recommendation: 
        final_recommendation: list[BookRecommendation] = []
        i = 0
        
        for rank, book_det in zip(range(1, len(ranked_books) + 1), ranked_books):
            i += 1
            if i > preferences.max_recommendations:
                break
            
            book: BookRecord = final_books[book_det[0]]
            rec = BookRecommendation(
                book_id=book.book_id,
                title=book.title,
                subtitle=book.subtitle,
                author=book.author,
                genre=book.genre,
                crop_image_path=book.crop_image_path,
                summary=book.summary,
                rank=rank,
                match_reason="",
            )
            final_recommendation.append(rec)
           
        # NOTE: Using LLM to generate matching reason and adding it to book recommendations 
        #       is left for now.
        
        # ANOTHER IDEA: LLM Refinement
        # For “human-like ranking explanation”:
        # 1. Use LLM only after ranking
        # 2. Explain why these books match the user's taste
        # 3. Give input of - <User preference + top 3 books (summary, genre etc.)>
           
        return final_books, final_recommendation
        
        
    
    def load_embedding_model(self) -> None:
        """
        Load and set embedding model for generating text embeddings.

        Args:
            text (str): Text for which we want embeddings
        """
        logger.info("loading_MiniLm-L6-v2_embedding_model")
        
        from sentence_transformers import SentenceTransformer
        
        ## Laoding the embedding model:
        try:
            model = SentenceTransformer(self._embedding_model_path)
            logger.info("MiniLm-L6-v2_embedding_model_loaded")
            
            return model
            
        except Exception as exc:
            logger.error("Falied to load MiniLm-L6-v2_embedding_model", 
                         detail = {"message": f"Falied to load MiniLm-L6-v2_embedding_model. {exc}"})
    
      
    
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
        
        

    def get_book_rec_score_via_embedding_model(self,
                                               user_pref_emb: np.array, 
                                               book_info: list[str],
                                               weights: list[float],
                                               ) -> float:
        """
        Generate embedding for book_info text.
        Compute similarity score between user_pref_emb and embedding of book_info.
        Return the similarity score.

        Args:
            user_pref_emb (np.array):    User Preference text pre-generated embedding
            book_info (str):             Book information text
            weights (list[float]):       weughts for weighted sum

        Returns:
            float: Similarity score between user_pref_emb and embedding of book_info
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        ## Generating embedding for raw genre:
        book_info_emb = self._emb_model.encode(
                        book_info,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                
        ## Calculating similarity score:
        score_cate = ['genre', 'demand', 'length']
        scores: dict = {}
        
        for cate, user_pref, book_info in zip(score_cate, user_pref_emb, book_info_emb):
            similarity_scr = cosine_similarity(book_info_emb,
                                               user_pref_emb
                                               )[0][0]
            
            scores[cate] = similarity_scr
            
        final_score = scores['genre'] * weights[0] + scores['demand'] * weights[1] + scores['length'] * weights[2]
                    
        return final_score