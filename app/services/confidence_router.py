# ─────────────────────────────────────────────────────────────────────────────
# app/services/confidence_router.py
#
# ThresholdConfidenceRouter – concrete implementation of BaseConfidenceRouter.
#
# Responsibility:
#   Partition a list of OCRResult objects into two buckets:
#     - auto_accepted:   confidence >= threshold → proceed to catalog/LLM
#     - flagged_for_review: confidence < threshold → sent to HITL review queue
#
# The threshold is injected at construction time from Settings, so changing
# OCR_CONFIDENCE_THRESHOLD in .env takes effect without touching this file.
#
# SOLID note (Single Responsibility):
#   This class does ONLY routing. It doesn't run OCR, it doesn't store anything,
#   it doesn't know about the review queue — it just splits a list.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from app.core.logging import get_logger
from app.models.book import OCRValidatedResult, OCRResult, BookRecord
from app.services.base import BaseConfidenceRouter

logger = get_logger(__name__)


class ThresholdConfidenceRouter(BaseConfidenceRouter):
    """
    Routes OCR results based on a fixed numeric confidence threshold.

    Books at or above the threshold are auto-accepted.
    Books below the threshold are flagged for librarian review.

    This is the simplest possible router. Future alternatives could:
      - Use a learned model that considers text length + detection quality
      - Apply different thresholds per language or book genre
      - Account for spine orientation confidence in the routing decision
    """

    def __init__(self, threshold: float = 0.75) -> None:
        """
        Args:
            threshold: Confidence value (inclusive) above which a book is
                       auto-accepted. Must be in [0.0, 1.0].
                       Default 0.75 means "confident OCR" = 75% composite score.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"Confidence threshold must be between 0.0 and 1.0, got {threshold}"
            )
        self._threshold = threshold

    def route(
        self,
        ocr_results: list,
    ) -> tuple[list, list]:
        """
        Split OCR results into (auto_accepted, flagged_for_review).

        Side effect: sets `flagged_for_review = True` on all flagged OCRValidatedResult
        objects. This mutation is intentional — the flag travels with the
        object through the rest of the pipeline so downstream code can check
        it without re-running the threshold comparison.

        Args:
            ocr_results: All OCR validated results from a single job.
                         Contains either BookRecord cached from vector DB 
                         or OCRValidatedResult
                         
        Returns:
            (auto_accepted, flagged_for_review) — both preserve input order.
        """
        auto_accepted: list = []
        flagged: list = []
        
        ocr_results_list = ocr_results.copy()
                
        for result in ocr_results_list:
            if isinstance(result, BookRecord):  # If book is an instance of BookRecord then it must 
                auto_accepted.append(result)    # be cached from DB put such books in auto_accepted list.
            else:
                if not result.title:
                    # No reliable OCR text available for validation. Must for REVIEW.
                    result.flagged_for_review = True
                    flagged.append(result)
                elif result.ocr_confidence >= self._threshold:
                    # High-confidence: proceed automatically
                    result.flagged_for_review = False
                    auto_accepted.append(result)
                else:
                    # Low-confidence: mark and send to review queue
                    result.flagged_for_review = True
                    flagged.append(result)

        logger.info(
            "confidence_routing_complete",
            total=len(ocr_results_list),
            auto_accepted=len(auto_accepted),
            flagged=len(flagged),
            threshold=self._threshold,
        )

        return auto_accepted, flagged

    @property
    def threshold(self) -> float:
        """Read-only access to the configured threshold (useful for logging)."""
        return self._threshold
