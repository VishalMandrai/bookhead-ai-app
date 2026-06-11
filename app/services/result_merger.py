# ─────────────────────────────────────────────────────────────────────────────
# app/services/result_merger.py
#
# DefaultResultMerger – concrete implementation of BaseResultMerger.
#
# Responsibility:
#   Merge the two output streams from the HITL correction stage:
#     Stream A: auto_accepted OCRResults (high-confidence, no human needed)
#     Stream B: human corrections (BookCorrection objects from the review UI)
#
#   Both streams are converted to BookRecord objects and returned as a
#   single unified list, preserving the original detection order.
#
# Why preserve original order?
#   The catalog is typically displayed shelf-left-to-shelf-right.
#   Preserving detection order keeps the catalog spatially coherent, which
#   helps librarians cross-reference the output against the physical shelf.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from app.core.logging import get_logger
from app.models.book import BookCorrection, BookRecord, OCRResult
from app.services.base import BaseResultMerger

logger = get_logger(__name__)


class DefaultResultMerger(BaseResultMerger):
    """
    Merges auto-accepted OCR results and human corrections into BookRecords.

    The merge preserves the original detection order by using the book_id
    as a stable key, then rebuilding the list in the order established by
    the combined auto_accepted + flagged_originals sequence.
    """

    def merge(
        self,
        auto_accepted: list[OCRResult],
        corrections: list[BookCorrection],
        flagged_originals: list[OCRResult],
    ) -> list[BookRecord]:
        """
        Produce a unified, ordered list of BookRecord objects.

        See BaseResultMerger.merge() for the full contract.

        Implementation detail – ordering:
          We rebuild order from auto_accepted + flagged_originals combined,
          because that pair covers ALL books in their original detection order.
          Each book_id is then resolved to either its OCR text or its
          human-supplied correction.
        """
        # ── Build lookup maps ──────────────────────────────────────────────

        # Map book_id → OCRResult for auto-accepted books
        auto_map: dict[str, OCRResult] = {
            r.book_id: r for r in auto_accepted
        }

        # Map book_id → BookCorrection for human-corrected books
        correction_map: dict[str, BookCorrection] = {
            c.book_id: c for c in corrections
        }

        # Map book_id → original OCRResult for flagged books
        # (needed to recover crop_image_path and ocr_confidence)
        flagged_map: dict[str, OCRResult] = {
            r.book_id: r for r in flagged_originals
        }

        # ── Determine the unified ordering ─────────────────────────────────
        # The full ordered sequence of book_ids is:
        # all auto_accepted (in detection order) + all flagged_originals (in detection order)
        # This reconstructs the left-to-right shelf order.
        ordered_ids: list[str] = (
            [r.book_id for r in auto_accepted]
            + [r.book_id for r in flagged_originals]
        )

        # ── Build BookRecord for each book_id in order ─────────────────────
        records: list[BookRecord] = []

        for book_id in ordered_ids:
            if book_id in auto_map:
                # ── Auto-accepted: use OCR text directly ──────────────────
                ocr = auto_map[book_id]
                record = BookRecord(
                    book_id=ocr.book_id,
                    title=ocr.raw_title,
                    author=ocr.raw_author,
                    crop_image_path=ocr.crop_image_path,
                    ocr_confidence=ocr.confidence,
                    source="ocr_auto",
                )
                records.append(record)

            elif book_id in correction_map:
                # ── Human-corrected: use librarian-supplied text ───────────
                correction = correction_map[book_id]
                original   = flagged_map.get(book_id)

                record = BookRecord(
                    book_id=correction.book_id,
                    title=correction.corrected_title,
                    author=correction.corrected_author,
                    # Recover crop path and original confidence from flagged map
                    crop_image_path=original.crop_image_path if original else "",
                    ocr_confidence=original.confidence if original else 0.0,
                    ori_ocr_ext_spine_txt=original.raw_title + " " + original.raw_author,
                    source="human_corrected",
                )
                records.append(record)

            else:
                # This should never happen if the pipeline is wired correctly.
                # Log a warning rather than silently dropping the book.
                logger.warning(
                    "result_merger_unresolved_book_id",
                    book_id=book_id,
                    detail=(
                        "book_id found in ordered list but not in auto_map "
                        "or correction_map — book will be skipped."
                    ),
                )

        logger.info(
            "result_merger_complete",
            total_records=len(records),
            auto_accepted=len(auto_accepted),
            human_corrected=len(corrections),
        )

        return records
