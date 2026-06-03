# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/services/test_text_parser.py
#
# Unit tests for SpineTextParser.
# All tests use synthetic EasyOCR-format boxes — no model loading needed.
# ─────────────────────────────────────────────────────────────────────────────

import pytest
from app.services.text_parser import SpineTextParser


# ── Helpers ────────────────────────────────────────────────────────────────────

def _box(text: str, confidence: float, y_top: float = 10.0) -> tuple:
    """
    Build a minimal EasyOCR-format box.
    EasyOCR format: ([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, confidence)
    We set a simple rectangular box; only the y-coordinates matter for ordering.
    """
    return (
        [[0, y_top], [100, y_top], [100, y_top + 20], [0, y_top + 20]],
        text,
        confidence,
    )


class TestSpineTextParserFiltering:

    def test_empty_input_returns_empty_result(self):
        parser = SpineTextParser()
        result = parser.parse([])
        assert result.title == ""
        assert result.author == ""
        assert result.confidence == 0.0

    def test_barcode_digits_are_filtered(self):
        """A box containing only digits (≥8) should be treated as a barcode and filtered."""
        parser = SpineTextParser()
        boxes = [
            _box("9780451524935", 0.99, y_top=0),   # ISBN-like digits
            _box("The Great Gatsby", 0.92, y_top=20),
        ]
        result = parser.parse(boxes)
        assert "9780451524935" not in result.title
        assert result.title == "The Great Gatsby"

    def test_very_low_confidence_box_filtered(self):
        """Boxes with confidence < 0.1 must be discarded."""
        parser = SpineTextParser()
        boxes = [
            _box("Noise Fragment", 0.05, y_top=0),  # Below threshold
            _box("Real Title", 0.88, y_top=30),
        ]
        result = parser.parse(boxes)
        assert result.title == "Real Title"
        assert "Noise Fragment" not in result.title

    def test_single_character_box_filtered(self):
        """A single-character detection is noise, not meaningful text."""
        parser = SpineTextParser()
        boxes = [
            _box("I", 0.95, y_top=0),       # Single char noise
            _box("Dune", 0.91, y_top=30),
        ]
        result = parser.parse(boxes)
        assert result.title == "Dune"

    def test_all_noise_returns_zero_confidence(self):
        """If every box is filtered, confidence should be 0.0."""
        parser = SpineTextParser()
        boxes = [
            _box("12345678", 0.95, y_top=0),  # barcode
            _box("X", 0.95, y_top=20),        # single char
        ]
        result = parser.parse(boxes)
        assert result.confidence == 0.0


class TestSpineTextParserOrdering:

    def test_boxes_sorted_top_to_bottom(self):
        """Boxes with lower y values (closer to top) should appear before lower ones."""
        parser = SpineTextParser()
        # Deliberately put author text ABOVE title in the list (wrong order)
        # but with a higher y value (lower on the image)
        boxes = [
            _box("F. Scott Fitzgerald", 0.88, y_top=200),  # bottom of spine
            _box("The Great Gatsby",    0.95, y_top=10),   # top of spine
        ]
        result = parser.parse(boxes)
        # Title should come from the top (y=10) block
        assert result.title == "The Great Gatsby"


class TestSpineTextParserTitleAuthorAssignment:

    def test_single_block_is_title(self):
        """With only one text block, it should be the title."""
        parser = SpineTextParser()
        boxes = [_box("Dune", 0.95, y_top=10)]
        result = parser.parse(boxes)
        assert result.title == "Dune"
        assert result.author == ""

    def test_author_detected_by_pattern(self):
        """A block matching 'Firstname Lastname' pattern should be the author."""
        parser = SpineTextParser()
        boxes = [
            _box("The Hitchhiker's Guide to the Galaxy", 0.91, y_top=10),
            _box("Douglas Adams", 0.87, y_top=80),
        ]
        result = parser.parse(boxes)
        assert result.title == "The Hitchhiker's Guide to the Galaxy"
        assert result.author == "Douglas Adams"

    def test_by_prefix_detected_as_author(self):
        """A block starting with 'by' should be identified as author."""
        parser = SpineTextParser()
        boxes = [
            _box("1984", 0.93, y_top=10),
            _box("by George Orwell", 0.88, y_top=60),
        ]
        result = parser.parse(boxes)
        assert result.author == "by George Orwell"

    def test_initials_detected_as_author(self):
        """A block like 'J.K. Rowling' (with initials) should be author."""
        parser = SpineTextParser()
        boxes = [
            _box("Harry Potter and the Philosopher's Stone", 0.92, y_top=10),
            _box("J.K. Rowling", 0.88, y_top=80),
        ]
        result = parser.parse(boxes)
        assert result.author == "J.K. Rowling"

    def test_fallback_longest_is_title(self):
        """
        When no author pattern matches, the longest block should be the title
        and the second-longest the author.
        """
        parser = SpineTextParser()
        boxes = [
            _box("Short", 0.90, y_top=10),
            _box("A Much Longer Title Text Here", 0.88, y_top=50),
        ]
        result = parser.parse(boxes)
        assert result.title == "A Much Longer Title Text Here"
        assert result.author == "Short"


class TestSpineTextParserConfidence:

    def test_confidence_is_between_zero_and_one(self):
        """Composite confidence must always be in [0, 1]."""
        parser = SpineTextParser()
        boxes = [
            _box("Some Title", 0.75, y_top=10),
            _box("Some Author", 0.65, y_top=60),
        ]
        result = parser.parse(boxes)
        assert 0.0 <= result.confidence <= 1.0

    def test_high_confidence_boxes_produce_high_score(self):
        """Two boxes both at 0.95 should produce a composite near 0.95."""
        parser = SpineTextParser()
        boxes = [
            _box("Clear Title", 0.95, y_top=10),
            _box("Clear Author", 0.95, y_top=60),
        ]
        result = parser.parse(boxes)
        assert result.confidence >= 0.90

    def test_low_confidence_boxes_produce_low_score(self):
        """Two boxes at 0.30 should produce a composite near 0.30."""
        parser = SpineTextParser()
        boxes = [
            _box("Blurry Text", 0.30, y_top=10),
            _box("More Blurry", 0.30, y_top=60),
        ]
        result = parser.parse(boxes)
        assert result.confidence < 0.50

    def test_longer_text_weighted_more(self):
        """
        A short low-confidence box should have less weight than a long
        high-confidence box, pulling the composite closer to the longer one.
        """
        parser = SpineTextParser()
        boxes = [
            # Long high-confidence block
            _box("The Complete History of Everything", 0.92, y_top=10),
            # Short low-confidence block (should have low weight)
            _box("Hi", 0.20, y_top=60),
        ]
        result = parser.parse(boxes)
        # Should be pulled toward 0.92, not dragged down to (0.92+0.20)/2 = 0.56
        assert result.confidence > 0.70

    def test_very_short_total_text_penalised(self):
        """
        If all detected text is very short (< 5 chars total), confidence
        should be penalised (halved).
        """
        parser = SpineTextParser()
        boxes = [_box("Hi", 0.90, y_top=10)]   # Total text length = 2
        result = parser.parse(boxes)
        # With penalty, confidence should be well below 0.9
        assert result.confidence < 0.55

    def test_raw_blocks_preserved(self):
        """raw_blocks should contain the original unmodified OCR boxes."""
        parser = SpineTextParser()
        boxes = [_box("Dune", 0.95, y_top=10)]
        result = parser.parse(boxes)
        assert result.raw_blocks == boxes
