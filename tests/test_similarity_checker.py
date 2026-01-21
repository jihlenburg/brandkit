"""
Tests for Similarity Checker
============================
Tests Soundex, Metaphone, Cologne Phonetics, and market-aware similarity checking.
"""

import pytest
import sys
from pathlib import Path

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from similarity_checker import (
    soundex,
    metaphone,
    cologne_phonetics,
    SimilarityChecker,
    SimilarityMatch,
    normalized_similarity,
)


class TestSoundex:
    """Tests for Soundex algorithm."""

    def test_basic_soundex(self):
        """Test basic Soundex encoding."""
        # Same Soundex for similar sounding names
        assert soundex("Robert") == soundex("Rupert")
        assert soundex("Smith") == soundex("Smyth")

    def test_soundex_format(self):
        """Test Soundex returns correct format (letter + 3 digits)."""
        code = soundex("Johnson")
        assert len(code) == 4
        assert code[0].isalpha()
        assert code[1:].isdigit()

    def test_soundex_preserves_first_letter(self):
        """Test first letter is preserved."""
        assert soundex("Michael")[0] == "M"
        assert soundex("Peter")[0] == "P"
        assert soundex("Zack")[0] == "Z"

    def test_empty_and_short(self):
        """Test edge cases."""
        # Empty string returns default code
        assert len(soundex("")) == 4  # Returns '0000' or similar
        assert len(soundex("A")) == 4


class TestMetaphone:
    """Tests for Metaphone algorithm."""

    def test_basic_metaphone(self):
        """Test basic Metaphone encoding."""
        # Similar sounding words should have similar codes
        assert metaphone("Smith") == metaphone("Smythe")

    def test_silent_letters(self):
        """Test silent letter handling."""
        # kn → N
        assert metaphone("Knight")[0] == "N"
        # wr → R
        assert metaphone("Write")[0] == "R"

    def test_ph_to_f(self):
        """Test ph → f conversion."""
        assert "F" in metaphone("Phoenix")

    def test_empty_string(self):
        """Test empty string."""
        assert metaphone("") == ""


class TestColognePhonetics:
    """Tests for Cologne Phonetics (Kölner Phonetik) algorithm."""

    def test_german_name_equivalents(self):
        """Test that German name variants produce same code."""
        # Mueller/Müller should be equivalent
        assert cologne_phonetics("Mueller") == cologne_phonetics("Müller")
        # Meyer/Meier should be equivalent
        assert cologne_phonetics("Meyer") == cologne_phonetics("Meier")
        # Schmidt/Schmitt should be equivalent
        assert cologne_phonetics("Schmidt") == cologne_phonetics("Schmitt")

    def test_umlaut_handling(self):
        """Test that umlauts are normalized."""
        # Umlauts should be converted before processing
        code_with_umlaut = cologne_phonetics("Müller")
        code_without = cologne_phonetics("Muller")
        # Should be same after umlaut normalization
        assert code_with_umlaut == code_without

    def test_eszett_handling(self):
        """Test ß handling."""
        assert cologne_phonetics("Straße") == cologne_phonetics("Strasse")

    def test_numeric_output(self):
        """Test that output is numeric string."""
        code = cologne_phonetics("Test")
        assert code.isdigit() or code == "0"

    def test_consecutive_duplicate_removal(self):
        """Test that consecutive duplicates are removed."""
        # Double letters should not produce double codes
        code = cologne_phonetics("Aachen")
        # Should not have consecutive identical digits (except 0s which are removed)
        for i in range(len(code) - 1):
            if code[i] != '0':
                assert code[i] != code[i + 1] or code[i] == '0'

    def test_empty_string(self):
        """Test empty string."""
        assert cologne_phonetics("") == ""

    def test_vowel_handling(self):
        """Test vowels are coded as 0 (typically removed from result)."""
        # A pure vowel word
        code = cologne_phonetics("AEI")
        # Should be mostly 0s or reduced
        assert code == "0" or all(c == '0' for c in code)


class TestSimilarityChecker:
    """Tests for SimilarityChecker class."""

    def test_init_with_markets(self):
        """Test initialization with markets parameter."""
        checker_en = SimilarityChecker(markets='en')
        checker_de = SimilarityChecker(markets='de')
        checker_both = SimilarityChecker(markets='en_de')

        assert checker_en.markets == 'en'
        assert checker_de.markets == 'de'
        assert checker_both.markets == 'en_de'

    def test_check_returns_result(self):
        """Test that check returns SimilarityResult."""
        checker = SimilarityChecker()
        result = checker.check("TestBrand")

        assert hasattr(result, 'is_safe')
        assert hasattr(result, 'similar_brands')
        assert hasattr(result, 'highest_similarity')

    def test_check_with_markets_override(self):
        """Test check with markets parameter override."""
        checker = SimilarityChecker(markets='en')
        # Should be able to override at check time
        result = checker.check("TestBrand", markets='de')
        # Should not raise

    def test_similar_brands_detection(self):
        """Test that similar brands are detected."""
        checker = SimilarityChecker()
        # Test with a name similar to a known brand
        # "Amazan" is similar to "Amazon"
        result = checker.check("Amazan")
        # If Amazon is in known brands, should detect similarity
        if any('amazon' in b.lower() for b in checker.known_brands):
            assert result.highest_similarity > 0.5

    def test_cologne_match_in_result(self):
        """Test that Cologne Phonetics match is included in results."""
        checker = SimilarityChecker()
        # Check a name - the match object should have cologne_match attribute
        result = checker.check("TestName")
        if result.similar_brands:
            match = result.similar_brands[0]
            assert hasattr(match, 'cologne_match')


class TestSimilarityMatch:
    """Tests for SimilarityMatch dataclass."""

    def test_is_problematic_soundex(self):
        """Test is_problematic with Soundex match."""
        match = SimilarityMatch(
            name="Test",
            known_brand="Tist",
            soundex_match=True,
            metaphone_match=False,
            cologne_match=False,
            text_similarity=0.5
        )
        assert match.is_problematic

    def test_is_problematic_metaphone(self):
        """Test is_problematic with Metaphone match."""
        match = SimilarityMatch(
            name="Test",
            known_brand="Tist",
            soundex_match=False,
            metaphone_match=True,
            cologne_match=False,
            text_similarity=0.5
        )
        assert match.is_problematic

    def test_is_problematic_cologne(self):
        """Test is_problematic with Cologne match."""
        match = SimilarityMatch(
            name="Test",
            known_brand="Tist",
            soundex_match=False,
            metaphone_match=False,
            cologne_match=True,
            text_similarity=0.5
        )
        assert match.is_problematic

    def test_is_problematic_high_similarity(self):
        """Test is_problematic with high text similarity."""
        match = SimilarityMatch(
            name="Test",
            known_brand="Tist",
            soundex_match=False,
            metaphone_match=False,
            cologne_match=False,
            text_similarity=0.8
        )
        assert match.is_problematic

    def test_is_safe(self):
        """Test safe match (no phonetic match, low similarity)."""
        match = SimilarityMatch(
            name="XYZ",
            known_brand="ABC",
            soundex_match=False,
            metaphone_match=False,
            cologne_match=False,
            text_similarity=0.2
        )
        assert not match.is_problematic

    def test_phonetic_match_count(self):
        """Test phonetic_match_count property."""
        match = SimilarityMatch(
            name="Test",
            known_brand="Tist",
            soundex_match=True,
            metaphone_match=True,
            cologne_match=True,
            text_similarity=0.5
        )
        assert match.phonetic_match_count == 3

        match2 = SimilarityMatch(
            name="Test",
            known_brand="Tist",
            soundex_match=True,
            metaphone_match=False,
            cologne_match=False,
            text_similarity=0.5
        )
        assert match2.phonetic_match_count == 1


class TestNormalizedSimilarity:
    """Tests for normalized_similarity function."""

    def test_identical_strings(self):
        """Test identical strings return 1.0."""
        assert normalized_similarity("test", "test") == 1.0

    def test_completely_different(self):
        """Test very different strings return low score."""
        score = normalized_similarity("abc", "xyz")
        assert score < 0.5

    def test_similar_strings(self):
        """Test similar strings return high score."""
        score = normalized_similarity("test", "tест")  # Note: Cyrillic 'е'
        # May not be high due to encoding differences
        # But "test" vs "tests" should be high
        score2 = normalized_similarity("test", "tests")
        assert score2 > 0.7

    def test_empty_strings(self):
        """Test empty string handling."""
        score = normalized_similarity("", "")
        assert score == 1.0  # Both empty = identical

        score2 = normalized_similarity("test", "")
        assert score2 == 0.0  # One empty = no similarity
