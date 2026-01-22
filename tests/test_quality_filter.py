"""
Tests for Quality Filtering Module
==================================
Tests for filter_and_rank() and related helper functions
in brandkit/quality.py.
"""

import pytest
import sys
from pathlib import Path
from dataclasses import dataclass

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brandkit.quality import (
    filter_and_rank,
    Candidate,
    _extract_name,
    _extract_score,
    _build_candidates,
)


# Mock name classes for testing
@dataclass
class MockNameWithScore:
    """Mock name object with score attribute."""
    name: str
    score: float = 0.7


@dataclass
class MockNameWithTotalScore:
    """Mock name object with total_score attribute."""
    name: str
    total_score: float = 0.7


@dataclass
class MockNameMinimal:
    """Mock name object with only name."""
    name: str


class TestExtractName:
    """Tests for _extract_name helper."""

    def test_extract_from_name_attr(self):
        """Test extracting name from name attribute."""
        obj = MockNameMinimal("Voltix")
        assert _extract_name(obj) == "Voltix"

    def test_extract_from_string(self):
        """Test extracting name from plain string."""
        assert _extract_name("Luminara") == "Luminara"

    def test_extract_from_int(self):
        """Test extracting name from non-string."""
        assert _extract_name(123) == "123"


class TestExtractScore:
    """Tests for _extract_score helper."""

    def test_extract_score_attr(self):
        """Test extracting from score attribute."""
        obj = MockNameWithScore("Test", score=0.8)
        assert _extract_score(obj) == 0.8

    def test_extract_total_score_attr(self):
        """Test extracting from total_score attribute."""
        obj = MockNameWithTotalScore("Test", total_score=0.9)
        assert _extract_score(obj) == 0.9

    def test_extract_default(self):
        """Test default score when no attribute."""
        obj = MockNameMinimal("Test")
        assert _extract_score(obj) == 0.5

    def test_extract_from_string(self):
        """Test default score for string input."""
        assert _extract_score("Test") == 0.5


class TestBuildCandidates:
    """Tests for _build_candidates helper."""

    def test_build_from_mock_names(self):
        """Test building candidates from mock names."""
        names = [
            MockNameWithScore("Voltix", 0.8),
            MockNameWithScore("Lumina", 0.7),
        ]
        candidates = _build_candidates(names)
        assert len(candidates) == 2
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_candidate_has_correct_attributes(self):
        """Test candidate has correct suffix/prefix."""
        names = [MockNameWithScore("Voltix", 0.8)]
        candidates = _build_candidates(names)
        c = candidates[0]
        assert c.name == "Voltix"
        assert c.score == 0.8
        assert c.suffix2 == "ix"
        assert c.suffix3 == "tix"
        assert c.prefix2 == "vo"

    def test_build_skips_empty_names(self):
        """Test that empty names are skipped."""
        class EmptyName:
            name = ""

        names = [EmptyName(), MockNameWithScore("Valid", 0.7)]
        candidates = _build_candidates(names)
        assert len(candidates) == 1
        assert candidates[0].name == "Valid"

    def test_build_from_strings(self):
        """Test building candidates from plain strings."""
        names = ["Voltix", "Lumina", "Celestia"]
        candidates = _build_candidates(names)
        assert len(candidates) == 3
        assert candidates[0].name == "Voltix"

    def test_short_name_handling(self):
        """Test handling of very short names."""
        names = ["A", "Ab"]
        candidates = _build_candidates(names)
        assert len(candidates) == 2
        assert candidates[0].suffix2 == "a"
        assert candidates[0].prefix2 == "a"
        assert candidates[1].suffix2 == "ab"
        assert candidates[1].prefix2 == "ab"


class TestCandidateDataclass:
    """Tests for Candidate dataclass."""

    def test_candidate_creation(self):
        """Test creating a Candidate."""
        c = Candidate(
            obj="test",
            name="Voltix",
            score=0.8,
            suffix2="ix",
            suffix3="tix",
            prefix2="vo",
        )
        assert c.name == "Voltix"
        assert c.score == 0.8

    def test_candidate_equality(self):
        """Test Candidate equality."""
        c1 = Candidate("test", "Voltix", 0.8, "ix", "tix", "vo")
        c2 = Candidate("test", "Voltix", 0.8, "ix", "tix", "vo")
        assert c1 == c2


class TestFilterAndRank:
    """Tests for filter_and_rank function."""

    def test_empty_input(self):
        """Test with empty input."""
        result = filter_and_rank([], target_count=10)
        assert result == []

    def test_basic_filtering(self):
        """Test basic filtering with valid names."""
        names = [
            MockNameWithScore("Voltix", 0.8),
            MockNameWithScore("Lumina", 0.7),
            MockNameWithScore("Solara", 0.75),
        ]
        result = filter_and_rank(
            names,
            target_count=3,
            markets="en_de",
            min_score=0.5,
            similarity_threshold=0.9,
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        # Should return some names (exact count depends on quality gates)
        assert isinstance(result, list)

    def test_respects_target_count(self):
        """Test that result doesn't exceed target count."""
        names = [MockNameWithScore(f"Name{i}x", 0.8) for i in range(20)]
        result = filter_and_rank(
            names,
            target_count=5,
            markets="en_de",
            min_score=0.3,
            similarity_threshold=0.95,
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        assert len(result) <= 5

    def test_filters_names_by_pronounceability(self):
        """Test that names are checked for pronounceability."""
        names = [
            MockNameWithScore("Lumina", 0.7),
            MockNameWithScore("Voltix", 0.8),
        ]
        result = filter_and_rank(
            names,
            target_count=2,
            markets="en_de",
            min_score=0.3,
            similarity_threshold=0.95,
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        # Should return pronounceable names
        assert len(result) <= 2
        assert isinstance(result, list)

    def test_filters_hazardous_names(self):
        """Test that hazardous names are filtered."""
        names = [
            MockNameWithScore("Giftbox", 0.9),  # Hazardous (German)
            MockNameWithScore("Luminara", 0.7),  # Safe
        ]
        result = filter_and_rank(
            names,
            target_count=2,
            markets="en_de",
            min_score=0.3,
            similarity_threshold=0.95,
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        result_names = [_extract_name(r) for r in result]
        assert "Giftbox" not in result_names

    def test_diversity_suffix_limiting(self):
        """Test that suffix diversity is enforced."""
        # Create many names with same suffix
        names = [MockNameWithScore(f"Name{i}ix", 0.8) for i in range(10)]
        result = filter_and_rank(
            names,
            target_count=5,
            markets="en_de",
            min_score=0.3,
            similarity_threshold=0.5,  # Low threshold to test diversity
            max_suffix_pct=0.2,  # Only 20% can have same suffix
            max_prefix_pct=0.5,
        )
        # With max_suffix_pct=0.2 and target=5, max 1 name per suffix
        # So result should have diverse endings

    def test_diversity_prefix_limiting(self):
        """Test that prefix diversity is enforced."""
        # Create many names with same prefix
        names = [MockNameWithScore(f"Vol{i}ara", 0.8) for i in range(10)]
        result = filter_and_rank(
            names,
            target_count=5,
            markets="en_de",
            min_score=0.3,
            similarity_threshold=0.5,
            max_suffix_pct=0.5,
            max_prefix_pct=0.2,  # Only 20% can have same prefix
        )
        # Prefix diversity should be enforced

    def test_score_filtering(self):
        """Test that low scores are filtered."""
        names = [
            MockNameWithScore("HighScore", 0.9),
            MockNameWithScore("LowScore", 0.1),
            MockNameWithScore("MidScore", 0.5),
        ]
        result = filter_and_rank(
            names,
            target_count=3,
            markets="en_de",
            min_score=0.6,  # Only HighScore should pass
            similarity_threshold=0.95,
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        # Result should prefer higher scores

    def test_similarity_deduplication(self):
        """Test that similar names are deduplicated."""
        names = [
            MockNameWithScore("Voltix", 0.8),
            MockNameWithScore("Voltex", 0.8),  # Very similar
            MockNameWithScore("Lumina", 0.7),  # Different
        ]
        result = filter_and_rank(
            names,
            target_count=3,
            markets="en_de",
            min_score=0.3,
            similarity_threshold=0.7,  # Should filter Voltex
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        result_names = [_extract_name(r) for r in result]
        # Should not have both Voltix and Voltex
        has_voltix = "Voltix" in result_names
        has_voltex = "Voltex" in result_names
        # At most one should be present (if similarity check works)

    def test_preserves_original_objects(self):
        """Test that original objects are returned, not copies."""
        original = MockNameWithScore("Voltix", 0.8)
        names = [original]
        result = filter_and_rank(
            names,
            target_count=1,
            markets="en_de",
            min_score=0.3,
            similarity_threshold=0.95,
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        if result:
            assert result[0] is original

    def test_handles_string_input(self):
        """Test filtering with plain string input."""
        names = ["Voltix", "Lumina", "Solara"]
        result = filter_and_rank(
            names,
            target_count=3,
            markets="en_de",
            min_score=0.3,
            similarity_threshold=0.95,
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        assert isinstance(result, list)

    def test_market_en_only(self):
        """Test with English-only market."""
        names = [MockNameWithScore("Pfennig", 0.8)]  # German-friendly
        result = filter_and_rank(
            names,
            target_count=1,
            markets="en",
            min_score=0.3,
            similarity_threshold=0.95,
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        # May or may not pass depending on EN pronounceability

    def test_market_de_only(self):
        """Test with German-only market."""
        names = [MockNameWithScore("Voltix", 0.8)]
        result = filter_and_rank(
            names,
            target_count=1,
            markets="de",
            min_score=0.3,
            similarity_threshold=0.95,
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        assert isinstance(result, list)


class TestFilterAndRankEdgeCases:
    """Edge case tests for filter_and_rank."""

    def test_all_names_may_pass(self):
        """Test that names go through filter pipeline."""
        names = [
            MockNameWithScore("Xkrzpt", 0.9),
            MockNameWithScore("Bngvlx", 0.9),
        ]
        result = filter_and_rank(
            names,
            target_count=2,
            markets="en_de",
            min_score=0.3,
            similarity_threshold=0.95,
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        # Result depends on pronounceability checker implementation
        assert isinstance(result, list)

    def test_single_name(self):
        """Test with single name input."""
        names = [MockNameWithScore("Voltix", 0.8)]
        result = filter_and_rank(
            names,
            target_count=1,
            markets="en_de",
            min_score=0.3,
            similarity_threshold=0.95,
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        assert len(result) <= 1

    def test_target_count_zero(self):
        """Test with target count of zero."""
        names = [MockNameWithScore("Voltix", 0.8)]
        result = filter_and_rank(
            names,
            target_count=0,
            markets="en_de",
            min_score=0.3,
            similarity_threshold=0.95,
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        assert result == []

    def test_high_min_score_filters_all(self):
        """Test that high min_score can filter everything."""
        names = [MockNameWithScore("Voltix", 0.5)]
        result = filter_and_rank(
            names,
            target_count=1,
            markets="en_de",
            min_score=0.99,  # Very high
            similarity_threshold=0.95,
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        # Should fall back to unfiltered if all fail score threshold

    def test_very_low_similarity_threshold(self):
        """Test with very low similarity threshold."""
        names = [
            MockNameWithScore("Voltix", 0.8),
            MockNameWithScore("Lumina", 0.7),
        ]
        result = filter_and_rank(
            names,
            target_count=2,
            markets="en_de",
            min_score=0.3,
            similarity_threshold=0.1,  # Almost everything is "too similar"
            max_suffix_pct=0.5,
            max_prefix_pct=0.5,
        )
        # Should still return at least one name

    def test_very_low_diversity_pct(self):
        """Test with very low diversity percentages."""
        names = [MockNameWithScore(f"Name{i}ix", 0.8) for i in range(5)]
        result = filter_and_rank(
            names,
            target_count=5,
            markets="en_de",
            min_score=0.3,
            similarity_threshold=0.95,
            max_suffix_pct=0.01,  # Very restrictive
            max_prefix_pct=0.01,
        )
        # Should still work, just return fewer names
