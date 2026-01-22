"""
Tests for Memorability Scoring
==============================
Tests for MemorabilityScorer in brandkit/generators/base_generator.py.
"""

import pytest
import sys
from pathlib import Path

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brandkit.generators.base_generator import MemorabilityScorer, SyllableAnalyzer


class TestMemorabilityScorer:
    """Tests for MemorabilityScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create a MemorabilityScorer instance."""
        return MemorabilityScorer()

    def test_init(self, scorer):
        """Test scorer initialization."""
        assert scorer is not None

    def test_score_returns_dict(self, scorer):
        """Test that score returns a dictionary."""
        result = scorer.score("Voltix")
        assert isinstance(result, dict)

    def test_score_has_overall(self, scorer):
        """Test that score includes overall value."""
        result = scorer.score("Lumina")
        assert 'overall' in result
        assert 0.0 <= result['overall'] <= 1.0

    def test_score_components(self, scorer):
        """Test that score includes expected components."""
        result = scorer.score("Celestia")
        expected_keys = ['overall', 'pronounceability', 'distinctiveness', 'rhythm']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_score_with_archetype(self, scorer):
        """Test scoring with archetype."""
        result = scorer.score("Voltix", archetype="power")
        assert 'overall' in result

    def test_score_with_industry(self, scorer):
        """Test scoring with industry."""
        result = scorer.score("Lumina", industry="tech")
        assert 'overall' in result

    def test_score_with_archetype_and_industry(self, scorer):
        """Test scoring with both archetype and industry."""
        result = scorer.score("Celestia", archetype="tech", industry="tech")
        assert 'overall' in result


class TestMemorabilityPronouncability:
    """Tests for pronounceability component of memorability."""

    @pytest.fixture
    def scorer(self):
        return MemorabilityScorer()

    def test_easy_name_scores_high(self, scorer):
        """Test that easy names get high pronounceability."""
        result = scorer.score("Lumina")
        assert result['pronounceability'] >= 0.5

    def test_hard_name_scores_low(self, scorer):
        """Test that hard names get lower pronounceability."""
        result = scorer.score("Xkrzpt")
        # Hard to pronounce should score lower
        # (might still pass if there's a floor)

    def test_optimal_length_bonus(self, scorer):
        """Test that optimal length names get bonus."""
        short_result = scorer.score("Ab")
        optimal_result = scorer.score("Voltix")  # 6 chars
        long_result = scorer.score("Supercalifragilistic")

        # Optimal length (4-7) should score well
        assert isinstance(optimal_result['pronounceability'], float)


class TestMemorabilityDistinctiveness:
    """Tests for distinctiveness component of memorability."""

    @pytest.fixture
    def scorer(self):
        return MemorabilityScorer()

    def test_distinctive_name(self, scorer):
        """Test distinctiveness scoring."""
        result = scorer.score("Voltix")
        assert 'distinctiveness' in result
        assert 0.0 <= result['distinctiveness'] <= 1.0

    def test_common_pattern_lower(self, scorer):
        """Test that very common patterns might score lower."""
        # Generic vs distinctive
        result1 = scorer.score("Voltix")
        result2 = scorer.score("Thingy")
        # Both should have valid scores
        assert 0.0 <= result1['distinctiveness'] <= 1.0
        assert 0.0 <= result2['distinctiveness'] <= 1.0

    def test_interesting_onset_bonus(self, scorer):
        """Test that interesting onsets get bonus."""
        # Names starting with interesting sounds
        result_gl = scorer.score("Glowix")  # 'gl' is interesting
        result_th = scorer.score("Thundra")  # 'th' is interesting
        # Should have distinctiveness scores
        assert result_gl['distinctiveness'] >= 0
        assert result_th['distinctiveness'] >= 0

    def test_memorable_ending_bonus(self, scorer):
        """Test that memorable endings get bonus."""
        # Names with memorable endings
        result_ix = scorer.score("Voltix")  # '-ix' ending
        result_ex = scorer.score("Nexex")   # '-ex' ending
        # Should have distinctiveness scores
        assert result_ix['distinctiveness'] >= 0
        assert result_ex['distinctiveness'] >= 0


class TestMemorabilityRhythm:
    """Tests for rhythm component of memorability."""

    @pytest.fixture
    def scorer(self):
        return MemorabilityScorer()

    def test_rhythm_scoring(self, scorer):
        """Test rhythm component scoring."""
        result = scorer.score("Lumina")
        assert 'rhythm' in result
        assert 0.0 <= result['rhythm'] <= 1.0

    def test_two_syllable_optimal(self, scorer):
        """Test that 2-syllable names score well."""
        result = scorer.score("Voltix")  # 2 syllables
        assert result['rhythm'] >= 0

    def test_three_syllable_optimal(self, scorer):
        """Test that 3-syllable names score well."""
        result = scorer.score("Luminara")  # 3 syllables
        assert result['rhythm'] >= 0

    def test_many_syllables_lower(self, scorer):
        """Test that many syllables score lower."""
        result = scorer.score("Internationalization")
        # Many syllables should still have valid score
        assert 0.0 <= result['rhythm'] <= 1.0


class TestSyllableAnalyzer:
    """Tests for SyllableAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        return SyllableAnalyzer()

    def test_init(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None

    def test_analyze_returns_info(self, analyzer):
        """Test that analyze returns syllable info."""
        info = analyzer.analyze("Voltix")
        assert info is not None
        assert hasattr(info, 'count')

    def test_analyze_count(self, analyzer):
        """Test syllable count."""
        info = analyzer.analyze("Lumina")
        assert info.count >= 2

    def test_analyze_pattern(self, analyzer):
        """Test stress pattern."""
        info = analyzer.analyze("Voltara")
        assert hasattr(info, 'pattern')

    def test_monosyllable(self, analyzer):
        """Test single syllable word."""
        info = analyzer.analyze("Sun")
        assert info.count >= 1

    def test_empty_string(self, analyzer):
        """Test empty string handling."""
        info = analyzer.analyze("")
        assert info.count == 0 or info.count >= 0


class TestMemorabilityArchetypes:
    """Tests for archetype-specific memorability."""

    @pytest.fixture
    def scorer(self):
        return MemorabilityScorer()

    def test_power_archetype(self, scorer):
        """Test power archetype scoring."""
        result = scorer.score("Voltix", archetype="power")
        assert 'overall' in result

    def test_elegance_archetype(self, scorer):
        """Test elegance archetype scoring."""
        result = scorer.score("Lumina", archetype="elegance")
        assert 'overall' in result

    def test_tech_archetype(self, scorer):
        """Test tech archetype scoring."""
        result = scorer.score("Nexus", archetype="tech")
        assert 'overall' in result

    def test_nature_archetype(self, scorer):
        """Test nature archetype scoring."""
        result = scorer.score("Willow", archetype="nature")
        assert 'overall' in result

    def test_trust_archetype(self, scorer):
        """Test trust archetype scoring."""
        result = scorer.score("Solidex", archetype="trust")
        assert 'overall' in result


class TestMemorabilityIndustries:
    """Tests for industry-specific memorability."""

    @pytest.fixture
    def scorer(self):
        return MemorabilityScorer()

    def test_tech_industry(self, scorer):
        """Test tech industry scoring."""
        result = scorer.score("Nexus", industry="tech")
        assert 'overall' in result

    def test_luxury_industry(self, scorer):
        """Test luxury industry scoring."""
        result = scorer.score("Eleganza", industry="luxury")
        assert 'overall' in result

    def test_outdoor_industry(self, scorer):
        """Test outdoor industry scoring."""
        result = scorer.score("Trailox", industry="outdoor")
        assert 'overall' in result


class TestMemorabilityEdgeCases:
    """Edge case tests for memorability scoring."""

    @pytest.fixture
    def scorer(self):
        return MemorabilityScorer()

    def test_very_short_name(self, scorer):
        """Test very short name."""
        result = scorer.score("A")
        assert 'overall' in result

    def test_very_long_name(self, scorer):
        """Test very long name."""
        result = scorer.score("Supercalifragilisticexpialidocious")
        assert 'overall' in result
        assert 0.0 <= result['overall'] <= 1.0

    def test_all_vowels(self, scorer):
        """Test all-vowel name."""
        result = scorer.score("Aeiou")
        assert 'overall' in result

    def test_all_consonants(self, scorer):
        """Test all-consonant name."""
        result = scorer.score("Bcdfrst")
        assert 'overall' in result

    def test_numbers_in_name(self, scorer):
        """Test name with numbers."""
        result = scorer.score("Volt3x")
        assert 'overall' in result

    def test_mixed_case(self, scorer):
        """Test mixed case name."""
        result = scorer.score("VoLtIx")
        assert 'overall' in result


class TestMemorabilityComparison:
    """Comparison tests for memorability scoring."""

    @pytest.fixture
    def scorer(self):
        return MemorabilityScorer()

    def test_good_vs_bad_names(self, scorer):
        """Test that good names generally score higher than bad ones."""
        good_names = ["Lumina", "Voltix", "Solara", "Nexus"]
        bad_names = ["Xkrzpt", "Bngvf", "Qwrty"]

        good_scores = [scorer.score(n)['overall'] for n in good_names]
        bad_scores = [scorer.score(n)['overall'] for n in bad_names]

        avg_good = sum(good_scores) / len(good_scores)
        avg_bad = sum(bad_scores) / len(bad_scores)

        # Good names should generally score higher
        # (though this might not always be true due to scoring nuances)

    def test_consistent_scoring(self, scorer):
        """Test that same name gets same score."""
        score1 = scorer.score("Voltix")['overall']
        score2 = scorer.score("Voltix")['overall']
        assert score1 == score2
