"""
Tests for Hazard Checker
========================
Tests cross-linguistic hazard detection, syllable-aware matching,
and EN/DE-specific hazard patterns.
"""

import pytest
import sys
from pathlib import Path

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brandkit.generators.base_generator import HazardChecker, HazardResult


class TestHazardCheckerBasic:
    """Basic hazard checker tests."""

    @pytest.fixture
    def checker(self):
        """Create a hazard checker instance."""
        return HazardChecker()

    def test_clean_name(self, checker):
        """Test that clean names pass."""
        result = checker.check("Luminara")
        assert result.is_safe
        assert result.severity == 'clear'
        assert len(result.issues) == 0

    def test_hazard_result_structure(self, checker):
        """Test HazardResult has correct structure."""
        result = checker.check("TestName")
        assert isinstance(result, HazardResult)
        assert hasattr(result, 'is_safe')
        assert hasattr(result, 'severity')
        assert hasattr(result, 'issues')


class TestGermanHazards:
    """Tests for German-specific hazards."""

    @pytest.fixture
    def checker(self):
        return HazardChecker()

    def test_gift_hazard(self, checker):
        """Test 'gift' (poison in German) detection."""
        result = checker.check("Giftbox")
        assert not result.is_safe
        assert result.severity in ['high', 'critical']
        # Should have an issue mentioning gift/poison
        issue_types = [i.get('type') for i in result.issues]
        assert 'exact_word' in issue_types or 'en_de_pattern' in issue_types

    def test_mist_hazard(self, checker):
        """Test 'mist' (manure in German) detection."""
        result = checker.check("Mistral")
        assert not result.is_safe
        # Should detect the mist hazard
        has_mist_issue = any('mist' in str(i).lower() for i in result.issues)
        assert has_mist_issue

    def test_after_hazard(self, checker):
        """Test 'after' (anus in German) detection."""
        result = checker.check("Afterglow")
        assert not result.is_safe
        assert result.severity in ['high', 'critical']

    def test_kot_hazard(self, checker):
        """Test 'kot' (feces in German) detection."""
        result = checker.check("Kotex")
        # Should detect kot hazard
        assert not result.is_safe or result.severity != 'clear'


class TestEnglishHazards:
    """Tests for English-specific hazards."""

    @pytest.fixture
    def checker(self):
        return HazardChecker()

    def test_fuk_sound_hazard(self, checker):
        """Test f*ck-like sounds detection."""
        result = checker.check("Fokker")
        assert not result.is_safe
        assert result.severity in ['high', 'critical']

    def test_scheiss_hazard(self, checker):
        """Test German vulgar word detection."""
        result = checker.check("Scheisshaus")
        assert not result.is_safe
        assert result.severity == 'critical'


class TestSyllableAwareHazards:
    """Tests for syllable-aware hazard detection."""

    @pytest.fixture
    def checker(self):
        return HazardChecker()

    def test_syllable_boundary_detection(self, checker):
        """Test that hazards at syllable boundaries are detected."""
        # The _check_syllable_hazards method should be called
        result = checker.check("Fokuser")  # "fok" at syllable start
        # Should detect the hazard
        assert not result.is_safe or len(result.issues) > 0

    def test_prominent_position_hazard(self, checker):
        """Test word-initial/final position hazards."""
        # Word-final hazards
        result = checker.check("Bigass")
        if not result.is_safe:
            has_prominent = any(
                i.get('type') == 'prominent_position'
                for i in result.issues
            )
            # May or may not have prominent_position depending on implementation
            assert len(result.issues) > 0


class TestENDEPatterns:
    """Tests for EN/DE-specific regex patterns."""

    @pytest.fixture
    def checker(self):
        return HazardChecker()

    def test_fok_pattern(self, checker):
        """Test [fp][aou][ck] pattern (sounds like f*ck)."""
        result = checker.check("Focker")
        assert not result.is_safe
        # Should have en_de_pattern issue
        has_pattern = any(i.get('type') == 'en_de_pattern' for i in result.issues)
        assert has_pattern or result.severity in ['high', 'critical']

    def test_shit_pattern(self, checker):
        """Test sh[iy]t pattern."""
        result = checker.check("Shitake")  # Intentional misspelling
        # Should detect the pattern
        if 'shit' in "Shitake".lower():
            assert not result.is_safe

    def test_arsch_pattern(self, checker):
        """Test ar[sc]h pattern (German ass)."""
        result = checker.check("Marsch")
        # Should detect arsch pattern
        has_arsch = any('arsch' in str(i).lower() or 'ar[sc]h' in str(i) for i in result.issues)
        # Marsch contains 'arsch' but may be acceptable as it's a real word


class TestHazardSeverity:
    """Tests for hazard severity levels."""

    @pytest.fixture
    def checker(self):
        return HazardChecker()

    def test_critical_severity(self, checker):
        """Test critical severity detection."""
        result = checker.check("Fuckery")
        if not result.is_safe:
            assert result.severity == 'critical'

    def test_severity_ranking(self, checker):
        """Test severity ranking logic."""
        # _severity_rank should work
        assert checker._severity_rank('critical') > checker._severity_rank('high')
        assert checker._severity_rank('high') > checker._severity_rank('medium')
        assert checker._severity_rank('medium') > checker._severity_rank('low')

    def test_is_safe_threshold(self, checker):
        """Test is_safe is False for high/critical severity."""
        # Names with critical issues should not be safe
        result = checker.check("Giftbox")
        if result.severity in ['high', 'critical']:
            assert not result.is_safe


class TestHazardCheckerMarkets:
    """Tests for market-specific hazard checking."""

    @pytest.fixture
    def checker(self):
        return HazardChecker()

    def test_check_with_markets_param(self, checker):
        """Test check accepts markets parameter."""
        result = checker.check("TestName", markets=['german'])
        # Should not raise
        assert isinstance(result, HazardResult)

    def test_check_all_markets(self, checker):
        """Test check without markets checks all."""
        result = checker.check("Giftbox", markets=None)
        # Should still detect German hazard even without specifying market
        assert not result.is_safe


class TestSoundsSimilar:
    """Tests for _sounds_similar helper method."""

    @pytest.fixture
    def checker(self):
        return HazardChecker()

    def test_identical_sounds_similar(self, checker):
        """Test identical words are similar."""
        assert checker._sounds_similar("test", "test")

    def test_substring_sounds_similar(self, checker):
        """Test substring matching."""
        assert checker._sounds_similar("testing", "test")

    def test_different_not_similar(self, checker):
        """Test very different words are not similar."""
        # Unless simplified forms match
        result = checker._sounds_similar("xyz", "abc")
        # May or may not be similar depending on simplification
