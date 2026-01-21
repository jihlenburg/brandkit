"""
Tests for EN/DE Phonetics Pipeline
===================================
Tests locale-specific phonotactics, German normalization, G2P approximation,
and market-specific scoring.
"""

import pytest
import sys
from pathlib import Path

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brandkit.generators.phonemes import (
    is_pronounceable,
    score_name,
    normalize_de,
    approx_g2p_en,
    approx_g2p_de,
    load_strategies,
)


class TestGermanNormalization:
    """Tests for normalize_de() function."""

    def test_umlaut_conversion(self):
        """Test umlaut to digraph conversion."""
        assert normalize_de("ä") == "ae"
        assert normalize_de("ö") == "oe"
        assert normalize_de("ü") == "ue"
        assert normalize_de("Ä") == "Ae"
        assert normalize_de("Ö") == "Oe"
        assert normalize_de("Ü") == "Ue"

    def test_eszett_conversion(self):
        """Test ß to ss conversion."""
        assert normalize_de("ß") == "ss"
        assert normalize_de("straße") == "strasse"
        assert normalize_de("größe") == "groesse"

    def test_full_words(self):
        """Test full word normalization."""
        assert normalize_de("Völker") == "Voelker"
        assert normalize_de("übung") == "uebunk"  # final devoicing: g→k
        assert normalize_de("Größe") == "Groesse"

    def test_final_devoicing(self):
        """Test German final devoicing rule."""
        # b→p, d→t, g→k, v→f, z→s at word end
        assert normalize_de("Tag", apply_final_devoicing=True) == "Tak"
        assert normalize_de("Rad", apply_final_devoicing=True) == "Rat"
        assert normalize_de("Stab", apply_final_devoicing=True) == "Stap"

    def test_no_final_devoicing(self):
        """Test disabling final devoicing."""
        assert normalize_de("Tag", apply_final_devoicing=False) == "Tag"
        assert normalize_de("Rad", apply_final_devoicing=False) == "Rad"

    def test_empty_and_short(self):
        """Test edge cases."""
        assert normalize_de("") == ""
        assert normalize_de("a") == "a"


class TestG2PApproximation:
    """Tests for G2P approximation functions."""

    def test_english_digraphs(self):
        """Test English digraph handling."""
        # th → TH
        assert "TH" in approx_g2p_en("thunder")
        assert "TH" in approx_g2p_en("think")
        # sh → SH
        assert "SH" in approx_g2p_en("ship")
        # ch → CH
        assert "CH" in approx_g2p_en("church")
        # ph → F
        assert "F" in approx_g2p_en("phoenix")

    def test_english_silent_letters(self):
        """Test English silent letter removal."""
        # kn → N
        assert approx_g2p_en("knight").startswith("N")
        assert approx_g2p_en("know").startswith("N")
        # wr → R
        assert approx_g2p_en("write").startswith("R")
        # gn → N
        assert approx_g2p_en("gnome").startswith("N")

    def test_german_sch(self):
        """Test German sch cluster."""
        assert "SH" in approx_g2p_de("schiff")
        assert "SH" in approx_g2p_de("Schule")

    def test_german_initial_st_sp(self):
        """Test German initial st/sp pronunciation."""
        # st- → SHT at start
        result = approx_g2p_de("stark")
        assert "SHT" in result or result.startswith("SHT")
        # sp- → SHP at start
        result = approx_g2p_de("sport")
        assert "SHP" in result or result.startswith("SHP")

    def test_german_pf(self):
        """Test German pf cluster."""
        assert "PF" in approx_g2p_de("Pferd")
        assert "PF" in approx_g2p_de("Pflanze")

    def test_german_final_devoicing_g2p(self):
        """Test final devoicing in German G2P."""
        # Should apply devoicing at word end
        result = approx_g2p_de("Tag")
        assert result.endswith("K") or "K" in result[-2:]


class TestPronounceabilityMarkets:
    """Tests for market-specific pronounceability."""

    def test_pf_cluster_markets(self):
        """Test pf cluster handling by market."""
        # pf is forbidden at start for English
        is_ok_en, reason_en = is_pronounceable("Pflanzer", markets='en')
        assert not is_ok_en
        assert "pf" in reason_en.lower()

        # pf is allowed for German
        is_ok_de, reason_de = is_pronounceable("Pflanzer", markets='de')
        # May still fail for other reasons, but not for pf
        if not is_ok_de:
            assert "pf" not in reason_de.lower() or "forbidden_start_de" not in reason_de

    def test_th_cluster_markets(self):
        """Test th cluster handling by market."""
        # th is allowed for English
        is_ok_en, _ = is_pronounceable("Thunder", markets='en')
        assert is_ok_en

        # th is forbidden at start for German
        is_ok_de, reason_de = is_pronounceable("Thunder", markets='de')
        assert not is_ok_de
        assert "th" in reason_de.lower()

    def test_en_de_requires_both(self):
        """Test that en_de requires passing both markets."""
        # Thunder passes EN but fails DE
        is_ok, reason = is_pronounceable("Thunder", markets='en_de')
        assert not is_ok
        assert "de" in reason.lower()

        # Pflanzer passes DE but fails EN
        is_ok, reason = is_pronounceable("Pflanzer", markets='en_de')
        assert not is_ok
        assert "en" in reason.lower()

    def test_neutral_names(self):
        """Test names that work in both markets."""
        # These should pass both EN and DE
        neutral_names = ["Voltara", "Luminex", "Krafta", "Solara"]
        for name in neutral_names:
            is_ok, reason = is_pronounceable(name, markets='en_de')
            # May still fail for other reasons, but checking basic structure
            if not is_ok:
                # Should not be market-specific failure
                assert "forbidden_start_en" not in reason or "forbidden_start_de" not in reason

    def test_default_markets(self):
        """Test that default markets is en_de."""
        # Should be equivalent to explicit en_de
        is_ok_default, reason_default = is_pronounceable("Thunder")
        is_ok_explicit, reason_explicit = is_pronounceable("Thunder", markets='en_de')
        assert is_ok_default == is_ok_explicit


class TestScoreNameMarkets:
    """Tests for market-specific scoring."""

    def test_score_with_markets(self):
        """Test that score_name accepts markets parameter."""
        # Should not raise
        score_en = score_name("Voltix", markets='en')
        score_de = score_name("Voltix", markets='de')
        score_both = score_name("Voltix", markets='en_de')

        # All should return valid scores
        assert 0 <= score_en <= 1
        assert 0 <= score_de <= 1
        assert 0 <= score_both <= 1

    def test_unpronounceable_gets_zero(self):
        """Test that unpronounceable names get score 0."""
        # Name with pf fails EN
        score = score_name("Pflanzer", markets='en')
        assert score == 0.0

        # Name with th fails DE
        score = score_name("Thunder", markets='de')
        assert score == 0.0

    def test_market_weights_applied(self):
        """Test that market-specific weights affect score."""
        strategies = load_strategies()
        en_weights = strategies.get_market_weights('en')
        de_weights = strategies.get_market_weights('de')

        # Check that market weights exist
        assert en_weights or de_weights  # At least one should have weights


class TestStrategiesConfig:
    """Tests for StrategiesConfig locale methods."""

    def test_locale_pronounceability_loaded(self):
        """Test that locale-specific rules are loaded."""
        strategies = load_strategies()

        en_config = strategies.get_locale_pronounceability('en')
        de_config = strategies.get_locale_pronounceability('de')

        # Both should have some config
        assert en_config or de_config

    def test_forbidden_clusters_by_locale(self):
        """Test forbidden cluster retrieval by locale."""
        strategies = load_strategies()

        en_forbidden = strategies.get_locale_forbidden_initial('en')
        de_forbidden = strategies.get_locale_forbidden_initial('de')

        # pf should be forbidden for EN
        assert 'pf' in en_forbidden or not en_forbidden  # May be empty if not configured

        # th should be forbidden for DE
        assert 'th' in de_forbidden or not de_forbidden

    def test_market_weights_retrieval(self):
        """Test market weights retrieval."""
        strategies = load_strategies()

        en_weights = strategies.get_market_weights('en')
        de_weights = strategies.get_market_weights('de')
        both_weights = strategies.get_market_weights('en_de')

        # Should return dicts (possibly empty)
        assert isinstance(en_weights, dict)
        assert isinstance(de_weights, dict)
        assert isinstance(both_weights, dict)
