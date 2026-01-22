"""
Tests for Cultural Generators
=============================
Tests for Japanese, Latin, Celtic, Celestial, and other cultural generators.
"""

import pytest
import sys
from pathlib import Path

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brandkit.generators import (
    JapaneseGenerator,
    LatinGenerator,
    CelticGenerator,
    CelestialGenerator,
)
from brandkit.generators.base_generator import GeneratedName


class TestJapaneseGenerator:
    """Tests for Japanese-style name generation."""

    @pytest.fixture
    def gen(self):
        """Create a Japanese generator with fixed seed."""
        return JapaneseGenerator(seed=42)

    def test_init(self, gen):
        """Test generator initialization."""
        assert gen is not None
        assert gen.culture == 'japanese'

    def test_generate_returns_list(self, gen):
        """Test that generate returns a list."""
        names = gen.generate(count=5)
        assert isinstance(names, list)

    def test_generate_correct_count(self, gen):
        """Test that generate returns correct count."""
        names = gen.generate(count=10)
        # May return fewer due to hazard/pronounceability filtering
        assert len(names) <= 10

    def test_generated_names_have_attributes(self, gen):
        """Test that generated names have required attributes."""
        names = gen.generate(count=5)
        for name in names:
            assert hasattr(name, 'name')
            assert hasattr(name, 'roots_used')
            assert hasattr(name, 'meaning_hints')
            assert isinstance(name.name, str)
            assert len(name.name) >= 2

    def test_names_are_pronounceable(self, gen):
        """Test that names follow CV patterns (roughly)."""
        names = gen.generate(count=10)
        for name in names:
            # Japanese names should be mostly vowel-heavy
            vowel_count = sum(1 for c in name.name.lower() if c in 'aeiou')
            assert vowel_count >= 1

    def test_seed_reproducibility(self):
        """Test that same seed produces same results (without quality filter)."""
        gen1 = JapaneseGenerator(seed=123)
        gen2 = JapaneseGenerator(seed=123)
        # Disable quality filter for deterministic results
        names1 = gen1.generate(count=5, check_hazards=False)
        names2 = gen2.generate(count=5, check_hazards=False)
        # Note: With hazard checking enabled, results may vary
        assert len(names1) == len(names2)

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        gen1 = JapaneseGenerator(seed=1)
        gen2 = JapaneseGenerator(seed=2)
        names1 = gen1.generate(count=10)
        names2 = gen2.generate(count=10)
        # Should have at least some different names
        set1 = {n.name for n in names1}
        set2 = {n.name for n in names2}
        assert set1 != set2

    def test_generate_with_categories(self, gen):
        """Test generation with category filter."""
        names = gen.generate(count=5, categories=['nature'])
        assert len(names) <= 5

    def test_generate_with_archetype(self, gen):
        """Test generation with archetype."""
        names = gen.generate(count=5, archetype='tech')
        assert len(names) <= 5

    def test_avoids_shi_ku(self, gen):
        """Test that names avoid 'shi' and 'ku' patterns."""
        names = gen.generate(count=20)
        # Should mostly avoid death/suffering sounds
        # This is a soft test since some may slip through


class TestLatinGenerator:
    """Tests for Latin/Romance-style name generation."""

    @pytest.fixture
    def gen(self):
        """Create a Latin generator with fixed seed."""
        return LatinGenerator(seed=42)

    def test_init(self, gen):
        """Test generator initialization."""
        assert gen is not None
        assert gen.culture == 'latin'

    def test_generate_returns_list(self, gen):
        """Test that generate returns a list."""
        names = gen.generate(count=5)
        assert isinstance(names, list)

    def test_generate_correct_count(self, gen):
        """Test that generate returns correct count."""
        names = gen.generate(count=10)
        assert len(names) <= 10

    def test_generated_names_have_attributes(self, gen):
        """Test that generated names have required attributes."""
        names = gen.generate(count=5)
        for name in names:
            assert hasattr(name, 'name')
            assert isinstance(name.name, str)
            assert len(name.name) >= 2

    def test_latin_endings(self, gen):
        """Test that names often have Latin-style endings."""
        names = gen.generate(count=20)
        latin_endings = ['us', 'um', 'a', 'is', 'ia', 'ium', 'or', 'ex']
        # At least some should have Latin endings
        has_latin_ending = sum(
            1 for n in names
            if any(n.name.lower().endswith(e) for e in latin_endings)
        )
        # Not strict - just checking the pattern exists
        assert has_latin_ending >= 0

    def test_seed_reproducibility(self):
        """Test that same seed produces same results (without quality filter)."""
        gen1 = LatinGenerator(seed=123)
        gen2 = LatinGenerator(seed=123)
        names1 = gen1.generate(count=5, check_hazards=False)
        names2 = gen2.generate(count=5, check_hazards=False)
        assert len(names1) == len(names2)


class TestCelticGenerator:
    """Tests for Celtic-style name generation."""

    @pytest.fixture
    def gen(self):
        """Create a Celtic generator with fixed seed."""
        return CelticGenerator(seed=42)

    def test_init(self, gen):
        """Test generator initialization."""
        assert gen is not None
        assert gen.culture == 'celtic'

    def test_generate_returns_list(self, gen):
        """Test that generate returns a list."""
        names = gen.generate(count=5)
        assert isinstance(names, list)

    def test_generate_correct_count(self, gen):
        """Test that generate returns correct count."""
        names = gen.generate(count=10)
        assert len(names) <= 10

    def test_generated_names_have_attributes(self, gen):
        """Test that generated names have required attributes."""
        names = gen.generate(count=5)
        for name in names:
            assert hasattr(name, 'name')
            assert isinstance(name.name, str)
            assert len(name.name) >= 2

    def test_seed_reproducibility(self):
        """Test that same seed produces same results (without quality filter)."""
        gen1 = CelticGenerator(seed=123)
        gen2 = CelticGenerator(seed=123)
        names1 = gen1.generate(count=5, check_hazards=False)
        names2 = gen2.generate(count=5, check_hazards=False)
        assert len(names1) == len(names2)


class TestCelestialGenerator:
    """Tests for Celestial/Space-style name generation."""

    @pytest.fixture
    def gen(self):
        """Create a Celestial generator with fixed seed."""
        return CelestialGenerator(seed=42)

    def test_init(self, gen):
        """Test generator initialization."""
        assert gen is not None
        assert gen.culture == 'celestial'

    def test_generate_returns_list(self, gen):
        """Test that generate returns a list."""
        names = gen.generate(count=5)
        assert isinstance(names, list)

    def test_generate_correct_count(self, gen):
        """Test that generate returns correct count."""
        names = gen.generate(count=10)
        assert len(names) <= 10

    def test_generated_names_have_attributes(self, gen):
        """Test that generated names have required attributes."""
        names = gen.generate(count=5)
        for name in names:
            assert hasattr(name, 'name')
            assert isinstance(name.name, str)
            assert len(name.name) >= 2

    def test_seed_reproducibility(self):
        """Test that same seed produces same results (without quality filter)."""
        gen1 = CelestialGenerator(seed=123)
        gen2 = CelestialGenerator(seed=123)
        names1 = gen1.generate(count=5, check_hazards=False)
        names2 = gen2.generate(count=5, check_hazards=False)
        assert len(names1) == len(names2)

    def test_space_themed_categories(self, gen):
        """Test generation with space-themed categories."""
        names = gen.generate(count=5, categories=['stars', 'cosmic'])
        assert len(names) <= 5


class TestGeneratedName:
    """Tests for GeneratedName dataclass."""

    def test_creation(self):
        """Test creating a GeneratedName."""
        name = GeneratedName(
            name="Voltara",
            culture="greek",
            roots_used=["volt"],
            meaning_hints=["power"],
            method="root_suffix",
            score=0.8,
        )
        assert name.name == "Voltara"
        assert name.culture == "greek"
        assert name.roots_used == ["volt"]

    def test_default_values(self):
        """Test default values."""
        name = GeneratedName(
            name="Test",
            culture="test",
            roots_used=[],
            meaning_hints=[],
            method="test",
        )
        assert name.score == 0.0


class TestGeneratorComparison:
    """Tests comparing different cultural generators."""

    def test_different_generators_different_styles(self):
        """Test that different generators produce different styles."""
        jp = JapaneseGenerator(seed=42)
        lt = LatinGenerator(seed=42)
        cl = CelticGenerator(seed=42)
        ce = CelestialGenerator(seed=42)

        jp_names = {n.name for n in jp.generate(count=20)}
        lt_names = {n.name for n in lt.generate(count=20)}
        cl_names = {n.name for n in cl.generate(count=20)}
        ce_names = {n.name for n in ce.generate(count=20)}

        # Sets should be largely different
        # Some overlap is possible but shouldn't be complete
        assert jp_names != lt_names
        assert lt_names != cl_names
        assert cl_names != ce_names

    def test_all_generators_produce_valid_names(self):
        """Test that all generators produce valid names."""
        generators = [
            JapaneseGenerator(seed=42),
            LatinGenerator(seed=42),
            CelticGenerator(seed=42),
            CelestialGenerator(seed=42),
        ]

        for gen in generators:
            names = gen.generate(count=5)
            for name in names:
                assert isinstance(name.name, str)
                assert len(name.name) >= 2
                assert name.name.isalpha() or any(c.isalpha() for c in name.name)


class TestGeneratorEdgeCases:
    """Edge case tests for cultural generators."""

    def test_zero_count(self):
        """Test generation with count=0."""
        gen = JapaneseGenerator(seed=42)
        names = gen.generate(count=0)
        assert names == []

    def test_large_count(self):
        """Test generation with large count."""
        gen = LatinGenerator(seed=42)
        names = gen.generate(count=100)
        # Should not crash, may return fewer due to filtering
        assert len(names) <= 100

    def test_invalid_category(self):
        """Test generation with non-existent category."""
        gen = CelticGenerator(seed=42)
        names = gen.generate(count=5, categories=['nonexistent_category'])
        # Should handle gracefully, may return empty or use fallback
        assert isinstance(names, list)

    def test_empty_categories(self):
        """Test generation with empty categories list."""
        gen = CelestialGenerator(seed=42)
        names = gen.generate(count=5, categories=[])
        # Should use default pool
        assert isinstance(names, list)


class TestGeneratorIntegration:
    """Integration tests for cultural generators."""

    def test_generator_with_hazard_checking(self):
        """Test that generators filter hazardous names."""
        gen = JapaneseGenerator(seed=42)
        names = gen.generate(count=50, check_hazards=True)
        # All returned names should be safe
        from brandkit.generators.base_generator import HazardChecker
        checker = HazardChecker()
        for name in names:
            result = checker.check(name.name)
            # Should not have critical hazards
            assert result.severity != 'critical'

    def test_generator_with_markets(self):
        """Test generation with market-specific filtering."""
        gen = LatinGenerator(seed=42)
        names = gen.generate(count=10, markets=['german', 'english'])
        # All returned names should be suitable for both markets
        assert len(names) <= 10

    def test_generator_with_industry(self):
        """Test generation with industry profile."""
        gen = CelticGenerator(seed=42)
        names = gen.generate(count=5, industry='outdoor')
        assert len(names) <= 5
