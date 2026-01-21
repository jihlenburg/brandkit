#!/usr/bin/env python3
"""
Cultural Generators
===================
Concrete implementations of CulturalGenerator for various cultures:
- Japanese
- Latin/Romance
- Celtic
- Celestial (Space/Astronomy)
"""

from typing import List, Optional, Tuple

from .base_generator import CulturalGenerator
from .entropy import get_rng


# =============================================================================
# Japanese Generator
# =============================================================================

class JapaneseGenerator(CulturalGenerator):
    """
    Generate brand names with Japanese-inspired phonetics.

    Features:
    - Strict CV (consonant-vowel) syllable structure
    - Avoids shi (death) and ku (suffering)
    - Clean, minimalist sound
    - Tech-friendly aesthetics

    Great for: Tech, gaming, minimalist design, precision products
    """

    def __init__(self, seed: int = None):
        super().__init__('japanese', seed)
        # All phonetics loaded from japanese.yaml

    def _generate_one(self,
                      categories: List[str],
                      archetype: str,
                      industry: str) -> Optional[Tuple[str, List[str], List[str], str]]:
        """Generate a single Japanese-style name."""
        method = self._rng.choice([
            self._root_suffix,
            self._root_only,
            self._pattern_based,
            self._compound,
        ])
        return method(categories)

    def _root_suffix(self, categories: List[str]) -> Optional[Tuple]:
        """Root + Japanese suffix pattern."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)
        suffixes = self._get_suffixes('japanese', 'tech', 'neutral')
        suffix = self._rng.choice(suffixes)

        # Japanese style: often drop trailing vowel before adding suffix
        vowels = self._get_vowels()
        base = root
        if base[-1] in vowels and suffix[0] in vowels:
            base = base[:-1]

        name = base + suffix
        return (name, [root], [meaning], 'root_suffix')

    def _root_only(self, categories: List[str]) -> Optional[Tuple]:
        """Use root directly with possible vowel ending."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)
        vowels = self._get_vowels()

        # Japanese names typically end in vowels or 'n'
        if root[-1] not in vowels and root[-1] != 'n':
            root = root + self._rng.choice(vowels[:3])  # a, i, o

        return (root, [root], [meaning], 'root_only')

    def _pattern_based(self, categories: List[str]) -> Optional[Tuple]:
        """Generate from CV patterns."""
        patterns = [
            'CVCV',      # yama, kawa
            'CVCCV',     # sakura
            'VCVCV',     # akira
            'CVCVCV',    # toyota
        ]
        pattern = self._rng.choice(patterns)
        consonants = self._get_consonants()
        vowels = self._get_vowels()

        result = []
        for char in pattern:
            if char == 'C':
                result.append(self._rng.choice(consonants))
            elif char == 'V':
                result.append(self._rng.choice(vowels))

        name = ''.join(result)
        return (name, [], ['pattern-generated'], 'pattern')

    def _compound(self, categories: List[str]) -> Optional[Tuple]:
        """Combine two roots Japanese-style."""
        pool = self._get_pool(categories)
        if len(pool) < 2:
            return None

        root1, meaning1, _ = self._rng.choice(pool)
        root2, meaning2, _ = self._rng.choice(pool)

        # Take first part of root1 and second part of root2
        part1 = root1[:2] if len(root1) >= 2 else root1
        part2 = root2[-2:] if len(root2) >= 2 else root2

        name = part1 + part2
        vowels = self._get_vowels()

        # Ensure ends in vowel or 'n'
        if name[-1] not in vowels and name[-1] != 'n':
            name = name + 'a'

        return (name, [root1, root2], [meaning1, meaning2], 'compound')


# =============================================================================
# Latin Generator
# =============================================================================

class LatinGenerator(CulturalGenerator):
    """
    Generate brand names with Latin/Romance influences.

    Features:
    - Classical Latin endings (-us, -um, -a, -is)
    - Italian/French Romance flourishes
    - Elegant, premium feel
    - Avoids harsh Germanic clusters

    Great for: Luxury, fashion, cosmetics, wine, pharma
    """

    def __init__(self, seed: int = None):
        super().__init__('latin', seed)
        # All phonetics loaded from latin.yaml

    def _generate_one(self,
                      categories: List[str],
                      archetype: str,
                      industry: str) -> Optional[Tuple[str, List[str], List[str], str]]:
        """Generate a single Latin-style name."""
        method = self._rng.choice([
            self._classical_latin,
            self._italian_style,
            self._french_touch,
            self._romance_blend,
        ])
        return method(categories)

    def _classical_latin(self, categories: List[str]) -> Optional[Tuple]:
        """Root + classical Latin suffix."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)
        suffixes = self._get_suffixes('latin', 'modern')
        suffix = self._rng.choice(suffixes)

        # Handle root-suffix junction
        connector = self._get_connector(root[-1], suffix[0])
        name = root + connector + suffix

        return (name, [root], [meaning], 'classical_latin')

    def _italian_style(self, categories: List[str]) -> Optional[Tuple]:
        """Italian-influenced name (ending in -o, -a, -i)."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)
        suffixes = self._get_suffixes('italian')
        suffix = self._rng.choice(suffixes)

        # Italian style: soften consonant endings
        base = root
        if base[-1] in 'kg':
            base = base[:-1] + 'c'

        connector = self._get_connector(base[-1], suffix[0])
        name = base + connector + suffix

        return (name, [root], [meaning], 'italian_style')

    def _french_touch(self, categories: List[str]) -> Optional[Tuple]:
        """French-influenced name."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)
        suffixes = self._get_suffixes('french')
        suffix = self._rng.choice(suffixes)

        connector = self._get_connector(root[-1], suffix[0])
        name = root + connector + suffix

        return (name, [root], [meaning], 'french_touch')

    def _romance_blend(self, categories: List[str]) -> Optional[Tuple]:
        """Blend two roots Romance-style."""
        pool = self._get_pool(categories)
        if len(pool) < 2:
            return None

        root1, meaning1, _ = self._rng.choice(pool)
        root2, meaning2, _ = self._rng.choice(pool)

        # Take beginning of first, ending of second
        part1 = root1[:3] if len(root1) >= 3 else root1
        part2 = root2[-3:] if len(root2) >= 3 else root2

        # Add connector if needed
        connector = self._get_connector(part1[-1], part2[0])
        name = part1 + connector + part2

        return (name, [root1, root2], [meaning1, meaning2], 'romance_blend')


# =============================================================================
# Celtic Generator
# =============================================================================

class CelticGenerator(CulturalGenerator):
    """
    Generate brand names with Celtic (Irish, Welsh, Scottish) influences.

    Features:
    - Nature-based vocabulary
    - Mystical/mythological roots
    - Anglicized spellings for pronounceability
    - Strong consonant clusters softened

    Great for: Nature brands, craft products, heritage brands, mystical products
    """

    def __init__(self, seed: int = None):
        super().__init__('celtic', seed)
        # All phonetics loaded from celtic.yaml

    def _generate_one(self,
                      categories: List[str],
                      archetype: str,
                      industry: str) -> Optional[Tuple[str, List[str], List[str], str]]:
        """Generate a single Celtic-style name."""
        method = self._rng.choice([
            self._root_suffix,
            self._nature_compound,
            self._welsh_style,
            self._gaelic_style,
        ])
        return method(categories)

    def _root_suffix(self, categories: List[str]) -> Optional[Tuple]:
        """Root + Celtic suffix."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)
        suffixes = self._get_suffixes('celtic', 'gaelic', 'neutral')
        suffix = self._rng.choice(suffixes)

        connector = self._get_connector(root[-1], suffix[0])
        name = root + connector + suffix

        return (name, [root], [meaning], 'root_suffix')

    def _nature_compound(self, categories: List[str]) -> Optional[Tuple]:
        """Combine two nature roots."""
        # Prefer nature-related categories
        nature_cats = ['nature', 'elements', 'animals', 'welsh']
        pool = self._get_pool(nature_cats)
        if not pool:
            pool = self._get_pool(categories)
        if len(pool) < 2:
            return None

        root1, meaning1, _ = self._rng.choice(pool)
        root2, meaning2, _ = self._rng.choice(pool)

        # Celtic compound: often first + partial second
        part1 = root1
        part2 = root2[:3] if len(root2) >= 3 else root2

        connector = self._get_connector(part1[-1], part2[0])
        name = part1 + connector + part2

        return (name, [root1, root2], [meaning1, meaning2], 'nature_compound')

    def _welsh_style(self, categories: List[str]) -> Optional[Tuple]:
        """Welsh-influenced name."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)

        # Welsh suffixes from YAML
        suffixes = self._get_suffixes('welsh', 'celtic')
        suffix = self._rng.choice(suffixes)

        connector = self._get_connector(root[-1], suffix[0])
        name = root + connector + suffix

        return (name, [root], [meaning], 'welsh_style')

    def _gaelic_style(self, categories: List[str]) -> Optional[Tuple]:
        """Scottish/Irish Gaelic-influenced name."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)

        # Gaelic prefixes and suffixes from YAML
        use_prefix = self._rng.choice([True, False])
        if use_prefix:
            prefixes = self._config.get('prefixes', {}).get('gaelic', [])
            prefix = self._rng.choice(prefixes) if prefixes else ''
            name = prefix + root if prefix else root
        else:
            suffixes = self._get_suffixes('gaelic', 'celtic')
            suffix = self._rng.choice(suffixes)
            connector = self._get_connector(root[-1], suffix[0])
            name = root + connector + suffix

        return (name, [root], [meaning], 'gaelic_style')


# =============================================================================
# Celestial Generator
# =============================================================================

class CelestialGenerator(CulturalGenerator):
    """
    Generate brand names with space/astronomy influences.

    Features:
    - Stars, planets, galaxies vocabulary
    - Space exploration terminology
    - Scientific yet evocative sound
    - Latin/Greek astronomical heritage

    Great for: Tech, aerospace, energy, premium brands, futuristic products
    """

    def __init__(self, seed: int = None):
        super().__init__('celestial', seed)
        # All phonetics loaded from celestial.yaml

    def _generate_one(self,
                      categories: List[str],
                      archetype: str,
                      industry: str) -> Optional[Tuple[str, List[str], List[str], str]]:
        """Generate a single celestial-style name."""
        method = self._rng.choice([
            self._root_suffix,
            self._prefix_root,
            self._stellar_compound,
            self._cosmic_blend,
        ])
        return method(categories)

    def _root_suffix(self, categories: List[str]) -> Optional[Tuple]:
        """Root + celestial suffix."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)
        suffixes = self._get_suffixes('celestial', 'tech', 'science')
        suffix = self._rng.choice(suffixes) if suffixes else 'a'

        connector = self._get_connector(root[-1], suffix[0])
        name = root + connector + suffix

        return (name, [root], [meaning], 'root_suffix')

    def _prefix_root(self, categories: List[str]) -> Optional[Tuple]:
        """Celestial prefix + root."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)
        prefixes = self._get_prefixes()
        prefix = self._rng.choice(prefixes) if prefixes else self._rng.choice(['astro', 'cosmo', 'stellar', 'nova', 'neo', ''])

        if prefix:
            connector = self._get_connector(prefix[-1], root[0])
            name = prefix + connector + root
        else:
            name = root

        return (name, [root], [meaning], 'prefix_root')

    def _stellar_compound(self, categories: List[str]) -> Optional[Tuple]:
        """Compound two celestial roots."""
        pool = self._get_pool(categories)
        if len(pool) < 2:
            return None

        root1, meaning1, _ = self._rng.choice(pool)
        root2, meaning2, _ = self._rng.choice(pool)

        # Stellar compounds: take prefix of first, suffix of second
        part1 = root1[:3] if len(root1) >= 3 else root1
        part2 = root2[-3:] if len(root2) >= 3 else root2

        connector = self._get_connector(part1[-1], part2[0])
        name = part1 + connector + part2

        return (name, [root1, root2], [meaning1, meaning2], 'stellar_compound')

    def _cosmic_blend(self, categories: List[str]) -> Optional[Tuple]:
        """Cosmic-themed blend (space exploration style)."""
        # Prefer space-related categories
        space_cats = ['stars', 'planets', 'galaxies', 'light', 'exploration']
        pool = self._get_pool(space_cats)
        if not pool:
            pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)

        # Cosmic brand suffixes from YAML
        suffixes = self._get_suffixes('cosmic', 'celestial', 'tech')
        suffix = self._rng.choice(suffixes)

        # Optional tech/space prefix from YAML
        if self._rng.choice([True, False]):
            cosmic_prefixes = self._config.get('prefixes', {}).get('cosmic', [])
            prefix = self._rng.choice(cosmic_prefixes) if cosmic_prefixes else ''
            if prefix:
                connector = self._get_connector(prefix[-1], root[0])
                root = prefix + connector + root

        connector = self._get_connector(root[-1], suffix[0])
        name = root + connector + suffix

        return (name, [root], [meaning], 'cosmic_blend')


# =============================================================================
# English Animals Generator
# =============================================================================

class AnimalsGenerator(CulturalGenerator):
    """
    Generate brand names from global animal kingdom.

    Features:
    - Big cats, raptors, marine predators, canines
    - Speed, power, intelligence archetypes
    - Automotive/sports brand patterns (Jaguar, Mustang, Cobra)
    - Strong phonetic properties

    Great for: Automotive, sports, tech, outdoor, energy brands
    """

    def __init__(self, seed: int = None):
        super().__init__('animals', seed)
        # All phonetics loaded from animals.yaml

    def _generate_one(self,
                      categories: List[str],
                      archetype: str,
                      industry: str) -> Optional[Tuple[str, List[str], List[str], str]]:
        """Generate a single English animals-style name."""
        method = self._rng.choice([
            self._root_suffix,
            self._prefix_root,
            self._animal_compound,
            self._archetype_blend,
        ])
        return method(categories)

    def _root_suffix(self, categories: List[str]) -> Optional[Tuple]:
        """Animal root + power suffix."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)
        suffixes = self._get_suffixes('animal', 'power', 'neutral')
        suffix = self._rng.choice(suffixes)

        connector = self._get_connector(root[-1], suffix[0])
        name = root + connector + suffix

        return (name, [root], [meaning], 'root_suffix')

    def _prefix_root(self, categories: List[str]) -> Optional[Tuple]:
        """Power/color prefix + animal root."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)
        # Use prefixes from YAML config (includes color tiers)
        prefixes = self._get_prefixes()
        prefix = self._rng.choice(prefixes) if prefixes else ''

        if prefix:
            name = prefix + root
        else:
            name = root

        return (name, [root], [meaning], 'prefix_root')

    def _animal_compound(self, categories: List[str]) -> Optional[Tuple]:
        """Compound two animal roots."""
        pool = self._get_pool(categories)
        if len(pool) < 2:
            return None

        root1, meaning1, _ = self._rng.choice(pool)
        root2, meaning2, _ = self._rng.choice(pool)

        # Take abbreviated forms
        part1 = root1[:3] if len(root1) >= 3 else root1
        part2 = root2[-3:] if len(root2) >= 3 else root2

        connector = self._get_connector(part1[-1], part2[0])
        name = part1 + connector + part2

        return (name, [root1, root2], [meaning1, meaning2], 'animal_compound')

    def _archetype_blend(self, categories: List[str]) -> Optional[Tuple]:
        """Animal + quality archetype blend."""
        # Power animal categories
        animal_cats = ['big_cats', 'raptors', 'canines', 'marine', 'power_mammals']
        quality_cats = ['qualities']

        animal_pool = self._get_pool(animal_cats)
        quality_pool = self._get_pool(quality_cats)

        if not animal_pool:
            return self._root_suffix(categories)

        animal, animal_meaning, _ = self._rng.choice(animal_pool)

        if quality_pool and self._rng.choice([True, False]):
            quality, quality_meaning, _ = self._rng.choice(quality_pool)
            # Quality + animal blend
            name = quality[:4] + animal[:4] if len(quality) >= 4 else quality + animal[:4]
            return (name, [quality, animal], [quality_meaning, animal_meaning], 'archetype_blend')
        else:
            # Just animal with suffix
            suffixes = self._get_suffixes('animal', 'power', 'neutral')
            suffix = self._rng.choice(suffixes)
            connector = self._get_connector(animal[-1], suffix[0])
            name = animal + connector + suffix
            return (name, [animal], [animal_meaning], 'archetype_blend')


# =============================================================================
# Mythology Generator
# =============================================================================

class MythologyGenerator(CulturalGenerator):
    """
    Generate brand names from modern and popular mythology.

    Features:
    - Creatures from global folklore (banshee, phoenix, kitsune)
    - Fantasy and urban legend references
    - Mystical and magical feel
    - Gaming and entertainment appeal

    Great for: Gaming, entertainment, fantasy brands, creative products
    """

    def __init__(self, seed: int = None):
        super().__init__('mythology', seed)
        # All phonetics loaded from mythology.yaml

    def _generate_one(self,
                      categories: List[str],
                      archetype: str,
                      industry: str) -> Optional[Tuple[str, List[str], List[str], str]]:
        """Generate a single mythology-style name."""
        method = self._rng.choice([
            self._root_suffix,
            self._prefix_root,
            self._creature_compound,
            self._mythic_blend,
        ])
        return method(categories)

    def _root_suffix(self, categories: List[str]) -> Optional[Tuple]:
        """Mythic root + suffix."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)
        suffixes = self._get_suffixes('mythic', 'neutral')
        suffix = self._rng.choice(suffixes)

        connector = self._get_connector(root[-1], suffix[0])
        name = root + connector + suffix

        return (name, [root], [meaning], 'root_suffix')

    def _prefix_root(self, categories: List[str]) -> Optional[Tuple]:
        """Mythic prefix + creature root."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)
        # Use prefixes from YAML config
        prefixes = self._get_prefixes()
        prefix = self._rng.choice(prefixes) if prefixes else ''

        if prefix:
            name = prefix + root
        else:
            name = root

        return (name, [root], [meaning], 'prefix_root')

    def _creature_compound(self, categories: List[str]) -> Optional[Tuple]:
        """Compound two mythic roots."""
        pool = self._get_pool(categories)
        if len(pool) < 2:
            return None

        root1, meaning1, _ = self._rng.choice(pool)
        root2, meaning2, _ = self._rng.choice(pool)

        # Take abbreviated forms
        part1 = root1[:3] if len(root1) >= 3 else root1
        part2 = root2[-3:] if len(root2) >= 3 else root2

        connector = self._get_connector(part1[-1], part2[0])
        name = part1 + connector + part2

        return (name, [root1, root2], [meaning1, meaning2], 'creature_compound')

    def _mythic_blend(self, categories: List[str]) -> Optional[Tuple]:
        """Mythic creature + quality blend."""
        # Creature categories
        creature_cats = ['celtic', 'nordic', 'greek', 'japanese', 'slavic', 'fae', 'dragons', 'undead']
        quality_cats = ['qualities']

        creature_pool = self._get_pool(creature_cats)
        quality_pool = self._get_pool(quality_cats)

        if not creature_pool:
            return self._root_suffix(categories)

        creature, creature_meaning, _ = self._rng.choice(creature_pool)

        if quality_pool and self._rng.choice([True, False]):
            quality, quality_meaning, _ = self._rng.choice(quality_pool)
            # Quality prefix
            name = quality[:4] + creature[:4] if len(quality) >= 4 else quality + creature[:4]
            return (name, [quality, creature], [quality_meaning, creature_meaning], 'mythic_blend')
        else:
            # Just creature with suffix
            suffixes = self._get_suffixes('mythic', 'neutral')
            suffix = self._rng.choice(suffixes)
            connector = self._get_connector(creature[-1], suffix[0])
            name = creature + connector + suffix
            return (name, [creature], [creature_meaning], 'mythic_blend')


# =============================================================================
# Landmarks Generator
# =============================================================================

class LandmarksGenerator(CulturalGenerator):
    """
    Generate brand names from famous landmarks and natural wonders.

    Features:
    - US National Parks and mountains
    - International landmarks and reserves
    - Natural features (canyons, waterfalls, etc.)
    - Aspirational and adventurous feel

    Great for: Travel, adventure, outdoor, premium brands
    """

    def __init__(self, seed: int = None):
        super().__init__('landmarks', seed)
        # All phonetics loaded from landmarks.yaml

    def _generate_one(self,
                      categories: List[str],
                      archetype: str,
                      industry: str) -> Optional[Tuple[str, List[str], List[str], str]]:
        """Generate a single landmarks-style name."""
        method = self._rng.choice([
            self._root_suffix,
            self._prefix_root,
            self._landmark_compound,
            self._geographic_blend,
        ])
        return method(categories)

    def _root_suffix(self, categories: List[str]) -> Optional[Tuple]:
        """Landmark root + suffix."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)
        suffixes = self._get_suffixes('geographic', 'modern', 'neutral')
        suffix = self._rng.choice(suffixes)

        connector = self._get_connector(root[-1], suffix[0])
        name = root + connector + suffix

        return (name, [root], [meaning], 'root_suffix')

    def _prefix_root(self, categories: List[str]) -> Optional[Tuple]:
        """Direction/color prefix + landmark root."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)
        # Use prefixes from YAML config (includes color tiers and geographic)
        prefixes = self._get_prefixes()
        prefix = self._rng.choice(prefixes) if prefixes else ''

        if prefix:
            name = prefix + root
        else:
            name = root

        return (name, [root], [meaning], 'prefix_root')

    def _landmark_compound(self, categories: List[str]) -> Optional[Tuple]:
        """Compound two landmark roots."""
        pool = self._get_pool(categories)
        if len(pool) < 2:
            return None

        root1, meaning1, _ = self._rng.choice(pool)
        root2, meaning2, _ = self._rng.choice(pool)

        # Take abbreviated forms
        part1 = root1[:3] if len(root1) >= 3 else root1
        part2 = root2[-3:] if len(root2) >= 3 else root2

        connector = self._get_connector(part1[-1], part2[0])
        name = part1 + connector + part2

        return (name, [root1, root2], [meaning1, meaning2], 'landmark_compound')

    def _geographic_blend(self, categories: List[str]) -> Optional[Tuple]:
        """Geographic feature + quality blend."""
        # Feature categories
        feature_cats = ['us_parks', 'mountains', 'canyons', 'water', 'islands']
        quality_cats = ['qualities']

        feature_pool = self._get_pool(feature_cats)
        quality_pool = self._get_pool(quality_cats)

        if not feature_pool:
            return self._root_suffix(categories)

        feature, feature_meaning, _ = self._rng.choice(feature_pool)

        if quality_pool and self._rng.choice([True, False]):
            quality, quality_meaning, _ = self._rng.choice(quality_pool)
            # Quality prefix + shortened feature
            name = quality[:4] + feature[:4] if len(quality) >= 4 else quality + feature[:4]
            return (name, [quality, feature], [quality_meaning, feature_meaning], 'geographic_blend')
        else:
            # Just feature with suffix
            suffixes = self._get_suffixes('geographic', 'modern', 'neutral')
            suffix = self._rng.choice(suffixes)
            connector = self._get_connector(feature[-1], suffix[0])
            name = feature + connector + suffix
            return (name, [feature], [feature_meaning], 'geographic_blend')


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'JapaneseGenerator',
    'LatinGenerator',
    'CelticGenerator',
    'CelestialGenerator',
    'AnimalsGenerator',
    'MythologyGenerator',
    'LandmarksGenerator',
]
