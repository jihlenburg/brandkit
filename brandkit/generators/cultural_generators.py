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
        # Japanese-specific consonants (no L, V, TH)
        self._consonants = list('kstmnhryw')
        self._vowels = list('aiueo')
        # Common Japanese-style endings
        self._endings = ['ra', 'no', 'ta', 'ko', 'ro', 'ka', 'mi', 'ri', 'to', 'ya']

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
        suffix = self._rng.choice(suffixes) if suffixes else self._rng.choice(self._endings)

        # Japanese style: often drop trailing vowel before adding suffix
        base = root
        if base[-1] in self._vowels and suffix[0] in self._vowels:
            base = base[:-1]

        name = base + suffix
        return (name, [root], [meaning], 'root_suffix')

    def _root_only(self, categories: List[str]) -> Optional[Tuple]:
        """Use root directly with possible vowel ending."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)

        # Japanese names typically end in vowels or 'n'
        if root[-1] not in self._vowels and root[-1] != 'n':
            root = root + self._rng.choice(['a', 'o', 'i'])

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

        result = []
        for char in pattern:
            if char == 'C':
                result.append(self._rng.choice(self._consonants))
            elif char == 'V':
                result.append(self._rng.choice(self._vowels))

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

        # Ensure ends in vowel or 'n'
        if name[-1] not in self._vowels and name[-1] != 'n':
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
        self._elegant_consonants = list('lrsvnf')
        self._strong_consonants = list('ctpdg')

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
        suffix = self._rng.choice(suffixes) if suffixes else self._rng.choice(['us', 'a', 'um', 'is', 'or', 'ix'])

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
        suffix = self._rng.choice(suffixes) if suffixes else self._rng.choice(['o', 'a', 'i', 'ino', 'etto', 'ella'])

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
        suffix = self._rng.choice(suffixes) if suffixes else self._rng.choice(['e', 'ique', 'elle', 'eur', 'oir'])

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
        self._celtic_consonants = list('cglnrwdm')

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
        suffix = self._rng.choice(suffixes) if suffixes else self._rng.choice(['an', 'en', 'wen', 'wyn', 'och', 'in'])

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

        # Welsh suffixes
        welsh_suffixes = ['wyn', 'wen', 'an', 'ion', 'ydd']
        suffix = self._rng.choice(welsh_suffixes)

        connector = self._get_connector(root[-1], suffix[0])
        name = root + connector + suffix

        return (name, [root], [meaning], 'welsh_style')

    def _gaelic_style(self, categories: List[str]) -> Optional[Tuple]:
        """Scottish/Irish Gaelic-influenced name."""
        pool = self._get_pool(categories)
        if not pool:
            return None

        root, meaning, cat = self._rng.choice(pool)

        # Gaelic prefixes and suffixes
        use_prefix = self._rng.choice([True, False])
        if use_prefix:
            prefix = self._rng.choice(['mac', 'ben', 'dun', 'glen'])
            name = prefix + root
        else:
            suffix = self._rng.choice(['ach', 'och', 'more', 'beg', 'an'])
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
        self._tech_consonants = list('vxzkltr')
        self._cosmic_endings = ['ar', 'on', 'ix', 'ex', 'us', 'is', 'a']

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
        suffix = self._rng.choice(suffixes) if suffixes else self._rng.choice(self._cosmic_endings)

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

        # Cosmic brand suffixes
        cosmic_suffixes = ['on', 'ar', 'ix', 'ex', 'um', 'is', 'a', 'or']
        suffix = self._rng.choice(cosmic_suffixes)

        # Optional tech/space prefix
        if self._rng.choice([True, False]):
            prefix = self._rng.choice(['neo', 'hyper', 'ultra', 'nova', ''])
            if prefix:
                connector = self._get_connector(prefix[-1], root[0])
                root = prefix + connector + root

        connector = self._get_connector(root[-1], suffix[0])
        name = root + connector + suffix

        return (name, [root], [meaning], 'cosmic_blend')


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'JapaneseGenerator',
    'LatinGenerator',
    'CelticGenerator',
    'CelestialGenerator',
]
