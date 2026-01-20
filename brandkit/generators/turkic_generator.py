#!/usr/bin/env python3
"""
Turkic-Inspired Name Generator
==============================
Generates brand names inspired by Turkic language phonetics and roots.

Loads configuration from YAML for easy customization.

Draws from multiple Turkic language branches:
- Oghuz (Turkish, Azerbaijani, Turkmen)
- Kipchak (Kazakh, Kyrgyz, Tatar, Kumyk, Bashkir)
- Siberian (Yakut/Sakha, Tuvan, Altai, Khakas)
- Karluk (Uzbek, Uyghur)
- Ancient/Proto-Turkic (common roots)

All roots are Latin-alphabet friendly and filtered for German/English
pronounceability.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Load config from YAML
from .phonemes import (
    load_turkic, load_strategies, get_connector, score_name,
    apply_vowel_harmony
)


@dataclass
class TurkicName:
    """A generated Turkic-inspired name with metadata."""
    name: str
    roots_used: List[str]
    meaning_hints: List[str]
    score: float
    method: str


class TurkicGenerator:
    """
    Generates brand names with Turkic-inspired phonetics.

    Loads configuration from phonemes/turkic.yaml for easy customization.
    Draws from multiple Turkic branches for diverse, authentic-sounding names.

    Usage:
        gen = TurkicGenerator()
        names = gen.generate(count=20)
        for name in names:
            print(f"{name.name}: {name.meaning_hints}")
    """

    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)

        # Load configurations
        self._config = load_turkic()
        self._strategies = load_strategies()

        # Build root pool
        self._all_roots = self._config.get_all_roots()

    def generate(self,
                 count: int = 20,
                 categories: List[str] = None,
                 min_length: int = 4,
                 max_length: int = 8,
                 archetype: str = None,
                 vowel_harmony: bool = True) -> List[TurkicName]:
        """
        Generate Turkic-inspired brand names.

        Parameters
        ----------
        count : int
            Number of names to generate
        categories : list, optional
            Filter by root categories (power, travel, celestial, etc.)
        min_length : int
            Minimum name length
        max_length : int
            Maximum name length
        archetype : str, optional
            Brand archetype (power, elegant, dynamic, etc.)
        vowel_harmony : bool
            Apply Turkic vowel harmony rules (default True)

        Returns
        -------
        list[TurkicName]
            Generated names sorted by score
        """
        names = []
        attempts = 0
        max_attempts = count * 15
        seen = set()

        # Get method weights
        methods = self._get_weighted_methods()

        while len(names) < count and attempts < max_attempts:
            attempts += 1

            # Choose method based on weights
            method = random.choices(
                list(methods.keys()),
                weights=list(methods.values()),
                k=1
            )[0]

            result = self._dispatch_method(method, categories, archetype)

            if result is None:
                continue

            name, roots, meanings = result

            # Optionally apply vowel harmony
            if vowel_harmony and random.random() < 0.3:
                name = apply_vowel_harmony(name, use_back=random.choice([True, False]))

            if len(name) < min_length or len(name) > max_length:
                continue

            if name.lower() in seen:
                continue
            seen.add(name.lower())

            name_score = self._score_name(name)
            if name_score < 0.6:
                continue

            names.append(TurkicName(
                name=name.capitalize(),
                roots_used=roots,
                meaning_hints=meanings,
                score=name_score,
                method=method
            ))

        names.sort(key=lambda n: n.score, reverse=True)
        return names[:count]

    def _get_weighted_methods(self) -> dict:
        """Get generation methods with weights."""
        # Default weights for Turkic generator
        return {
            'root_combo': 20,
            'pattern': 15,
            'hybrid': 20,
            'fusion': 25,
            'european': 20,
        }

    def _dispatch_method(self, method: str, categories: List[str], archetype: str) -> Optional[Tuple]:
        """Dispatch to appropriate generation method."""
        if method == 'root_combo':
            return self._generate_root_combo(categories)
        elif method == 'pattern':
            return self._generate_pattern()
        elif method == 'hybrid':
            return self._generate_hybrid(categories)
        elif method == 'fusion':
            return self._generate_fusion(categories, archetype)
        elif method == 'european':
            return self._generate_european(categories)
        return None

    def _get_pool(self, categories: List[str] = None) -> List[Tuple[str, str, str]]:
        """Get root pool, optionally filtered by categories."""
        if categories:
            filtered = self._config.get_roots_by_category(*categories)
            if filtered:
                return filtered
        return self._all_roots

    def _get_suffixes(self, *suffix_types: str) -> List[str]:
        """Get suffix pool from specified types."""
        return self._config.get_suffix_pool(*suffix_types)

    def _get_prefixes(self) -> List[str]:
        """Get prefix pool."""
        prefixes = []
        for ptype, plist in self._config.prefixes.items():
            prefixes.extend(plist)
        return prefixes

    # -------------------------------------------------------------------------
    # Generation Methods
    # -------------------------------------------------------------------------

    def _generate_root_combo(self, categories: List[str] = None) -> Optional[Tuple]:
        """Combine two Turkic roots."""
        pool = self._get_pool(categories)

        root1, meaning1, _ = random.choice(pool)
        root2, meaning2, _ = random.choice(pool)

        strategy = random.choice(['direct', 'overlap', 'truncate'])

        if strategy == 'direct':
            name = root1 + root2
        elif strategy == 'overlap':
            # Try to find overlapping sounds
            overlap = 0
            for i in range(1, min(len(root1), len(root2))):
                if root1[-i:] == root2[:i]:
                    overlap = i
            name = root1 + root2[overlap:] if overlap else root1 + root2
        else:
            name = root1[:3] + root2

        return name, [root1, root2], [meaning1, meaning2]

    def _generate_pattern(self) -> Optional[Tuple]:
        """Generate from Turkic phonetic patterns."""
        patterns = self._config.patterns.get('turkic_style', ['CVCVC', 'CVCan'])
        pattern = random.choice(patterns)

        phonetics = self._config.phonetics
        consonants_strong = phonetics.get('consonants', {}).get('strong', list('ktbdgr'))
        consonants_medium = phonetics.get('consonants', {}).get('medium', list('slmnvyzp'))
        vowels_back = phonetics.get('vowels', {}).get('back', list('aou'))
        vowels_front = phonetics.get('vowels', {}).get('front', list('ei'))

        # Choose vowel set for harmony
        use_back = random.choice([True, False])
        vowel_set = vowels_back if use_back else vowels_front

        name = ""
        i = 0
        while i < len(pattern):
            char = pattern[i]

            if char == 'C':
                if random.random() < 0.6:
                    name += random.choice(consonants_strong)
                else:
                    name += random.choice(consonants_medium)
            elif char == 'V':
                name += random.choice(vowel_set)
            elif char == 'R':
                name += 'r'
            else:
                # Literal character (like 'an', 'ay', 'ar', 'er')
                literal = ""
                while i < len(pattern) and pattern[i] not in 'CVR':
                    literal += pattern[i]
                    i += 1
                name += literal
                continue
            i += 1

        return name, [], ["Turkic phonetic pattern"]

    def _generate_hybrid(self, categories: List[str] = None) -> Optional[Tuple]:
        """Turkic root with neutral suffix."""
        pool = self._get_pool(categories)
        root, meaning, category = random.choice(pool)

        # Choose suffix pool based on category
        if category in ['power', 'fire']:
            suffix_pool = self._get_suffixes('tech', 'turkic')
        elif category in ['travel', 'spirit']:
            suffix_pool = self._get_suffixes('turkic', 'neutral')
        else:
            suffix_pool = self._get_suffixes('neutral', 'turkic')

        suffix = random.choice(suffix_pool) if suffix_pool else 'an'
        prefix = random.choice(self._get_prefixes())

        # Apply connector if needed
        if root and suffix:
            connector = get_connector(root[-1], suffix[0], self._strategies)
            name = prefix + root + connector + suffix
        else:
            name = prefix + root + suffix

        return name, [root], [meaning]

    def _generate_fusion(self, categories: List[str] = None, archetype: str = None) -> Optional[Tuple]:
        """Turkic root + international suffix (European style)."""
        pool = self._get_pool(categories)

        strategy = random.choice([
            'root_latin', 'double_root', 'root_modern', 'pattern_suffix'
        ])

        if strategy == 'root_latin':
            root, meaning, _ = random.choice(pool)
            suffix = random.choice(self._get_suffixes('latin', 'modern'))

            connector = get_connector(root[-1], suffix[0], self._strategies)
            name = root + connector + suffix

            return name, [root], [meaning]

        elif strategy == 'double_root':
            root1, meaning1, _ = random.choice(pool)
            root2, meaning2, _ = random.choice(pool)
            suffix = random.choice(self._get_suffixes('latin'))

            if len(root1) > 3:
                combined = root1[:4] + root2[:3]
            else:
                combined = root1 + root2[:3]

            name = combined + suffix
            return name, [root1, root2], [meaning1, meaning2]

        elif strategy == 'root_modern':
            root, meaning, _ = random.choice(pool)
            suffix = random.choice(self._get_suffixes('modern'))

            connector = get_connector(root[-1], suffix[0], self._strategies)
            name = root + connector + suffix

            return name, [root], [meaning]

        else:  # pattern_suffix
            phonetics = self._config.phonetics
            base_c = random.choices(
                phonetics.get('consonants', {}).get('strong', list('ktbdgr')),
                k=2
            )
            base_v = random.choices(
                phonetics.get('vowels', {}).get('back', list('aou')),
                k=2
            )
            base = base_c[0] + base_v[0] + base_c[1] + base_v[1]

            suffix = random.choice(self._get_suffixes('latin') + ['an', 'ar'])
            name = base + suffix

            return name, [], ["Turkic-Latin fusion pattern"]

    def _generate_european(self, categories: List[str] = None) -> Optional[Tuple]:
        """Turkic root + European-style suffix."""
        pool = self._get_pool(categories)

        strategy = random.choice([
            'root_euro', 'animal_euro', 'power_euro', 'nature_euro'
        ])

        if strategy == 'root_euro':
            root, meaning, _ = random.choice(pool)
            suffix = random.choice(self._get_suffixes('european'))

            if len(root) > 5:
                root = root[:5]

            connector = get_connector(root[-1], suffix[0], self._strategies)
            name = root + connector + suffix

            return name, [root], [meaning]

        elif strategy == 'animal_euro':
            # Use animal category with European suffix
            animals = self._config.get_roots_by_category('animal')
            if animals:
                root, meaning, _ = random.choice(animals)
            else:
                root, meaning, _ = random.choice(pool)

            suffix = random.choice(self._get_suffixes('european', 'latin'))

            if len(root) > 5:
                root = root[:5]

            connector = get_connector(root[-1], suffix[0], self._strategies)
            name = root + connector + suffix

            return name, [root], [meaning]

        elif strategy == 'power_euro':
            # Use power/fire categories with European suffix
            power_roots = self._config.get_roots_by_category('power', 'fire')
            if power_roots:
                root, meaning, _ = random.choice(power_roots)
            else:
                root, meaning, _ = random.choice(pool)

            suffix = random.choice(self._get_suffixes('european', 'tech'))

            if len(root) > 4:
                root = root[:4]

            connector = get_connector(root[-1], suffix[0], self._strategies)
            name = root + connector + suffix

            return name, [root], [meaning]

        else:  # nature_euro
            # Use nature/water/celestial categories
            nature_roots = self._config.get_roots_by_category('nature', 'water', 'celestial')
            if nature_roots:
                root, meaning, _ = random.choice(nature_roots)
            else:
                root, meaning, _ = random.choice(pool)

            suffix = random.choice(self._get_suffixes('european', 'neutral'))

            if len(root) > 5:
                root = root[:5]

            connector = get_connector(root[-1], suffix[0], self._strategies)
            name = root + connector + suffix

            return name, [root], [meaning]

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    def _score_name(self, name: str) -> float:
        """Score using strategy-based parameters with Turkic bonuses."""
        # Base score from strategies
        base_score = score_name(name, self._strategies)

        # Additional Turkic-specific scoring
        name_lower = name.lower()

        # Strong Turkic start bonus
        if name_lower[0] in 'tkbdgvr':
            base_score += 0.1

        # Turkic-feel endings bonus
        turkic_endings = ['an', 'ay', 'kan', 'tan', 'ar', 'er', 'on', 'yk', 'ik']
        for ending in turkic_endings:
            if name_lower.endswith(ending):
                base_score += 0.1
                break

        return min(max(base_score, 0.0), 1.0)

    def generate_batch(self, count: int = 20, **kwargs) -> List[TurkicName]:
        return self.generate(count=count, **kwargs)


def generate_turkic(count: int = 20, categories: List[str] = None) -> List[TurkicName]:
    """Quick generation function."""
    gen = TurkicGenerator()
    return gen.generate(count=count, categories=categories)


if __name__ == '__main__':
    import argparse

    # Get available categories from config
    config = load_turkic()
    available_categories = list(config.roots.keys())

    parser = argparse.ArgumentParser(description='Generate Turkic-inspired brand names')
    parser.add_argument('-n', '--count', type=int, default=20)
    parser.add_argument('-c', '--categories', nargs='+', choices=available_categories)
    parser.add_argument('-a', '--archetype', choices=['power', 'elegant', 'dynamic', 'natural', 'mythic'])
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--no-harmony', action='store_true', help='Disable vowel harmony')

    args = parser.parse_args()

    gen = TurkicGenerator()
    names = gen.generate(
        count=args.count,
        categories=args.categories,
        archetype=args.archetype,
        vowel_harmony=not args.no_harmony
    )

    print(f"Generated {len(names)} Turkic-inspired names:\n")
    for i, name in enumerate(names, 1):
        if args.verbose:
            print(f"{i:2}. {name.name:<12} [{name.score:.2f}] ({name.method})")
            print(f"    Roots: {name.roots_used}, Hints: {', '.join(name.meaning_hints)}")
        else:
            print(f"{i:2}. {name.name:<12} [{name.score:.2f}]")
