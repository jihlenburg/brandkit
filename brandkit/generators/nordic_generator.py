#!/usr/bin/env python3
"""
Nordic/Scandinavian-Inspired Name Generator
===========================================
Generates brand names inspired by Old Norse, Swedish, Norwegian, Danish, Icelandic.

Loads configuration from YAML for easy customization.

Features:
- Norse mythology (gods, creatures, realms)
- Scandinavian nature words (fjord, berg, skog)
- Rune names and meanings
- Power/strength vocabulary
- Nordic-style suffixes

All roots are Latin-alphabet friendly (no ö, å, ä, þ, ð).
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Load config from YAML
from .phonemes import (
    load_nordic, load_strategies, get_connector, score_name
)


@dataclass
class NordicName:
    """A generated Nordic-inspired name with metadata."""
    name: str
    roots_used: List[str]
    meaning_hints: List[str]
    score: float
    method: str


class NordicGenerator:
    """
    Generates brand names with Nordic/Scandinavian phonetics.

    Loads configuration from phonemes/nordic.yaml for easy customization.

    Usage:
        gen = NordicGenerator()
        names = gen.generate(count=20)
        for name in names:
            print(f"{name.name}: {name.meaning_hints}")
    """

    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)

        # Load configurations
        self._config = load_nordic()
        self._strategies = load_strategies()

        # Build root pool
        self._all_roots = self._config.get_all_roots()

    def generate(self,
                 count: int = 20,
                 categories: List[str] = None,
                 min_length: int = 4,
                 max_length: int = 9,
                 archetype: str = None) -> List[NordicName]:
        """
        Generate Nordic-inspired brand names.

        Parameters
        ----------
        count : int
            Number of names to generate
        categories : list, optional
            Filter by root categories (gods, beasts, power, nature, etc.)
        min_length : int
            Minimum name length
        max_length : int
            Maximum name length
        archetype : str, optional
            Brand archetype (power, elegant, dynamic, etc.)

        Returns
        -------
        list[NordicName]
            Generated names sorted by score
        """
        names = []
        attempts = 0
        max_attempts = count * 15
        seen = set()

        # Method weights
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

            if len(name) < min_length or len(name) > max_length:
                continue

            if name.lower() in seen:
                continue
            seen.add(name.lower())

            name_score = self._score_name(name)
            if name_score < 0.6:
                continue

            names.append(NordicName(
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
        return {
            'root_suffix': 25,
            'root_combo': 20,
            'truncate_suffix': 15,
            'god_blend': 15,
            'rune_tech': 10,
            'nature_euro': 15,
        }

    def _dispatch_method(self, method: str, categories: List[str], archetype: str) -> Optional[Tuple]:
        """Dispatch to appropriate generation method."""
        if method == 'root_suffix':
            return self._generate_root_suffix(categories, archetype)
        elif method == 'root_combo':
            return self._generate_root_combo(categories)
        elif method == 'truncate_suffix':
            return self._generate_truncate_suffix(categories)
        elif method == 'god_blend':
            return self._generate_god_blend()
        elif method == 'rune_tech':
            return self._generate_rune_tech()
        elif method == 'nature_euro':
            return self._generate_nature_euro()
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

    def _generate_root_suffix(self, categories: List[str] = None, archetype: str = None) -> Optional[Tuple]:
        """Root + suffix."""
        pool = self._get_pool(categories)
        root, meaning, _ = random.choice(pool)

        # Truncate long roots
        if len(root) > 5:
            root = root[:5]

        # Choose suffix based on archetype or random
        if archetype == 'power':
            suffix_pool = self._get_suffixes('nordic', 'tech')
        elif archetype == 'natural':
            suffix_pool = self._get_suffixes('nordic', 'neutral')
        else:
            suffix_pool = self._get_suffixes('nordic', 'neutral', 'tech')

        suffix = random.choice(suffix_pool) if suffix_pool else 'en'

        # Apply connector if needed
        connector = get_connector(root[-1], suffix[0], self._strategies)
        name = root + connector + suffix

        return name, [root], [meaning]

    def _generate_root_combo(self, categories: List[str] = None) -> Optional[Tuple]:
        """Combine two roots."""
        pool = self._get_pool(categories)
        root1, meaning1, _ = random.choice(pool)
        root2, meaning2, _ = random.choice(pool)

        # Truncate
        r1 = root1[:4] if len(root1) > 4 else root1
        r2 = root2[:4] if len(root2) > 4 else root2

        # Try overlapping sounds
        overlap = 0
        for i in range(1, min(len(r1), len(r2))):
            if r1[-i:] == r2[:i]:
                overlap = i

        if overlap:
            name = r1 + r2[overlap:]
        else:
            name = r1 + r2

        return name, [root1, root2], [meaning1, meaning2]

    def _generate_truncate_suffix(self, categories: List[str] = None) -> Optional[Tuple]:
        """Short root + strong suffix."""
        pool = self._get_pool(categories)
        root, meaning, _ = random.choice(pool)

        # Take first 3-4 chars
        short = root[:random.randint(3, 4)]
        suffix = random.choice(self._get_suffixes('latin', 'norse', 'tech'))

        # Apply connector
        connector = get_connector(short[-1], suffix[0], self._strategies)
        name = short + connector + suffix

        return name, [root], [meaning]

    def _generate_god_blend(self) -> Optional[Tuple]:
        """Blend god name with power/nature root."""
        gods = self._config.get_roots_by_category('gods', 'beasts')
        power = self._config.get_roots_by_category('power', 'nature')

        if not gods or not power:
            return None

        god, god_meaning, _ = random.choice(gods)
        other, other_meaning, _ = random.choice(power)

        # Short god name + suffix from other
        if len(god) > 4:
            god = god[:4]

        suffix = random.choice(self._get_suffixes('nordic', 'european', 'latin'))
        connector = get_connector(god[-1], suffix[0], self._strategies)
        name = god + connector + suffix

        return name, [god], [god_meaning]

    def _generate_rune_tech(self) -> Optional[Tuple]:
        """Rune name + tech suffix."""
        runes = self._config.get_roots_by_category('runes')

        if not runes:
            return self._generate_root_suffix()

        rune, meaning, _ = random.choice(runes)

        # Truncate if needed
        if len(rune) > 5:
            rune = rune[:5]

        suffix = random.choice(self._get_suffixes('tech', 'latin'))
        connector = get_connector(rune[-1], suffix[0], self._strategies)
        name = rune + connector + suffix

        return name, [rune], [meaning]

    def _generate_nature_euro(self) -> Optional[Tuple]:
        """Nature word + European suffix."""
        nature = self._config.get_roots_by_category('nature', 'celestial', 'animals')

        if not nature:
            return self._generate_root_suffix()

        root, meaning, _ = random.choice(nature)

        if len(root) > 5:
            root = root[:5]

        suffix = random.choice(self._get_suffixes('european', 'neutral'))
        connector = get_connector(root[-1], suffix[0], self._strategies)
        name = root + connector + suffix

        return name, [root], [meaning]

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    def _score_name(self, name: str) -> float:
        """Score using strategy-based parameters with Nordic bonuses."""
        # Base score from strategies
        base_score = score_name(name, self._strategies)

        name_lower = name.lower()

        # Nordic-feel endings bonus
        nordic_endings = ['en', 'ar', 'ir', 'ur', 'heim', 'gard', 'vik', 'dal', 'berg']
        for ending in nordic_endings:
            if name_lower.endswith(ending):
                base_score += 0.1
                break

        # Strong Nordic start bonus
        if name_lower[0] in 'tkbdgfrsvh':
            base_score += 0.05

        return min(max(base_score, 0.0), 1.0)

    def generate_batch(self, count: int = 20, **kwargs) -> List[NordicName]:
        return self.generate(count=count, **kwargs)


def generate_nordic(count: int = 20, categories: List[str] = None) -> List[NordicName]:
    """Quick generation function."""
    gen = NordicGenerator()
    return gen.generate(count=count, categories=categories)


if __name__ == '__main__':
    import argparse

    # Get available categories from config
    config = load_nordic()
    available_categories = list(config.roots.keys())

    parser = argparse.ArgumentParser(description='Generate Nordic-inspired brand names')
    parser.add_argument('-n', '--count', type=int, default=20)
    parser.add_argument('-c', '--categories', nargs='+', choices=available_categories)
    parser.add_argument('-a', '--archetype', choices=['power', 'elegant', 'dynamic', 'natural', 'mythic'])
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    gen = NordicGenerator()
    names = gen.generate(
        count=args.count,
        categories=args.categories,
        archetype=args.archetype
    )

    print(f"Generated {len(names)} Nordic-inspired names:\n")
    for i, name in enumerate(names, 1):
        if args.verbose:
            print(f"{i:2}. {name.name:<12} [{name.score:.2f}] ({name.method})")
            print(f"    Roots: {name.roots_used}, Hints: {', '.join(name.meaning_hints)}")
        else:
            print(f"{i:2}. {name.name:<12} [{name.score:.2f}]")
