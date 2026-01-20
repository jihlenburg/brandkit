#!/usr/bin/env python3
"""
Greek Mythology-Inspired Name Generator
=======================================
Generates brand names inspired by Greek mythology, philosophy, and classical roots.

Loads phoneme configuration from YAML for easy customization.
Includes beasts, titans, elements, and abstract concepts.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Load config from YAML
from .phonemes import load_greek, load_strategies, get_connector, score_name


@dataclass
class GreekName:
    """A generated Greek-inspired name with metadata."""
    name: str
    roots_used: List[str]
    meaning_hints: List[str]
    score: float
    method: str


class GreekGenerator:
    """
    Generates brand names inspired by Greek mythology and classical roots.

    Loads configuration from phonemes/greek.yaml for easy customization.

    Usage:
        gen = GreekGenerator()
        names = gen.generate(count=20)
        for name in names:
            print(f"{name.name}: {name.meaning_hints}")
    """

    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)

        # Load configurations
        self._config = load_greek()
        self._strategies = load_strategies()

        # Build root pool
        self._all_roots = self._config.get_all_roots()

    def generate(self,
                 count: int = 20,
                 categories: List[str] = None,
                 min_length: int = 4,
                 max_length: int = 9,
                 archetype: str = None) -> List[GreekName]:
        """
        Generate Greek-inspired brand names.

        Parameters
        ----------
        count : int
            Number of names to generate
        categories : list, optional
            Filter by root categories (titans, beasts, light, elements, etc.)
        min_length : int
            Minimum name length
        max_length : int
            Maximum name length
        archetype : str, optional
            Brand archetype to influence generation (power, elegant, mythic, etc.)

        Returns
        -------
        list[GreekName]
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

            if len(name) < min_length or len(name) > max_length:
                continue

            if name.lower() in seen:
                continue
            seen.add(name.lower())

            # Use strategy-based scoring
            name_score = self._score_name(name)
            if name_score < 0.6:
                continue

            names.append(GreekName(
                name=name.capitalize(),
                roots_used=roots,
                meaning_hints=meanings,
                score=name_score,
                method=method
            ))

        names.sort(key=lambda n: n.score, reverse=True)
        return names[:count]

    def _get_weighted_methods(self) -> dict:
        """Get generation methods with weights from config."""
        method_config = self._strategies.strategies.get('methods', {})
        weights = {}
        for method, config in method_config.items():
            if isinstance(config, dict):
                weights[method] = config.get('weight', 10)
        # Ensure we have all our methods
        defaults = {
            'root_suffix': 25,
            'root_combo': 20,
            'truncate_suffix': 15,
            'prefix_root': 10,
            'pattern': 10,
            'fusion': 20,
            'european': 15,  # New: European suffix style
        }
        for method, weight in defaults.items():
            if method not in weights:
                weights[method] = weight
        return weights

    def _dispatch_method(self, method: str, categories: List[str], archetype: str) -> Optional[Tuple]:
        """Dispatch to appropriate generation method."""
        if method == 'root_suffix':
            return self._generate_root_suffix(categories, archetype)
        elif method == 'root_combo':
            return self._generate_root_combo(categories)
        elif method == 'truncate_suffix':
            return self._generate_truncate_suffix(categories, archetype)
        elif method == 'prefix_root':
            return self._generate_prefix_root(categories)
        elif method == 'pattern':
            return self._generate_pattern()
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

    def _generate_root_suffix(self, categories: List[str] = None, archetype: str = None) -> Optional[Tuple]:
        """Root + suffix."""
        pool = self._get_pool(categories)
        root, meaning, _ = random.choice(pool)

        # Truncate long roots
        if len(root) > 5:
            root = root[:5]

        # Choose suffix based on archetype or random
        if archetype and archetype in self._strategies.archetypes:
            arch = self._strategies.archetypes[archetype]
            suffix_pool = arch.get('preferred_suffixes', [])
            if not suffix_pool:
                suffix_pool = self._get_suffixes('tech', 'modern', 'neutral')
        else:
            suffix_pool = self._get_suffixes('tech', 'modern', 'neutral')

        suffix = random.choice(suffix_pool)

        # Use connector if needed
        if root and suffix:
            connector = get_connector(root[-1], suffix[0], self._strategies)
            name = root + connector + suffix
        else:
            name = root + suffix

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

    def _generate_truncate_suffix(self, categories: List[str] = None, archetype: str = None) -> Optional[Tuple]:
        """Short root + strong suffix."""
        pool = self._get_pool(categories)
        root, meaning, _ = random.choice(pool)

        # Take first 3-4 chars
        short = root[:random.randint(3, 4)]
        suffix = random.choice(self._get_suffixes('latin', 'greek'))

        # Apply connector if needed
        if short and suffix:
            connector = get_connector(short[-1], suffix[0], self._strategies)
            name = short + connector + suffix
        else:
            name = short + suffix

        return name, [root], [meaning]

    def _generate_prefix_root(self, categories: List[str] = None) -> Optional[Tuple]:
        """Prefix + root."""
        pool = self._get_pool(categories)
        root, meaning, _ = random.choice(pool)
        prefix = random.choice(self._get_prefixes())

        # Truncate if needed
        if len(root) > 6:
            root = root[:6]

        return prefix + root, [root], [meaning]

    def _generate_pattern(self) -> Optional[Tuple]:
        """Greek phonetic pattern."""
        patterns = self._config.patterns.get('greek_style', ['CVCVC', 'CVCCV', 'CVCVS'])
        pattern = random.choice(patterns)

        phonetics = self._config.phonetics
        consonants = phonetics.get('consonants', {}).get('greek_common', list('klmnprst'))
        vowels = phonetics.get('vowels', {}).get('all', list('aeiou'))

        name = ""
        for char in pattern:
            if char == 'C':
                name += random.choice(consonants)
            elif char == 'V':
                name += random.choice(vowels)
            elif char == 'S':
                name += 's'

        return name, [], ["constructed from Greek patterns"]

    def _generate_fusion(self, categories: List[str] = None, archetype: str = None) -> Optional[Tuple]:
        """Blend Greek root with modern tech suffix."""
        pool = self._get_pool(categories)
        root, meaning, _ = random.choice(pool)

        # Short form of root
        short = root[:4] if len(root) > 4 else root

        # Modern tech suffixes
        suffix = random.choice(['ix', 'ex', 'on', 'um', 'ia', 'eon', 'or', 'ax'])

        # Apply connector
        connector = get_connector(short[-1], suffix[0], self._strategies)
        name = short + connector + suffix

        return name, [root], [meaning]

    def _generate_european(self, categories: List[str] = None) -> Optional[Tuple]:
        """Greek root + European-style suffix (like Turkic fusion)."""
        pool = self._get_pool(categories)

        strategy = random.choice([
            'root_euro', 'double_root', 'beast_euro', 'archetype_blend'
        ])

        if strategy == 'root_euro':
            root, meaning, _ = random.choice(pool)
            suffix = random.choice(self._get_suffixes('european'))

            if len(root) > 5:
                root = root[:5]

            connector = get_connector(root[-1], suffix[0], self._strategies)
            name = root + connector + suffix

            return name, [root], [meaning]

        elif strategy == 'double_root':
            root1, meaning1, _ = random.choice(pool)
            root2, meaning2, _ = random.choice(pool)
            suffix = random.choice(self._get_suffixes('latin', 'european'))

            # Short combination
            if len(root1) > 4:
                combined = root1[:4] + root2[:3]
            else:
                combined = root1 + root2[:3]

            name = combined + suffix
            return name, [root1, root2], [meaning1, meaning2]

        elif strategy == 'beast_euro':
            # Specifically use beasts category with European suffix
            beasts = self._config.get_roots_by_category('beasts')
            if beasts:
                root, meaning, _ = random.choice(beasts)
            else:
                root, meaning, _ = random.choice(pool)

            suffix = random.choice(self._get_suffixes('european', 'latin'))

            if len(root) > 5:
                root = root[:5]

            connector = get_connector(root[-1], suffix[0], self._strategies)
            name = root + connector + suffix

            return name, [root], [meaning]

        else:  # archetype_blend
            # Use power archetype settings
            arch = self._strategies.archetypes.get('mythic', {})
            preferred_sounds = arch.get('preferred_sounds', ['t', 'k', 's', 'n', 'r'])

            root, meaning, _ = random.choice(pool)
            if len(root) > 4:
                root = root[:4]

            suffix = random.choice(self._get_suffixes('european', 'neutral'))
            connector = get_connector(root[-1], suffix[0], self._strategies)
            name = root + connector + suffix

            return name, [root], [meaning]

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    def _score_name(self, name: str) -> float:
        """Score using strategy-based parameters."""
        return score_name(name, self._strategies)

    def generate_batch(self, count: int = 20, **kwargs) -> List[GreekName]:
        return self.generate(count=count, **kwargs)


def generate_greek(count: int = 20, categories: List[str] = None) -> List[GreekName]:
    """Quick generation function."""
    gen = GreekGenerator()
    return gen.generate(count=count, categories=categories)


if __name__ == '__main__':
    import argparse

    # Get available categories from config
    config = load_greek()
    available_categories = list(config.roots.keys())

    parser = argparse.ArgumentParser(description='Generate Greek-inspired brand names')
    parser.add_argument('-n', '--count', type=int, default=20)
    parser.add_argument('-c', '--categories', nargs='+', choices=available_categories)
    parser.add_argument('-a', '--archetype', choices=['power', 'elegant', 'dynamic', 'natural', 'mythic'])
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    gen = GreekGenerator()
    names = gen.generate(count=args.count, categories=args.categories, archetype=args.archetype)

    print(f"Generated {len(names)} Greek-inspired names:\n")
    for i, name in enumerate(names, 1):
        if args.verbose:
            print(f"{i:2}. {name.name:<12} [{name.score:.2f}] ({name.method})")
            print(f"    Roots: {name.roots_used}, Hints: {', '.join(name.meaning_hints)}")
        else:
            print(f"{i:2}. {name.name:<12} [{name.score:.2f}]")
