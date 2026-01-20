#!/usr/bin/env python3
"""
Phonetic Similarity Checker
============================
Checks if generated brand names sound similar to existing known brands.

Uses Soundex, Metaphone, and Levenshtein distance for comprehensive matching.
"""

from dataclasses import dataclass
from typing import Optional
import re


# Known brands that we should avoid sounding similar to
KNOWN_BRANDS = {
    # Camping/RV
    'dometic', 'truma', 'fiamma', 'thule', 'webasto', 'campingaz',
    'coleman', 'outwell', 'vango', 'karcher', 'waeco', 'mobicool',

    # Energy/Power
    'victron', 'renogy', 'ecoflow', 'bluetti', 'jackery', 'goalzero',
    'anker', 'duracell', 'energizer', 'varta', 'bosch', 'makita',
    'dewalt', 'milwaukee', 'ryobi', 'festool', 'metabo', 'hilti',

    # Tech Giants
    'tesla', 'apple', 'samsung', 'sony', 'panasonic', 'philips',
    'siemens', 'miele', 'braun', 'dyson', 'bose', 'harman',

    # Automotive
    'volkswagen', 'mercedes', 'bmw', 'audi', 'porsche', 'volvo',
    'ford', 'toyota', 'honda', 'mazda', 'nissan', 'hyundai',

    # Well-known consumer brands
    'amazon', 'google', 'microsoft', 'facebook', 'netflix', 'spotify',
    'nike', 'adidas', 'puma', 'reebok', 'asics', 'fila',
    'coca', 'pepsi', 'nestle', 'kraft', 'heinz', 'unilever',

    # Potentially confusing with our domain
    'voltron', 'volton', 'voltex', 'voltaic', 'voltage',
    'solaris', 'solar', 'solarex', 'sunpower', 'sunrun',
    'flux', 'fluxus', 'flexon', 'fluxx',
    'trek', 'trekking', 'tracker',
    'windex', 'windows', 'windstream',
    'ampex', 'ampere', 'amplitude',
    'luxor', 'luxus', 'luxury',
    'nomad', 'nomadic',
    'aurora', 'aura', 'aurum',
    'terra', 'terrain', 'terran',
    'aqua', 'aquafina', 'aquaman',
    'apex', 'apexel',
    'core', 'corel',
    'prime', 'primus', 'primo',
    'max', 'maxi', 'maxim',
    'neo', 'neon',
    'pulse', 'pulsar',
    'spark', 'sparky',
    'wave', 'waveform',
    'flow', 'flowtec',
    'link', 'lynx',
    'sync', 'synco',
    'hub', 'hubspot',
}


def soundex(name: str) -> str:
    """
    Generate Soundex code for a name.

    Soundex encodes similar-sounding names to the same code.
    Returns a 4-character code (letter + 3 digits).
    """
    name = name.upper()
    name = re.sub(r'[^A-Z]', '', name)

    if not name:
        return "0000"

    # Keep first letter
    first_letter = name[0]

    # Encoding map
    encoding = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
    }

    # Encode the rest
    code = first_letter
    prev_digit = encoding.get(first_letter, '0')

    for char in name[1:]:
        digit = encoding.get(char, '0')
        if digit != '0' and digit != prev_digit:
            code += digit
            prev_digit = digit
        elif digit == '0':
            prev_digit = '0'

    # Pad or truncate to 4 characters
    code = (code + '000')[:4]
    return code


def metaphone(name: str) -> str:
    """
    Generate Metaphone code for a name.

    More accurate than Soundex for English pronunciation.
    """
    name = name.upper()
    name = re.sub(r'[^A-Z]', '', name)

    if not name:
        return ""

    # Transformations
    result = []
    i = 0

    # Drop initial KN, GN, PN, AE, WR
    if name[:2] in ('KN', 'GN', 'PN', 'WR'):
        name = name[1:]
    elif name[:2] == 'AE':
        name = 'E' + name[2:]
    elif name[0] == 'X':
        name = 'S' + name[1:]
    elif name[:2] == 'WH':
        name = 'W' + name[2:]

    while i < len(name):
        char = name[i]

        # Vowels only kept at beginning
        if char in 'AEIOU':
            if i == 0:
                result.append(char)
            i += 1
            continue

        # Consonant transformations
        if char == 'B':
            if i == len(name) - 1 and name[i-1:i+1] == 'MB':
                pass  # Drop B after M at end
            else:
                result.append('B')
        elif char == 'C':
            if i < len(name) - 1 and name[i+1] in 'IEY':
                result.append('S')
            elif i < len(name) - 1 and name[i:i+2] == 'CH':
                result.append('X')
                i += 1
            else:
                result.append('K')
        elif char == 'D':
            if i < len(name) - 2 and name[i:i+3] in ('DGE', 'DGI', 'DGY'):
                result.append('J')
                i += 1
            else:
                result.append('T')
        elif char == 'F':
            result.append('F')
        elif char == 'G':
            if i < len(name) - 1 and name[i+1] in 'IEY':
                result.append('J')
            elif i < len(name) - 1 and name[i+1] == 'H':
                if i < len(name) - 2 and name[i+2] not in 'AEIOU':
                    i += 1  # GH silent
                else:
                    result.append('K')
            elif i < len(name) - 1 and name[i+1] == 'N':
                pass  # GN -> N
            else:
                result.append('K')
        elif char == 'H':
            if i > 0 and name[i-1] in 'AEIOU' and (i == len(name) - 1 or name[i+1] not in 'AEIOU'):
                pass  # H silent
            elif i == 0 or name[i-1] not in 'CSPTG':
                result.append('H')
        elif char == 'J':
            result.append('J')
        elif char == 'K':
            if i == 0 or name[i-1] != 'C':
                result.append('K')
        elif char == 'L':
            result.append('L')
        elif char == 'M':
            result.append('M')
        elif char == 'N':
            result.append('N')
        elif char == 'P':
            if i < len(name) - 1 and name[i+1] == 'H':
                result.append('F')
                i += 1
            else:
                result.append('P')
        elif char == 'Q':
            result.append('K')
        elif char == 'R':
            result.append('R')
        elif char == 'S':
            if i < len(name) - 1 and name[i:i+2] == 'SH':
                result.append('X')
                i += 1
            elif i < len(name) - 2 and name[i:i+3] in ('SIO', 'SIA'):
                result.append('X')
            else:
                result.append('S')
        elif char == 'T':
            if i < len(name) - 1 and name[i:i+2] == 'TH':
                result.append('0')  # TH sound
                i += 1
            elif i < len(name) - 2 and name[i:i+3] in ('TIO', 'TIA'):
                result.append('X')
            else:
                result.append('T')
        elif char == 'V':
            result.append('F')
        elif char == 'W':
            if i < len(name) - 1 and name[i+1] in 'AEIOU':
                result.append('W')
        elif char == 'X':
            result.append('KS')
        elif char == 'Y':
            if i < len(name) - 1 and name[i+1] in 'AEIOU':
                result.append('Y')
        elif char == 'Z':
            result.append('S')

        i += 1

    return ''.join(result)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalized_similarity(s1: str, s2: str) -> float:
    """Calculate normalized similarity (0-1, higher = more similar)."""
    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)


@dataclass
class SimilarityMatch:
    """A match found by similarity checking"""
    name: str
    known_brand: str
    soundex_match: bool
    metaphone_match: bool
    text_similarity: float

    @property
    def is_problematic(self) -> bool:
        """Check if this match is problematic (too similar)"""
        return (self.soundex_match or
                self.metaphone_match or
                self.text_similarity > 0.7)


@dataclass
class SimilarityResult:
    """Result of similarity checking"""
    name: str
    similar_brands: list  # List of SimilarityMatch
    is_safe: bool
    highest_similarity: float

    def __post_init__(self):
        if self.similar_brands is None:
            self.similar_brands = []


class SimilarityChecker:
    """
    Checks brand names for phonetic similarity to known brands.

    Usage:
        checker = SimilarityChecker()
        result = checker.check("Voltix")
        if not result.is_safe:
            print(f"Too similar to: {result.similar_brands}")
    """

    def __init__(self, additional_brands: set = None):
        """
        Initialize with known brands.

        Args:
            additional_brands: Extra brand names to check against
        """
        self.known_brands = KNOWN_BRANDS.copy()
        if additional_brands:
            self.known_brands.update(b.lower() for b in additional_brands)

        # Pre-compute phonetic codes for known brands
        self._soundex_cache = {b: soundex(b) for b in self.known_brands}
        self._metaphone_cache = {b: metaphone(b) for b in self.known_brands}

    def check(self, name: str, threshold: float = 0.6) -> SimilarityResult:
        """
        Check a name for similarity to known brands.

        Args:
            name: Brand name to check
            threshold: Minimum similarity score to flag (0-1)

        Returns:
            SimilarityResult with matches found
        """
        name_lower = name.lower()
        name_soundex = soundex(name)
        name_metaphone = metaphone(name)

        matches = []
        highest_sim = 0.0

        for brand in self.known_brands:
            # Skip if same name
            if brand == name_lower:
                continue

            # Check Soundex match
            soundex_match = self._soundex_cache[brand] == name_soundex

            # Check Metaphone match
            metaphone_match = self._metaphone_cache[brand] == name_metaphone

            # Calculate text similarity
            text_sim = normalized_similarity(name_lower, brand)

            # If any match is significant, record it
            if soundex_match or metaphone_match or text_sim > threshold:
                match = SimilarityMatch(
                    name=name,
                    known_brand=brand,
                    soundex_match=soundex_match,
                    metaphone_match=metaphone_match,
                    text_similarity=text_sim
                )
                matches.append(match)
                highest_sim = max(highest_sim, text_sim)

        # Sort by similarity
        matches.sort(key=lambda m: m.text_similarity, reverse=True)

        # Determine if safe (no problematic matches)
        is_safe = not any(m.is_problematic for m in matches)

        return SimilarityResult(
            name=name,
            similar_brands=matches[:5],  # Top 5 matches
            is_safe=is_safe,
            highest_similarity=highest_sim
        )

    def check_batch(self, names: list[str], threshold: float = 0.6) -> dict[str, SimilarityResult]:
        """Check multiple names."""
        return {name: self.check(name, threshold) for name in names}

    def filter_safe(self, names: list[str], threshold: float = 0.6) -> tuple[list[str], list[str]]:
        """
        Filter names into safe and problematic.

        Returns:
            Tuple of (safe_names, problematic_names)
        """
        safe = []
        problematic = []

        for name in names:
            result = self.check(name, threshold)
            if result.is_safe:
                safe.append(name)
            else:
                problematic.append(name)

        return safe, problematic


# Convenience function
def check_similarity(name: str) -> SimilarityResult:
    """Quick check for a single name."""
    checker = SimilarityChecker()
    return checker.check(name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Check brand name similarity')
    parser.add_argument('names', nargs='+', help='Names to check')
    parser.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold')

    args = parser.parse_args()

    checker = SimilarityChecker()

    for name in args.names:
        result = checker.check(name, args.threshold)

        print(f"\n{'='*50}")
        print(f"Name: {name}")
        print(f"Safe: {'Yes' if result.is_safe else 'NO - Similar to existing brands!'}")
        print(f"Highest similarity: {result.highest_similarity:.2f}")

        if result.similar_brands:
            print("\nSimilar brands found:")
            for m in result.similar_brands:
                flags = []
                if m.soundex_match:
                    flags.append("Soundex")
                if m.metaphone_match:
                    flags.append("Metaphone")
                flags.append(f"Sim:{m.text_similarity:.2f}")
                print(f"  - {m.known_brand}: [{', '.join(flags)}]")
