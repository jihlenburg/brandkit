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

from settings import get_setting

def _load_known_brands() -> set[str]:
    brands = get_setting("similarity_checker.known_brands")
    if not brands:
        raise ValueError("similarity_checker.known_brands must be set in app.yaml")
    return set(b.lower() for b in brands)


KNOWN_BRANDS = _load_known_brands()
DEFAULT_THRESHOLD = get_setting("similarity_checker.default_threshold")
PROBLEMATIC_THRESHOLD = get_setting("similarity_checker.problematic_threshold")
DEFAULT_MARKETS = get_setting("similarity_checker.default_markets")
TOP_MATCHES = get_setting("similarity_checker.top_matches")
if (DEFAULT_THRESHOLD is None or PROBLEMATIC_THRESHOLD is None or
        DEFAULT_MARKETS is None or TOP_MATCHES is None):
    raise ValueError("similarity_checker defaults must be set in app.yaml")


def cologne_phonetics(name: str) -> str:
    """
    Generate Cologne Phonetics (Kölner Phonetik) code for a name.

    Specifically designed for German names and pronunciation.
    More accurate than Soundex/Metaphone for German words.

    Based on Hans Joachim Postel's algorithm (1969).
    """
    name = name.upper()
    name = re.sub(r'[^A-ZÄÖÜß]', '', name)

    if not name:
        return ""

    # Handle German umlauts and eszett
    name = name.replace('Ä', 'A').replace('Ö', 'O').replace('Ü', 'U').replace('ß', 'SS')

    result = []
    i = 0

    while i < len(name):
        char = name[i]
        prev = name[i-1] if i > 0 else ''
        next_char = name[i+1] if i < len(name) - 1 else ''

        code = None

        # Vowels (A, E, I, O, U, Y) → 0
        if char in 'AEIOUY':
            code = '0'

        # H → ignored (no code)
        elif char == 'H':
            pass

        # B → 1
        elif char == 'B':
            code = '1'

        # P → 1 (except before H → 3)
        elif char == 'P':
            if next_char == 'H':
                code = '3'
            else:
                code = '1'

        # D, T → 2 (except before C, S, Z → 8)
        elif char in 'DT':
            if next_char in 'CSZ':
                code = '8'
            else:
                code = '2'

        # F, V, W → 3
        elif char in 'FVW':
            code = '3'

        # G, K, Q → 4
        elif char in 'GKQ':
            code = '4'

        # C rules (complex)
        elif char == 'C':
            # C at start: before A, H, K, L, O, Q, R, U, X → 4, else 8
            if i == 0:
                if next_char in 'AHKLOQRUX':
                    code = '4'
                else:
                    code = '8'
            # C after S, Z → 8
            elif prev in 'SZ':
                code = '8'
            # C not at start: before A, H, K, O, Q, U, X → 4
            elif next_char in 'AHKOQUX':
                code = '4'
            else:
                code = '8'

        # X → 48 (except after C, K, Q → 8)
        elif char == 'X':
            if prev in 'CKQ':
                code = '8'
            else:
                code = '48'

        # L → 5
        elif char == 'L':
            code = '5'

        # M, N → 6
        elif char in 'MN':
            code = '6'

        # R → 7
        elif char == 'R':
            code = '7'

        # S, Z → 8
        elif char in 'SZ':
            code = '8'

        # J → 0 (same as vowels in German context)
        elif char == 'J':
            code = '0'

        if code:
            result.append(code)

        i += 1

    # Remove consecutive duplicates
    if not result:
        return ""

    final = [result[0]]
    for code in result[1:]:
        if code != final[-1]:
            final.append(code)

    # Remove leading zeros (except if it's all zeros)
    code_str = ''.join(final)
    code_str = code_str.lstrip('0') or '0'

    return code_str


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
    cologne_match: bool  # Cologne Phonetics match (German-optimized)
    text_similarity: float

    @property
    def is_problematic(self) -> bool:
        """Check if this match is problematic (too similar)"""
        return (self.soundex_match or
                self.metaphone_match or
                self.cologne_match or
                self.text_similarity > PROBLEMATIC_THRESHOLD)

    @property
    def phonetic_match_count(self) -> int:
        """Number of phonetic algorithms that flagged this match"""
        return sum([self.soundex_match, self.metaphone_match, self.cologne_match])


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

    def __init__(self, additional_brands: set = None, markets: Optional[str] = None):
        """
        Initialize with known brands.

        Args:
            additional_brands: Extra brand names to check against
            markets: Target market(s): 'en', 'de', or 'en_de' (default).
                     Affects which phonetic algorithms are emphasized.
        """
        self.known_brands = KNOWN_BRANDS.copy()
        self.markets = markets or DEFAULT_MARKETS
        if additional_brands:
            self.known_brands.update(b.lower() for b in additional_brands)

        # Pre-compute phonetic codes for known brands
        self._soundex_cache = {b: soundex(b) for b in self.known_brands}
        self._metaphone_cache = {b: metaphone(b) for b in self.known_brands}
        self._cologne_cache = {b: cologne_phonetics(b) for b in self.known_brands}

    def check(self, name: str, threshold: Optional[float] = None,
              markets: str = None) -> SimilarityResult:
        """
        Check a name for similarity to known brands.

        Args:
            name: Brand name to check
            threshold: Minimum similarity score to flag (0-1)
            markets: Override instance markets setting for this check.
                     'en' = Soundex+Metaphone, 'de' = Cologne+Metaphone,
                     'en_de' = all three algorithms (default).

        Returns:
            SimilarityResult with matches found
        """
        markets = markets or self.markets
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        name_lower = name.lower()
        name_soundex = soundex(name)
        name_metaphone = metaphone(name)
        name_cologne = cologne_phonetics(name)

        matches = []
        highest_sim = 0.0

        for brand in self.known_brands:
            # Skip if same name
            if brand == name_lower:
                continue

            # Check Soundex match (English-focused)
            soundex_match = self._soundex_cache[brand] == name_soundex

            # Check Metaphone match (English, but good for both)
            metaphone_match = self._metaphone_cache[brand] == name_metaphone

            # Check Cologne Phonetics match (German-focused)
            cologne_match = self._cologne_cache[brand] == name_cologne

            # Calculate text similarity
            text_sim = normalized_similarity(name_lower, brand)

            # Determine if any phonetic match is significant based on market
            has_phonetic_match = False
            if markets == 'en':
                has_phonetic_match = soundex_match or metaphone_match
            elif markets == 'de':
                has_phonetic_match = cologne_match or metaphone_match
            else:  # en_de
                has_phonetic_match = soundex_match or metaphone_match or cologne_match

            # If any match is significant, record it
            if has_phonetic_match or text_sim > threshold:
                match = SimilarityMatch(
                    name=name,
                    known_brand=brand,
                    soundex_match=soundex_match,
                    metaphone_match=metaphone_match,
                    cologne_match=cologne_match,
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
            similar_brands=matches[:int(TOP_MATCHES)],
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
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='Similarity threshold')
    parser.add_argument('--markets', choices=['en', 'de', 'en_de'], default=DEFAULT_MARKETS,
                        help='Target market(s): en, de, or en_de (default)')

    args = parser.parse_args()

    checker = SimilarityChecker(markets=args.markets)

    for name in args.names:
        result = checker.check(name, args.threshold)

        print(f"\n{'='*50}")
        print(f"Name: {name}")
        print(f"Markets: {args.markets}")
        print(f"Safe: {'Yes' if result.is_safe else 'NO - Similar to existing brands!'}")
        print(f"Highest similarity: {result.highest_similarity:.2f}")

        # Show phonetic codes for the input name
        print(f"\nPhonetic codes for '{name}':")
        print(f"  Soundex:  {soundex(name)}")
        print(f"  Metaphone: {metaphone(name)}")
        print(f"  Cologne:  {cologne_phonetics(name)}")

        if result.similar_brands:
            print("\nSimilar brands found:")
            for m in result.similar_brands:
                flags = []
                if m.soundex_match:
                    flags.append("Soundex")
                if m.metaphone_match:
                    flags.append("Metaphone")
                if m.cologne_match:
                    flags.append("Cologne")
                flags.append(f"Sim:{m.text_similarity:.2f}")
                print(f"  - {m.known_brand}: [{', '.join(flags)}]")
