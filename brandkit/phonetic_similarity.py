#!/usr/bin/env python3
"""
Phonetic Similarity Module for Trademark Analysis
===================================================

Implements multiple algorithms for measuring phonetic similarity between brand names
and existing trademarks, as recommended for trademark likelihood of confusion analysis.

Algorithms:
- Soundex: Classic phonetic hashing (USPTO standard)
- Double Metaphone: Advanced phonetic encoding
- Normalized Levenshtein: Character-level edit distance
- Phonetic Edit Distance: Sound-based distance

For trademark purposes, phonetic similarity is critical because marks that
"sound alike" when spoken can cause consumer confusion even if spelled differently.

Reference:
- In re E.I. Du Pont de Nemours & Co., 476 F.2d 1357 (CCPA 1973)
- "DuPont factors" include similarity of sound, appearance, and meaning
"""

import re
from functools import lru_cache
from typing import Tuple, Optional


# =============================================================================
# Soundex (American Soundex - USPTO Standard)
# =============================================================================

def soundex(name: str) -> str:
    """
    Compute American Soundex code for a name.

    Soundex is the traditional algorithm used by USPTO for phonetic comparison.
    Names that sound similar will have the same Soundex code.

    Rules:
    1. Keep first letter
    2. Replace consonants with digits (B,F,P,V=1; C,G,J,K,Q,S,X,Z=2; D,T=3; L=4; M,N=5; R=6)
    3. Remove vowels and H,W,Y
    4. Remove duplicate adjacent digits
    5. Pad/truncate to 4 characters

    Args:
        name: Brand name to encode

    Returns:
        4-character Soundex code (letter + 3 digits)

    Examples:
        >>> soundex("Robert")
        'R163'
        >>> soundex("Rupert")
        'R163'
    """
    if not name:
        return "0000"

    # Normalize: uppercase and remove non-alpha
    name = re.sub(r'[^A-Za-z]', '', name.upper())
    if not name:
        return "0000"

    # Soundex mapping
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
        # A, E, I, O, U, H, W, Y are dropped (mapped to '')
    }

    first_letter = name[0]

    # Encode remaining letters
    encoded = []
    prev_code = mapping.get(first_letter, '')

    for char in name[1:]:
        code = mapping.get(char, '')
        if code and code != prev_code:  # Skip duplicates and non-coded
            encoded.append(code)
        # H and W don't separate identical codes
        if char not in ('H', 'W'):
            prev_code = code if code else prev_code

    # Build result: first letter + 3 digits (padded with zeros)
    result = first_letter + ''.join(encoded)[:3]
    return result.ljust(4, '0')


def soundex_similarity(name1: str, name2: str) -> float:
    """
    Compute similarity score based on Soundex codes.

    Returns:
        1.0 if Soundex codes match exactly
        0.75 if first 3 characters match
        0.5 if first 2 characters match
        0.25 if first letter matches
        0.0 otherwise
    """
    s1 = soundex(name1)
    s2 = soundex(name2)

    if s1 == s2:
        return 1.0
    elif s1[:3] == s2[:3]:
        return 0.75
    elif s1[:2] == s2[:2]:
        return 0.5
    elif s1[0] == s2[0]:
        return 0.25
    else:
        return 0.0


# =============================================================================
# Double Metaphone (More Sophisticated)
# =============================================================================

@lru_cache(maxsize=10000)
def double_metaphone(name: str) -> Tuple[str, str]:
    """
    Compute Double Metaphone encoding.

    Returns two codes: primary and alternate, accounting for different
    pronunciations of the same spelling (e.g., "Schmidt" vs "Smith").

    More accurate than Soundex for trademark comparison.

    Args:
        name: Brand name to encode

    Returns:
        Tuple of (primary_code, alternate_code)
    """
    if not name:
        return ('', '')

    # Normalize
    name = name.upper()
    current = 0
    length = len(name)
    primary = []
    secondary = []

    # Skip certain starting patterns
    if name[:2] in ('GN', 'KN', 'PN', 'WR', 'PS'):
        current = 1

    # Handle initial X as S
    if name[0] == 'X':
        primary.append('S')
        secondary.append('S')
        current = 1

    while current < length:
        char = name[current]

        if char in 'AEIOU':
            # Vowels at start contribute A
            if current == 0:
                primary.append('A')
                secondary.append('A')
            current += 1

        elif char == 'B':
            primary.append('P')
            secondary.append('P')
            current += 2 if current + 1 < length and name[current + 1] == 'B' else 1

        elif char == 'C':
            # Various C sounds
            if current > 0 and name[current-1:current+2] == 'ACH':
                primary.append('K')
                secondary.append('K')
                current += 2
            elif current == 0 and name[:5] == 'CAESAR':
                primary.append('S')
                secondary.append('S')
                current += 2
            elif name[current:current+2] == 'CH':
                primary.append('X')  # Germanic CH
                secondary.append('K')
                current += 2
            elif name[current:current+2] == 'CK':
                primary.append('K')
                secondary.append('K')
                current += 2
            elif name[current:current+2] in ('CI', 'CE', 'CY'):
                primary.append('S')
                secondary.append('S')
                current += 2
            else:
                primary.append('K')
                secondary.append('K')
                current += 2 if name[current:current+2] in ('CC', 'CQ', 'CG') else 1

        elif char == 'D':
            if name[current:current+2] == 'DG':
                if name[current+2:current+3] in 'IEY':
                    primary.append('J')
                    secondary.append('J')
                    current += 3
                else:
                    primary.append('TK')
                    secondary.append('TK')
                    current += 2
            else:
                primary.append('T')
                secondary.append('T')
                current += 2 if name[current:current+2] in ('DT', 'DD') else 1

        elif char == 'F':
            primary.append('F')
            secondary.append('F')
            current += 2 if current + 1 < length and name[current + 1] == 'F' else 1

        elif char == 'G':
            if current + 1 < length and name[current + 1] == 'H':
                if current > 0 and name[current-1] not in 'AEIOU':
                    current += 2
                else:
                    primary.append('K')
                    secondary.append('K')
                    current += 2
            elif name[current:current+2] == 'GN':
                primary.append('N')
                secondary.append('KN')
                current += 2
            elif name[current:current+2] in ('GI', 'GE', 'GY'):
                primary.append('J')
                secondary.append('K')
                current += 1
            else:
                primary.append('K')
                secondary.append('K')
                current += 2 if current + 1 < length and name[current + 1] == 'G' else 1

        elif char == 'H':
            # Only voiced between vowels
            if current + 1 < length and name[current + 1] in 'AEIOU':
                if current == 0 or name[current-1] in 'AEIOU':
                    primary.append('H')
                    secondary.append('H')
            current += 1

        elif char == 'J':
            primary.append('J')
            secondary.append('J')
            current += 2 if current + 1 < length and name[current + 1] == 'J' else 1

        elif char == 'K':
            primary.append('K')
            secondary.append('K')
            current += 2 if current + 1 < length and name[current + 1] == 'K' else 1

        elif char == 'L':
            primary.append('L')
            secondary.append('L')
            current += 2 if current + 1 < length and name[current + 1] == 'L' else 1

        elif char == 'M':
            primary.append('M')
            secondary.append('M')
            current += 2 if current + 1 < length and name[current + 1] == 'M' else 1

        elif char == 'N':
            primary.append('N')
            secondary.append('N')
            current += 2 if current + 1 < length and name[current + 1] == 'N' else 1

        elif char == 'P':
            if current + 1 < length and name[current + 1] == 'H':
                primary.append('F')
                secondary.append('F')
                current += 2
            else:
                primary.append('P')
                secondary.append('P')
                current += 2 if current + 1 < length and name[current + 1] in 'PB' else 1

        elif char == 'Q':
            primary.append('K')
            secondary.append('K')
            current += 2 if current + 1 < length and name[current + 1] == 'Q' else 1

        elif char == 'R':
            primary.append('R')
            secondary.append('R')
            current += 2 if current + 1 < length and name[current + 1] == 'R' else 1

        elif char == 'S':
            if name[current:current+2] == 'SH':
                primary.append('X')
                secondary.append('X')
                current += 2
            elif name[current:current+3] == 'SIO' or name[current:current+3] == 'SIA':
                primary.append('S')
                secondary.append('X')
                current += 3
            else:
                primary.append('S')
                secondary.append('S')
                current += 2 if current + 1 < length and name[current + 1] in 'SZ' else 1

        elif char == 'T':
            if name[current:current+3] == 'TIO' or name[current:current+3] == 'TIA':
                primary.append('X')
                secondary.append('X')
                current += 3
            elif name[current:current+2] == 'TH':
                primary.append('0')  # TH sound
                secondary.append('T')
                current += 2
            else:
                primary.append('T')
                secondary.append('T')
                current += 2 if current + 1 < length and name[current + 1] in 'TD' else 1

        elif char == 'V':
            primary.append('F')
            secondary.append('F')
            current += 2 if current + 1 < length and name[current + 1] == 'V' else 1

        elif char == 'W':
            if current + 1 < length and name[current + 1] in 'AEIOU':
                primary.append('A')
                secondary.append('A')
            current += 1

        elif char == 'X':
            primary.append('KS')
            secondary.append('KS')
            current += 2 if current + 1 < length and name[current + 1] == 'X' else 1

        elif char == 'Y':
            if current + 1 < length and name[current + 1] in 'AEIOU':
                primary.append('A')
                secondary.append('A')
            current += 1

        elif char == 'Z':
            primary.append('S')
            secondary.append('S')
            current += 2 if current + 1 < length and name[current + 1] == 'Z' else 1

        else:
            current += 1

    return (''.join(primary)[:4], ''.join(secondary)[:4])


def metaphone_similarity(name1: str, name2: str) -> float:
    """
    Compute similarity based on Double Metaphone codes.

    Compares both primary and secondary codes for each name.

    Returns:
        1.0 if any codes match exactly
        0.5-0.9 based on partial matches
        0.0 if no similarity
    """
    m1_prim, m1_sec = double_metaphone(name1)
    m2_prim, m2_sec = double_metaphone(name2)

    # Check all combinations
    matches = []
    for code1 in (m1_prim, m1_sec):
        for code2 in (m2_prim, m2_sec):
            if code1 and code2:
                if code1 == code2:
                    matches.append(1.0)
                elif code1[:3] == code2[:3]:
                    matches.append(0.8)
                elif code1[:2] == code2[:2]:
                    matches.append(0.6)
                elif code1[0] == code2[0]:
                    matches.append(0.3)

    return max(matches) if matches else 0.0


# =============================================================================
# Levenshtein Distance
# =============================================================================

@lru_cache(maxsize=10000)
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    Counts minimum number of single-character edits (insertions,
    deletions, substitutions) to transform s1 into s2.

    Args:
        s1, s2: Strings to compare

    Returns:
        Edit distance (0 = identical)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    s1_lower = s1.lower()
    s2_lower = s2.lower()

    previous_row = range(len(s2_lower) + 1)

    for i, c1 in enumerate(s1_lower):
        current_row = [i + 1]
        for j, c2 in enumerate(s2_lower):
            # Cost is 0 if characters match
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalized_levenshtein(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity (0 to 1).

    Returns:
        1.0 = identical
        0.0 = completely different
    """
    if not s1 and not s2:
        return 1.0

    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0

    distance = levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)


# =============================================================================
# Combined Phonetic Similarity Score
# =============================================================================

def compute_phonetic_similarity(name1: str, name2: str) -> float:
    """
    Compute overall phonetic similarity using multiple algorithms.

    This is the main function for trademark analysis. It combines:
    - Soundex similarity (USPTO standard)
    - Double Metaphone similarity (pronunciation-focused)
    - Normalized Levenshtein (visual similarity)

    The scores are weighted to emphasize phonetic similarity over
    visual similarity, as required for trademark confusion analysis.

    Args:
        name1: First brand name
        name2: Second brand name (trademark)

    Returns:
        Combined similarity score (0.0 to 1.0)
        - > 0.8: HIGH risk - likely confusion
        - > 0.6: MEDIUM risk - possible confusion
        - > 0.4: LOW risk - some similarity
        - <= 0.4: MINIMAL risk
    """
    # Individual scores
    sndx = soundex_similarity(name1, name2)
    meta = metaphone_similarity(name1, name2)
    lev = normalized_levenshtein(name1, name2)

    # Weight phonetic algorithms more heavily than visual
    # Soundex: 35%, Metaphone: 40%, Levenshtein: 25%
    combined = (sndx * 0.35) + (meta * 0.40) + (lev * 0.25)

    # Boost if exact case-insensitive match
    if name1.lower() == name2.lower():
        return 1.0

    return round(combined, 3)


# =============================================================================
# Risk Level Calculation
# =============================================================================

# Status to risk level mapping based on IP law principles
STATUS_RISK_MAP = {
    # Active/registered marks - highest risk
    'LIVE': 'HIGH',
    'REGISTERED': 'HIGH',
    'ACTIVE': 'HIGH',
    'LIVE/REGISTERED': 'HIGH',
    'REGISTRATION': 'HIGH',

    # Pending applications - medium risk (could still register)
    'PENDING': 'MEDIUM',
    'PENDING REGISTRATION': 'MEDIUM',
    'FILED': 'MEDIUM',
    'PUBLISHED': 'MEDIUM',
    'UNDER EXAMINATION': 'MEDIUM',
    'OPPOSITION PENDING': 'MEDIUM',

    # Abandoned/dead marks - lower risk but not zero
    'DEAD': 'LOW',
    'ABANDONED': 'LOW',
    'CANCELLED': 'LOW',
    'EXPIRED': 'LOW',
    'WITHDRAWN': 'LOW',
    'REFUSED': 'LOW',

    # Unknown status
    None: 'UNKNOWN',
    '': 'UNKNOWN',
}


def calculate_risk_level(match_status: str,
                        is_exact: bool = False,
                        phonetic_similarity: float = None,
                        classes_overlap: bool = False) -> str:
    """
    Calculate risk level for a trademark conflict.

    Based on IP law principles:
    - LIVE/REGISTERED marks in same class = highest risk
    - PENDING applications = still risky, could register
    - DEAD marks = lower risk but can inform search
    - Phonetic similarity amplifies risk
    - Class overlap is critical

    Args:
        match_status: Trademark status (LIVE, PENDING, DEAD, etc.)
        is_exact: True if exact name match
        phonetic_similarity: Phonetic similarity score (0-1)
        classes_overlap: True if Nice classes overlap

    Returns:
        Risk level: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', or 'UNKNOWN'
    """
    # Normalize status
    status_upper = (match_status or '').upper().strip()
    base_risk = STATUS_RISK_MAP.get(status_upper, 'UNKNOWN')

    # Exact matches are always critical if mark is active
    if is_exact and base_risk == 'HIGH':
        return 'CRITICAL'

    # High phonetic similarity + active mark = critical
    if phonetic_similarity and phonetic_similarity > 0.8 and base_risk == 'HIGH':
        return 'CRITICAL'

    # Class overlap elevates risk
    if classes_overlap and base_risk == 'MEDIUM':
        return 'HIGH'

    # Very high phonetic similarity elevates even pending/dead marks
    if phonetic_similarity and phonetic_similarity > 0.9:
        if base_risk == 'LOW':
            return 'MEDIUM'
        elif base_risk == 'MEDIUM':
            return 'HIGH'

    return base_risk


# =============================================================================
# Batch Processing for Efficiency
# =============================================================================

def compute_phonetic_similarities_batch(query_name: str,
                                        trademark_names: list) -> list:
    """
    Compute phonetic similarity for multiple trademarks efficiently.

    Args:
        query_name: The brand name being checked
        trademark_names: List of existing trademark names

    Returns:
        List of (trademark_name, similarity_score) tuples, sorted by score
    """
    results = []
    for tm_name in trademark_names:
        score = compute_phonetic_similarity(query_name, tm_name)
        results.append((tm_name, score))

    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python phonetic_similarity.py <name1> <name2>")
        print("\nExample: python phonetic_similarity.py Voltix Voltic")
        sys.exit(1)

    name1 = sys.argv[1]
    name2 = sys.argv[2]

    print(f"\nComparing: '{name1}' vs '{name2}'")
    print("=" * 50)

    print(f"\nSoundex:")
    print(f"  {name1}: {soundex(name1)}")
    print(f"  {name2}: {soundex(name2)}")
    print(f"  Similarity: {soundex_similarity(name1, name2):.2f}")

    print(f"\nDouble Metaphone:")
    m1 = double_metaphone(name1)
    m2 = double_metaphone(name2)
    print(f"  {name1}: {m1}")
    print(f"  {name2}: {m2}")
    print(f"  Similarity: {metaphone_similarity(name1, name2):.2f}")

    print(f"\nLevenshtein:")
    print(f"  Edit distance: {levenshtein_distance(name1, name2)}")
    print(f"  Normalized: {normalized_levenshtein(name1, name2):.2f}")

    print(f"\nCombined Phonetic Similarity: {compute_phonetic_similarity(name1, name2):.3f}")

    # Demonstrate risk level
    risk = calculate_risk_level(
        match_status='LIVE',
        is_exact=(name1.lower() == name2.lower()),
        phonetic_similarity=compute_phonetic_similarity(name1, name2)
    )
    print(f"Risk Level (if LIVE mark): {risk}")
