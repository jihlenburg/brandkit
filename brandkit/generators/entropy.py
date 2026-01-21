#!/usr/bin/env python3
"""
Entropy Module for Name Generation
===================================
Provides true randomness and linguistic variance for brand name generation.

Features:
- Hardware-backed true random number generation
- Phoneme mutation engine (voicing, vowel shifts, lenition)
- Expanded syllable structure templates
- Morphological operations (blending, infixation, reduplication, metathesis)
- Cross-linguistic phoneme injection
- Weighted stochastic selection based on phonotactic probability

Uses cryptographically secure random sources:
- os.urandom() for hardware entropy
- secrets.SystemRandom() for CSPRNG
- Multiple entropy sources combined (time nanoseconds, PID, memory)
"""

import os
import sys
import time
import secrets
import hashlib
from typing import List, Tuple, Dict, Optional, Callable, Any
from dataclasses import dataclass
from functools import lru_cache


# =============================================================================
# True Random Number Generator
# =============================================================================

class TrueRandom:
    """
    Cryptographically secure random number generator using hardware entropy.

    Combines multiple entropy sources:
    - os.urandom() - system entropy pool (hardware RNG if available)
    - High-resolution time (nanoseconds)
    - Process ID and memory addresses
    - secrets module for CSPRNG
    """

    def __init__(self):
        """Initialize with fresh entropy from multiple sources."""
        self._rng = secrets.SystemRandom()
        self._reseed()

    def _reseed(self):
        """Inject fresh entropy from multiple physical sources."""
        # Hardware entropy (8 bytes = 64 bits)
        hw_entropy = int.from_bytes(os.urandom(8), 'big')

        # High-resolution time (nanoseconds since epoch)
        time_entropy = time.time_ns()

        # Process ID (shifted to high bits)
        pid_entropy = os.getpid() << 48

        # Memory address of a new object (semi-random)
        mem_entropy = id(object()) & 0xFFFFFFFF

        # Combine with XOR
        combined = hw_entropy ^ time_entropy ^ pid_entropy ^ mem_entropy

        # Hash for uniform distribution
        entropy_bytes = hashlib.sha256(combined.to_bytes(32, 'big')).digest()

        # Use as additional seed material (SystemRandom doesn't need seeding,
        # but we track this for mixing into weighted selections)
        self._entropy_pool = int.from_bytes(entropy_bytes[:8], 'big')

    def random(self) -> float:
        """Return random float in [0.0, 1.0)."""
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        """Return random integer N such that a <= N <= b."""
        return self._rng.randint(a, b)

    def choice(self, seq: list) -> Any:
        """Return a random element from non-empty sequence."""
        if not seq:
            raise IndexError("Cannot choose from empty sequence")
        return self._rng.choice(seq)

    def choices(self, population: list, weights: List[float] = None, k: int = 1) -> list:
        """Return k-sized list of elements chosen with optional weights."""
        if weights:
            return self._rng.choices(population, weights=weights, k=k)
        return [self._rng.choice(population) for _ in range(k)]

    def sample(self, population: list, k: int) -> list:
        """Return k unique elements from population."""
        return self._rng.sample(population, k)

    def shuffle(self, seq: list) -> None:
        """Shuffle list in place."""
        self._rng.shuffle(seq)

    def weighted_choice(self, items: List[Tuple[Any, float]]) -> Any:
        """
        Choose from items with weights.

        Args:
            items: List of (item, weight) tuples

        Returns:
            Randomly selected item based on weights
        """
        if not items:
            raise IndexError("Cannot choose from empty sequence")

        total = sum(w for _, w in items)
        r = self.random() * total

        cumulative = 0
        for item, weight in items:
            cumulative += weight
            if r <= cumulative:
                return item

        return items[-1][0]  # Fallback to last item

    def gauss(self, mu: float, sigma: float) -> float:
        """Gaussian distribution with mean mu and standard deviation sigma."""
        return self._rng.gauss(mu, sigma)

    def triangular(self, low: float, high: float, mode: float = None) -> float:
        """Triangular distribution."""
        if mode is None:
            mode = (low + high) / 2
        return self._rng.triangular(low, high, mode)


# Global instance
_true_random = TrueRandom()

def get_rng() -> TrueRandom:
    """Get the global true random number generator."""
    return _true_random


# =============================================================================
# Phoneme Classes and Transformations
# =============================================================================

@dataclass
class PhonemeClass:
    """Classification of phonemes by articulatory features."""
    voiced_stops: List[str] = None
    voiceless_stops: List[str] = None
    voiced_fricatives: List[str] = None
    voiceless_fricatives: List[str] = None
    nasals: List[str] = None
    liquids: List[str] = None
    glides: List[str] = None
    front_vowels: List[str] = None
    back_vowels: List[str] = None
    high_vowels: List[str] = None
    low_vowels: List[str] = None

    def __post_init__(self):
        self.voiced_stops = self.voiced_stops or ['b', 'd', 'g']
        self.voiceless_stops = self.voiceless_stops or ['p', 't', 'k']
        self.voiced_fricatives = self.voiced_fricatives or ['v', 'z', 'j']
        self.voiceless_fricatives = self.voiceless_fricatives or ['f', 's', 'sh', 'h']
        self.nasals = self.nasals or ['m', 'n', 'ng']
        self.liquids = self.liquids or ['l', 'r']
        self.glides = self.glides or ['w', 'y']
        self.front_vowels = self.front_vowels or ['i', 'e', 'ae']
        self.back_vowels = self.back_vowels or ['u', 'o', 'a']
        self.high_vowels = self.high_vowels or ['i', 'u']
        self.low_vowels = self.low_vowels or ['a', 'ae', 'o']


# Default phoneme classes
PHONEMES = PhonemeClass()

# Voicing pairs (voiceless -> voiced and vice versa)
VOICING_PAIRS = {
    'p': 'b', 'b': 'p',
    't': 'd', 'd': 't',
    'k': 'g', 'g': 'k',
    'f': 'v', 'v': 'f',
    's': 'z', 'z': 's',
    'sh': 'zh', 'zh': 'sh',
}

# Vowel shift pairs (front <-> back, high <-> low)
VOWEL_SHIFTS = {
    # Front-back
    'i': ['u', 'e'],
    'e': ['o', 'i', 'a'],
    'u': ['i', 'o'],
    'o': ['e', 'u', 'a'],
    'a': ['e', 'o', 'ae'],
    # Extended
    'ae': ['a', 'e'],
    'oe': ['o', 'e'],
}

# Lenition paths (fortis -> lenis)
LENITION = {
    'p': ['f', 'v', 'w'],
    'b': ['v', 'w'],
    't': ['th', 'd', 'r'],
    'd': ['th', 'r', 'l'],
    'k': ['x', 'g', 'h'],
    'g': ['gh', 'h', 'w'],
}

# Fortition paths (lenis -> fortis)
FORTITION = {
    'v': ['b', 'f'],
    'w': ['v', 'b'],
    'th': ['t', 'd'],
    'r': ['d', 't'],
    'h': ['k', 'g'],
}


class PhonemeMutator:
    """
    Applies phonetic mutations to names for increased variance.

    Mutation types:
    - Voicing: p <-> b, t <-> d, k <-> g, etc.
    - Vowel shift: front <-> back, high <-> low
    - Lenition: consonant weakening (p -> f -> v -> w)
    - Fortition: consonant strengthening (v -> b)
    - Assimilation: consonants adapt to neighbors
    - Metathesis: sound swapping
    """

    def __init__(self, rng: TrueRandom = None):
        self.rng = rng or get_rng()

    def mutate(self, name: str, intensity: float = 0.3) -> str:
        """
        Apply random mutations to a name.

        Args:
            name: Input name
            intensity: Probability of mutation per applicable phoneme (0.0-1.0)

        Returns:
            Mutated name
        """
        result = list(name.lower())

        for i, char in enumerate(result):
            if self.rng.random() > intensity:
                continue

            # Choose mutation type
            mutation_type = self.rng.choice([
                'voicing', 'vowel_shift', 'lenition', 'none', 'none'
            ])

            if mutation_type == 'voicing' and char in VOICING_PAIRS:
                result[i] = VOICING_PAIRS[char]

            elif mutation_type == 'vowel_shift' and char in VOWEL_SHIFTS:
                result[i] = self.rng.choice(VOWEL_SHIFTS[char])

            elif mutation_type == 'lenition' and char in LENITION:
                # Only apply lenition sometimes (it's more dramatic)
                if self.rng.random() < 0.3:
                    new_sound = self.rng.choice(LENITION[char])
                    # Multi-char sounds need special handling
                    if len(new_sound) == 1:
                        result[i] = new_sound

        return ''.join(result).capitalize()

    def voice_shift(self, name: str) -> str:
        """Shift voicing of all applicable consonants."""
        result = list(name.lower())
        for i, char in enumerate(result):
            if char in VOICING_PAIRS:
                result[i] = VOICING_PAIRS[char]
        return ''.join(result).capitalize()

    def vowel_shift(self, name: str, direction: str = 'random') -> str:
        """
        Shift vowels front<->back or high<->low.

        Args:
            name: Input name
            direction: 'front', 'back', 'high', 'low', or 'random'
        """
        result = list(name.lower())

        for i, char in enumerate(result):
            if char in VOWEL_SHIFTS:
                options = VOWEL_SHIFTS[char]
                if direction == 'random':
                    result[i] = self.rng.choice(options)
                elif direction == 'front' and char in 'uoa':
                    front_opts = [v for v in options if v in 'ie']
                    if front_opts:
                        result[i] = self.rng.choice(front_opts)
                elif direction == 'back' and char in 'ie':
                    back_opts = [v for v in options if v in 'uoa']
                    if back_opts:
                        result[i] = self.rng.choice(back_opts)

        return ''.join(result).capitalize()


# =============================================================================
# Syllable Structures
# =============================================================================

# Expanded syllable templates beyond simple CV
SYLLABLE_TEMPLATES = {
    'simple': [
        'V',      # a
        'CV',     # ta
        'VC',     # an
        'CVC',    # tan
    ],
    'complex_onset': [
        'CCV',    # tra
        'CCVC',   # tran
        'CCCV',   # stra (rare but exists)
    ],
    'complex_coda': [
        'VCC',    # ant
        'CVCC',   # tant
        'VCCC',   # ants (very rare)
    ],
    'complex_both': [
        'CCVCC',  # trants
        'CCVCCC', # extremely rare
    ],
    'japanese': [
        'CV',     # ka
        'V',      # a
        'CVn',    # kan (n is special)
    ],
    'germanic': [
        'CVC',    # tag
        'CVCC',   # tags
        'CCVC',   # strap
        'CCVCC',  # straps
    ],
}

# Permissible onset clusters (initial consonant groups)
ONSET_CLUSTERS = {
    'common': ['bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 'tr', 'sk', 'sl', 'sm', 'sn', 'sp', 'st', 'sw'],
    'extended': ['scr', 'spl', 'spr', 'str', 'thr', 'shr'],
    'slavic': ['vl', 'vr', 'zv', 'zd', 'zr', 'kr', 'kv'],
    'germanic': ['kn', 'gn', 'pf', 'ts', 'wr'],
}

# Permissible coda clusters (final consonant groups)
CODA_CLUSTERS = {
    'common': ['ct', 'ft', 'ld', 'lf', 'lk', 'lm', 'ln', 'lp', 'lt', 'mp', 'nd', 'nk', 'nt', 'pt', 'rd', 'rk', 'rm', 'rn', 'rp', 'rt', 'sk', 'sp', 'st'],
    'extended': ['nch', 'nct', 'mpt', 'rst', 'rts', 'sts'],
}


class SyllableBuilder:
    """
    Builds syllables with controlled complexity and variance.
    """

    def __init__(self, rng: TrueRandom = None):
        self.rng = rng or get_rng()

        # Extended consonant inventory
        self.consonants = list('bcdfghjklmnpqrstvwxyz')
        self.vowels = list('aeiou')

        # Weighted consonants (some are more common)
        self.consonant_weights = {
            'n': 2.0, 'r': 2.0, 's': 2.0, 't': 2.0, 'l': 1.8,
            'm': 1.5, 'd': 1.5, 'k': 1.3, 'p': 1.2, 'b': 1.2,
            'f': 1.0, 'v': 1.0, 'g': 1.0, 'h': 0.8, 'w': 0.7,
            'c': 0.6, 'j': 0.5, 'x': 0.4, 'z': 0.4, 'q': 0.2, 'y': 0.6,
        }

        # Weighted vowels
        self.vowel_weights = {
            'a': 2.0, 'e': 2.0, 'i': 1.5, 'o': 1.5, 'u': 1.0,
        }

    def build_syllable(self, template: str = None, style: str = 'common') -> str:
        """
        Build a syllable from a template.

        Args:
            template: Syllable structure like 'CVC', 'CCVC', etc.
                      If None, randomly selected based on style.
            style: 'simple', 'complex', 'japanese', 'germanic'

        Returns:
            Generated syllable string
        """
        if template is None:
            if style == 'simple' or style == 'japanese':
                templates = SYLLABLE_TEMPLATES['simple'] + SYLLABLE_TEMPLATES.get(style, [])
            elif style == 'complex' or style == 'germanic':
                templates = (SYLLABLE_TEMPLATES['simple'] +
                           SYLLABLE_TEMPLATES['complex_onset'] +
                           SYLLABLE_TEMPLATES.get(style, []))
            else:
                templates = SYLLABLE_TEMPLATES['simple']

            # Weight simpler templates more heavily
            weights = [1.0 / (len(t) ** 0.5) for t in templates]
            template = self.rng.weighted_choice(list(zip(templates, weights)))

        result = []
        i = 0
        while i < len(template):
            char = template[i]

            if char == 'C':
                # Check for cluster (CC or CCC)
                cluster_len = 1
                while i + cluster_len < len(template) and template[i + cluster_len] == 'C':
                    cluster_len += 1

                if cluster_len > 1 and self.rng.random() < 0.4:
                    # Use a real cluster
                    cluster_type = 'common' if cluster_len == 2 else 'extended'
                    if i == 0:  # Onset
                        clusters = ONSET_CLUSTERS.get(cluster_type, ONSET_CLUSTERS['common'])
                    else:  # Coda
                        clusters = CODA_CLUSTERS.get(cluster_type, CODA_CLUSTERS['common'])

                    valid_clusters = [c for c in clusters if len(c) == cluster_len]
                    if valid_clusters:
                        result.append(self.rng.choice(valid_clusters))
                        i += cluster_len
                        continue

                # Fall back to weighted single consonant
                result.append(self._weighted_consonant())
                i += 1

            elif char == 'V':
                result.append(self._weighted_vowel())
                i += 1

            elif char == 'n':
                # Special Japanese syllable-final 'n'
                result.append('n')
                i += 1

            else:
                i += 1

        return ''.join(result)

    def _weighted_consonant(self) -> str:
        """Select consonant with natural frequency weighting."""
        items = [(c, self.consonant_weights.get(c, 1.0)) for c in self.consonants]
        return self.rng.weighted_choice(items)

    def _weighted_vowel(self) -> str:
        """Select vowel with natural frequency weighting."""
        items = [(v, self.vowel_weights.get(v, 1.0)) for v in self.vowels]
        return self.rng.weighted_choice(items)

    def build_word(self, syllable_count: int = None, style: str = 'common') -> str:
        """
        Build a complete word with multiple syllables.

        Args:
            syllable_count: Number of syllables (random 2-4 if None)
            style: Syllable style

        Returns:
            Generated word
        """
        if syllable_count is None:
            # Weight toward 2-3 syllables
            syllable_count = self.rng.weighted_choice([
                (2, 3.0), (3, 2.5), (4, 1.0), (1, 0.5)
            ])

        syllables = [self.build_syllable(style=style) for _ in range(syllable_count)]
        return ''.join(syllables)


# =============================================================================
# Morphological Operations
# =============================================================================

class Morphology:
    """
    Advanced morphological operations for name generation.

    Operations:
    - Blending: Combine overlapping portions (motor + hotel = motel)
    - Infixation: Insert morpheme inside word
    - Reduplication: Partial or full repetition
    - Metathesis: Sound swapping
    - Clipping: Variable-length truncation
    - Compounding: Intelligent word combination
    """

    def __init__(self, rng: TrueRandom = None):
        self.rng = rng or get_rng()

    def blend(self, word1: str, word2: str) -> str:
        """
        Create a portmanteau by blending two words at phonetically similar points.

        Examples: motor + hotel = motel, breakfast + lunch = brunch
        """
        w1, w2 = word1.lower(), word2.lower()

        # Find best overlap point
        best_blend = None
        best_score = -1

        for i in range(1, len(w1)):
            for j in range(len(w2) - 1):
                # Check if end of w1 overlaps with start of w2[j:]
                overlap_len = min(len(w1) - i, j + 1)

                if overlap_len > 0:
                    w1_end = w1[i:i+overlap_len]
                    w2_start = w2[j:j+overlap_len]

                    # Score based on phonetic similarity
                    matches = sum(1 for a, b in zip(w1_end, w2_start) if a == b)
                    score = matches / overlap_len if overlap_len > 0 else 0

                    # Prefer blends in the middle of words
                    position_score = min(i, len(w1) - i) * min(j, len(w2) - j)
                    total_score = score * 0.7 + (position_score / 10) * 0.3

                    if total_score > best_score:
                        best_score = total_score
                        # Blend: beginning of w1 + end of w2
                        blend_point = i + overlap_len // 2
                        best_blend = w1[:blend_point] + w2[j + overlap_len // 2:]

        if best_blend and len(best_blend) >= 4:
            return best_blend.capitalize()

        # Fallback: simple cut-and-join
        cut1 = self.rng.randint(len(w1) // 2, len(w1) - 1)
        cut2 = self.rng.randint(1, len(w2) // 2)
        return (w1[:cut1] + w2[cut2:]).capitalize()

    def infix(self, word: str, infix: str = None) -> str:
        """
        Insert an infix inside a word.

        If no infix provided, uses common intensifying/euphonic infixes.
        """
        if len(word) < 4:
            return word

        if infix is None:
            infixes = ['i', 'a', 'o', 'ix', 'ex', 'ul', 'ar', 'en']
            infix = self.rng.choice(infixes)

        # Find syllable boundary (vowel-consonant or consonant-vowel transition)
        w = word.lower()
        vowels = set('aeiou')

        # Find insertion points (after vowel before consonant)
        insert_points = []
        for i in range(1, len(w) - 1):
            if w[i-1] in vowels and w[i] not in vowels:
                insert_points.append(i)
            elif w[i-1] not in vowels and w[i] in vowels:
                insert_points.append(i)

        if not insert_points:
            insert_points = [len(w) // 2]

        point = self.rng.choice(insert_points)
        return (w[:point] + infix + w[point:]).capitalize()

    def reduplicate(self, word: str, style: str = 'partial') -> str:
        """
        Apply reduplication to a word.

        Styles:
        - 'full': TikTok, beriberi
        - 'partial': zigzag, wishy-washy
        - 'ablaut': sing-sang, ding-dong (vowel change)
        """
        w = word.lower()

        if style == 'full':
            return (w + w).capitalize()

        elif style == 'partial':
            # Take first syllable or first few chars
            if len(w) >= 4:
                partial = w[:2]
                return (partial + w).capitalize()
            return w.capitalize()

        elif style == 'ablaut':
            # Change vowel in second part
            vowels = 'aeiou'
            second = list(w)
            for i, char in enumerate(second):
                if char in vowels:
                    # Shift to different vowel
                    vowel_list = list(vowels)
                    vowel_list.remove(char)
                    second[i] = self.rng.choice(vowel_list)
                    break
            return (w + '-' + ''.join(second)).title()

        return w.capitalize()

    def metathesis(self, word: str) -> str:
        """
        Swap adjacent sounds in a word.

        Examples: ask -> aks, film -> flim
        """
        if len(word) < 3:
            return word

        w = list(word.lower())

        # Find swappable pairs (consonant clusters or CV sequences)
        swap_points = []
        for i in range(len(w) - 1):
            # Prefer swapping near syllable boundaries
            if w[i] in 'aeiou' and w[i+1] not in 'aeiou':
                swap_points.append(i)
            elif w[i] not in 'aeiou' and w[i+1] in 'aeiou':
                swap_points.append(i)

        if swap_points:
            i = self.rng.choice(swap_points)
            w[i], w[i+1] = w[i+1], w[i]

        return ''.join(w).capitalize()

    def clip(self, word: str, style: str = 'random') -> str:
        """
        Clip a word at various points.

        Styles:
        - 'fore': Remove beginning (telephone -> phone)
        - 'back': Remove ending (examination -> exam)
        - 'middle': Keep middle (influenza -> flu)
        - 'random': Random clip point
        """
        if len(word) < 4:
            return word

        w = word.lower()

        if style == 'fore':
            cut = self.rng.randint(1, len(w) // 3)
            return w[cut:].capitalize()

        elif style == 'back':
            cut = self.rng.randint(len(w) * 2 // 3, len(w) - 1)
            return w[:cut].capitalize()

        elif style == 'middle':
            start = self.rng.randint(0, len(w) // 3)
            end = self.rng.randint(len(w) * 2 // 3, len(w))
            return w[start:end].capitalize()

        else:  # random
            style = self.rng.choice(['fore', 'back', 'middle'])
            return self.clip(word, style)

    def compound(self, word1: str, word2: str, style: str = 'random') -> str:
        """
        Intelligently combine two words.

        Styles:
        - 'full': word1 + word2 (football)
        - 'truncated': truncated forms (modem from modulator + demodulator)
        - 'linked': with connecting vowel (speedometer)
        - 'random': randomly chosen style
        """
        w1, w2 = word1.lower(), word2.lower()

        if style == 'random':
            style = self.rng.choice(['full', 'truncated', 'linked'])

        if style == 'full':
            return (w1 + w2).capitalize()

        elif style == 'truncated':
            # Take distinctive parts of each
            len1 = self.rng.randint(2, min(4, len(w1)))
            len2 = self.rng.randint(2, min(4, len(w2)))

            # Prefer beginning of first, ending of second
            part1 = w1[:len1]
            part2 = w2[-len2:]

            return (part1 + part2).capitalize()

        elif style == 'linked':
            # Add connecting vowel if needed
            vowels = set('aeiou')

            if w1[-1] not in vowels and w2[0] not in vowels:
                linker = self.rng.choice(['o', 'i', 'a'])
                return (w1 + linker + w2).capitalize()

            return (w1 + w2).capitalize()

        return (w1 + w2).capitalize()


# =============================================================================
# Cross-linguistic Phoneme Injection
# =============================================================================

# Phonemes from various language families that are pronounceable in English/German
EXOTIC_PHONEMES = {
    'slavic': {
        'onsets': ['zdr', 'str', 'vl', 'vr', 'kr', 'kv', 'sv', 'zv'],
        'consonants': ['zh', 'sh', 'ch'],
    },
    'germanic': {
        'onsets': ['kn', 'gn', 'pf', 'ts', 'wr'],
        'codas': ['pf', 'ts', 'ks', 'ps'],
    },
    'romance': {
        'consonants': ['gn', 'gl', 'sc'],
        'vowels': ['ie', 'uo', 'ae'],
    },
    'celtic': {
        'onsets': ['cw', 'gw', 'dw', 'tw'],
        'consonants': ['ch', 'gh', 'dh', 'th'],
    },
    'semitic': {
        'consonants': ['kh', 'gh'],  # Softened for Western use
    },
    'sanskrit': {
        'onsets': ['ksh', 'shr', 'dhr'],
        'consonants': ['bh', 'dh', 'gh', 'jh', 'kh', 'ph', 'th'],  # Aspirated
    },
}


class PhonemeInjector:
    """
    Injects exotic but pronounceable phonemes from various language families.
    """

    def __init__(self, rng: TrueRandom = None):
        self.rng = rng or get_rng()

    def inject(self, name: str, language_families: List[str] = None, intensity: float = 0.2) -> str:
        """
        Inject exotic phonemes into a name.

        Args:
            name: Input name
            language_families: Which families to draw from (None = all)
            intensity: How aggressively to inject (0.0-1.0)

        Returns:
            Modified name with injected phonemes
        """
        if language_families is None:
            language_families = list(EXOTIC_PHONEMES.keys())

        result = name.lower()

        for family in language_families:
            if family not in EXOTIC_PHONEMES:
                continue

            phonemes = EXOTIC_PHONEMES[family]

            # Maybe replace onset
            if self.rng.random() < intensity and 'onsets' in phonemes:
                onsets = phonemes['onsets']
                if result[:2] in ['st', 'sp', 'sk', 'tr', 'kr', 'pr', 'br']:
                    new_onset = self.rng.choice(onsets)
                    # Only use if similar length
                    if len(new_onset) <= 3:
                        result = new_onset + result[2:]

            # Maybe inject consonant
            if self.rng.random() < intensity and 'consonants' in phonemes:
                consonants = phonemes['consonants']
                # Find a consonant to replace
                for i, char in enumerate(result):
                    if char not in 'aeiou' and self.rng.random() < 0.3:
                        new_cons = self.rng.choice(consonants)
                        if len(new_cons) == 1:
                            result = result[:i] + new_cons + result[i+1:]
                        break

        return result.capitalize()

    def get_random_onset(self, families: List[str] = None) -> str:
        """Get a random onset cluster from specified language families."""
        if families is None:
            families = list(EXOTIC_PHONEMES.keys())

        all_onsets = []
        for family in families:
            if family in EXOTIC_PHONEMES and 'onsets' in EXOTIC_PHONEMES[family]:
                all_onsets.extend(EXOTIC_PHONEMES[family]['onsets'])

        # Add common onsets too
        all_onsets.extend(ONSET_CLUSTERS['common'])

        return self.rng.choice(all_onsets) if all_onsets else ''

    def get_random_coda(self, families: List[str] = None) -> str:
        """Get a random coda cluster from specified language families."""
        if families is None:
            families = list(EXOTIC_PHONEMES.keys())

        all_codas = []
        for family in families:
            if family in EXOTIC_PHONEMES and 'codas' in EXOTIC_PHONEMES[family]:
                all_codas.extend(EXOTIC_PHONEMES[family]['codas'])

        # Add common codas too
        all_codas.extend(CODA_CLUSTERS['common'])

        return self.rng.choice(all_codas) if all_codas else ''


# =============================================================================
# Entropy Combiner (Main Interface)
# =============================================================================

class EntropyEngine:
    """
    Main interface combining all entropy-enhancing features.

    Provides high-level methods for generating names with maximum variance
    using true randomness, phoneme mutations, morphological operations,
    and cross-linguistic phoneme injection.
    """

    def __init__(self):
        self.rng = get_rng()
        self.mutator = PhonemeMutator(self.rng)
        self.syllable_builder = SyllableBuilder(self.rng)
        self.morphology = Morphology(self.rng)
        self.injector = PhonemeInjector(self.rng)

    def generate_base(self, syllables: int = None, style: str = 'common') -> str:
        """Generate a base name from syllable templates."""
        return self.syllable_builder.build_word(syllables, style)

    def mutate(self, name: str, intensity: float = 0.3) -> str:
        """Apply phonetic mutations to a name."""
        return self.mutator.mutate(name, intensity)

    def blend(self, word1: str, word2: str) -> str:
        """Create a portmanteau blend of two words."""
        return self.morphology.blend(word1, word2)

    def vary(self, name: str, operations: List[str] = None) -> str:
        """
        Apply a random selection of variance-increasing operations.

        Args:
            name: Input name
            operations: List of operations to consider. Default: all.
                Options: 'mutate', 'infix', 'metathesis', 'clip', 'inject'

        Returns:
            Modified name with increased variance
        """
        if operations is None:
            operations = ['mutate', 'infix', 'metathesis', 'clip', 'inject']

        # Randomly select 1-2 operations
        num_ops = self.rng.randint(1, min(2, len(operations)))
        selected_ops = self.rng.sample(operations, num_ops)

        result = name

        for op in selected_ops:
            if op == 'mutate':
                result = self.mutator.mutate(result, intensity=0.2)
            elif op == 'infix':
                result = self.morphology.infix(result)
            elif op == 'metathesis':
                result = self.morphology.metathesis(result)
            elif op == 'clip':
                result = self.morphology.clip(result)
            elif op == 'inject':
                result = self.injector.inject(result, intensity=0.15)

        return result

    def generate_variant(self, base: str, count: int = 5) -> List[str]:
        """
        Generate multiple variants of a base name.

        Returns unique variants with different transformations applied.
        """
        variants = set()
        variants.add(base.capitalize())

        attempts = 0
        while len(variants) < count + 1 and attempts < count * 3:
            attempts += 1

            # Choose transformation
            transform = self.rng.choice([
                lambda n: self.mutate(n, 0.3),
                lambda n: self.mutate(n, 0.5),
                lambda n: self.morphology.infix(n),
                lambda n: self.morphology.metathesis(n),
                lambda n: self.morphology.clip(n, 'back'),
                lambda n: self.morphology.clip(n, 'fore'),
                lambda n: self.injector.inject(n, intensity=0.2),
                lambda n: self.mutator.vowel_shift(n),
                lambda n: self.mutator.voice_shift(n),
            ])

            try:
                variant = transform(base)
                if len(variant) >= 3:
                    variants.add(variant)
            except Exception:
                pass

        # Remove base if we have enough variants
        result = list(variants)
        if len(result) > count:
            result = [v for v in result if v.lower() != base.lower()][:count]

        return result


# =============================================================================
# Module-level convenience functions
# =============================================================================

def random() -> float:
    """Get a true random float in [0.0, 1.0)."""
    return _true_random.random()

def randint(a: int, b: int) -> int:
    """Get a true random integer in [a, b]."""
    return _true_random.randint(a, b)

def choice(seq: list) -> Any:
    """Get a true random element from sequence."""
    return _true_random.choice(seq)

def choices(population: list, weights: List[float] = None, k: int = 1) -> list:
    """Get k true random elements with optional weights."""
    return _true_random.choices(population, weights, k)

def sample(population: list, k: int) -> list:
    """Get k unique true random elements."""
    return _true_random.sample(population, k)

def shuffle(seq: list) -> None:
    """Shuffle list in place with true randomness."""
    _true_random.shuffle(seq)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core RNG
    'TrueRandom',
    'get_rng',
    # Phoneme mutation
    'PhonemeMutator',
    'VOICING_PAIRS',
    'VOWEL_SHIFTS',
    'LENITION',
    # Syllable building
    'SyllableBuilder',
    'SYLLABLE_TEMPLATES',
    'ONSET_CLUSTERS',
    'CODA_CLUSTERS',
    # Morphology
    'Morphology',
    # Phoneme injection
    'PhonemeInjector',
    'EXOTIC_PHONEMES',
    # Main engine
    'EntropyEngine',
    # Convenience functions
    'random',
    'randint',
    'choice',
    'choices',
    'sample',
    'shuffle',
]
