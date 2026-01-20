#!/usr/bin/env python3
"""
Cultural Generator Base Class
=============================
Base class for all cultural/linguistic name generators with advanced features:
- Sound symbolism (phonaesthetics)
- Cross-linguistic hazard checking
- Syllable stress/rhythm analysis
- Industry-specific generation
- Memorability scoring
- Cross-culture blending
- Archetype-driven generation
- Competitive differentiation
"""

import re
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set
from functools import lru_cache

import yaml
from pathlib import Path

# Phoneme config directory
PHONEMES_DIR = Path(__file__).parent / 'phonemes'


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GeneratedName:
    """A generated brand name with comprehensive metadata."""
    name: str
    roots_used: List[str] = field(default_factory=list)
    meaning_hints: List[str] = field(default_factory=list)
    score: float = 0.0
    method: str = ""
    culture: str = ""
    archetype: str = ""
    syllables: int = 0
    stress_pattern: str = ""
    hazards: List[Dict] = field(default_factory=list)
    phonaesthetics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HazardResult:
    """Result of hazard checking."""
    is_safe: bool
    severity: str  # 'clear', 'low', 'medium', 'high', 'critical'
    issues: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SyllableInfo:
    """Syllable analysis result."""
    count: int
    pattern: str  # e.g., "STRONG-weak" (trochaic)
    weight: str   # 'light', 'heavy', 'superheavy'
    rhythm_type: str  # 'trochaic', 'iambic', 'dactylic', etc.


# =============================================================================
# Config Loaders
# =============================================================================

@lru_cache(maxsize=10)
def _load_yaml(filename: str) -> Dict:
    """Load a YAML file from phonemes directory."""
    filepath = PHONEMES_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_phonaesthemes() -> Dict:
    """Load sound symbolism configuration."""
    return _load_yaml('phonaesthemes.yaml')


def load_hazards() -> Dict:
    """Load cross-linguistic hazards database."""
    return _load_yaml('hazards.yaml')


def load_industries() -> Dict:
    """Load industry profiles."""
    return _load_yaml('industries.yaml')


def load_culture_config(culture: str) -> Dict:
    """Load a specific culture's phoneme configuration."""
    return _load_yaml(f'{culture}.yaml')


# =============================================================================
# Hazard Checker
# =============================================================================

class HazardChecker:
    """
    Checks brand names for cross-linguistic hazards.
    """

    def __init__(self):
        self._hazards = load_hazards()
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._compiled_patterns = {}
        patterns = self._hazards.get('phonetic_patterns', {})
        for name, data in patterns.items():
            if isinstance(data, dict) and 'regex' in data:
                try:
                    self._compiled_patterns[name] = {
                        'regex': re.compile(data['regex'], re.IGNORECASE),
                        'sounds_like': data.get('sounds_like', ''),
                        'severity': data.get('severity', 'medium')
                    }
                except re.error:
                    pass

    def check(self, name: str, markets: List[str] = None) -> HazardResult:
        """
        Check a name for hazards.

        Parameters
        ----------
        name : str
            Brand name to check
        markets : list, optional
            Target markets to check (e.g., ['german', 'spanish', 'french'])
            If None, checks all markets.

        Returns
        -------
        HazardResult
            Result with safety assessment and issues found
        """
        issues = []
        name_lower = name.lower()

        # Check exact word hazards
        words = self._hazards.get('words', {})
        for hazard_word, data in words.items():
            if hazard_word in name_lower:
                if markets is None or data.get('language', '') in markets:
                    issues.append({
                        'type': 'exact_word',
                        'word': hazard_word,
                        'meaning': data.get('meaning', ''),
                        'language': data.get('language', ''),
                        'severity': data.get('severity', 'medium'),
                        'note': data.get('note', '')
                    })

        # Check pattern hazards
        patterns = self._hazards.get('patterns', {})
        for pattern, data in patterns.items():
            if pattern in name_lower:
                issues.append({
                    'type': 'pattern',
                    'pattern': pattern,
                    'similar_to': data.get('similar_to', ''),
                    'severity': data.get('severity', 'medium'),
                    'languages': data.get('languages', [])
                })

        # Check sound-alike hazards
        sound_alikes = self._hazards.get('sound_alikes', {})
        for hazard, data in sound_alikes.items():
            if hazard in name_lower or self._sounds_similar(name_lower, hazard):
                issues.append({
                    'type': 'sound_alike',
                    'hazard': hazard,
                    'sounds_like': data.get('sounds_like', ''),
                    'severity': data.get('severity', 'medium'),
                    'note': data.get('note', '')
                })

        # Check phonetic patterns (regex)
        for pattern_name, pattern_data in self._compiled_patterns.items():
            if pattern_data['regex'].search(name_lower):
                issues.append({
                    'type': 'phonetic_pattern',
                    'pattern': pattern_name,
                    'sounds_like': pattern_data['sounds_like'],
                    'severity': pattern_data['severity']
                })

        # Check risky endings
        risky_endings = self._hazards.get('phonetic_patterns', {}).get('risky_endings', [])
        for ending in risky_endings:
            if name_lower.endswith(ending):
                issues.append({
                    'type': 'risky_ending',
                    'ending': ending,
                    'severity': 'medium'
                })

        # Determine overall severity
        if not issues:
            return HazardResult(is_safe=True, severity='clear', issues=[])

        max_severity = max(
            self._severity_rank(i.get('severity', 'low'))
            for i in issues
        )
        severity_map = {0: 'low', 1: 'medium', 2: 'high', 3: 'critical'}

        return HazardResult(
            is_safe=max_severity < 2,  # Safe if below 'high'
            severity=severity_map.get(max_severity, 'medium'),
            issues=issues
        )

    def _severity_rank(self, severity: str) -> int:
        """Convert severity string to numeric rank."""
        return {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}.get(severity, 1)

    def _sounds_similar(self, name: str, hazard: str) -> bool:
        """Check if name sounds similar to hazard (basic phonetic matching)."""
        # Simple Soundex-like comparison
        def simplify(s):
            # Remove vowels except first, collapse doubles
            result = s[0] if s else ''
            for c in s[1:]:
                if c not in 'aeiou' and c != result[-1]:
                    result += c
            return result

        return simplify(name) == simplify(hazard) or hazard in name


# =============================================================================
# Syllable Analyzer
# =============================================================================

class SyllableAnalyzer:
    """
    Analyzes syllable structure, stress patterns, and rhythm.
    """

    VOWELS = set('aeiouy')
    CONSONANTS = set('bcdfghjklmnpqrstvwxz')

    def analyze(self, name: str) -> SyllableInfo:
        """
        Analyze syllable structure of a name.

        Returns syllable count, stress pattern, and rhythm type.
        """
        syllables = self._count_syllables(name)
        pattern = self._detect_stress_pattern(name, syllables)
        weight = self._calculate_weight(name)
        rhythm = self._classify_rhythm(pattern)

        return SyllableInfo(
            count=syllables,
            pattern=pattern,
            weight=weight,
            rhythm_type=rhythm
        )

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower()
        count = 0
        prev_vowel = False

        for char in word:
            is_vowel = char in self.VOWELS
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Handle silent e
        if word.endswith('e') and count > 1:
            count -= 1

        return max(1, count)

    def _detect_stress_pattern(self, word: str, syllable_count: int) -> str:
        """
        Detect likely stress pattern.

        Uses heuristics based on English/German stress rules.
        """
        if syllable_count == 1:
            return "STRONG"

        # Common patterns
        if syllable_count == 2:
            # Most 2-syllable words are trochaic (STRONG-weak)
            # Exceptions: words ending in -tion, -sion, -ic
            word_lower = word.lower()
            if any(word_lower.endswith(s) for s in ['tion', 'sion', 'ic', 'ique']):
                return "weak-STRONG"
            return "STRONG-weak"

        if syllable_count == 3:
            word_lower = word.lower()
            # Latin-style words often have penultimate stress
            if any(word_lower.endswith(s) for s in ['ius', 'ium', 'ial', 'ian']):
                return "weak-STRONG-weak"
            return "STRONG-weak-weak"

        # Default for longer words
        pattern = []
        for i in range(syllable_count):
            if i == 0 or i == syllable_count - 2:
                pattern.append("STRONG")
            else:
                pattern.append("weak")
        return "-".join(pattern)

    def _calculate_weight(self, word: str) -> str:
        """Calculate syllable weight (light, heavy, superheavy)."""
        # Based on coda complexity
        word_lower = word.lower()

        # Count final consonant cluster
        final_consonants = 0
        for char in reversed(word_lower):
            if char in self.CONSONANTS:
                final_consonants += 1
            else:
                break

        if final_consonants >= 2:
            return 'superheavy'
        elif final_consonants == 1:
            return 'heavy'
        else:
            return 'light'

    def _classify_rhythm(self, pattern: str) -> str:
        """Classify the rhythm type based on stress pattern."""
        if pattern == "STRONG-weak":
            return "trochaic"
        elif pattern == "weak-STRONG":
            return "iambic"
        elif pattern == "STRONG-weak-weak":
            return "dactylic"
        elif pattern == "weak-weak-STRONG":
            return "anapestic"
        elif "STRONG-weak" in pattern:
            return "trochaic"
        elif "weak-STRONG" in pattern:
            return "iambic"
        return "mixed"


# =============================================================================
# Phonaesthetics Engine
# =============================================================================

class PhonaestheticsEngine:
    """
    Generates and scores names based on sound symbolism.
    """

    def __init__(self):
        self._config = load_phonaesthemes()

    def get_sounds_for_archetype(self, archetype: str) -> Dict[str, List[str]]:
        """Get preferred sounds for a given brand archetype."""
        archetypes = self._config.get('archetypes', {})
        arch_config = archetypes.get(archetype, {})

        return {
            'preferred_onsets': arch_config.get('preferred_onsets', []),
            'preferred_consonants': arch_config.get('preferred_consonants', []),
            'preferred_vowels': arch_config.get('preferred_vowels', []),
            'preferred_codas': arch_config.get('preferred_codas', []),
            'avoid_onsets': arch_config.get('avoid_onsets', []),
            'avoid_consonants': arch_config.get('avoid_consonants', []),
        }

    def get_sounds_for_meaning(self, meaning: str) -> List[str]:
        """Get sounds associated with a semantic meaning."""
        semantic = self._config.get('semantic_sounds', {})
        meaning_config = semantic.get(meaning, {})
        return meaning_config.get('sounds', [])

    def score_for_archetype(self, name: str, archetype: str) -> float:
        """
        Score how well a name fits an archetype based on phonaesthetics.

        Returns 0.0 to 1.0
        """
        sounds = self.get_sounds_for_archetype(archetype)
        if not sounds['preferred_consonants']:
            return 0.5  # Neutral if archetype not found

        name_lower = name.lower()
        score = 0.5  # Start neutral

        # Check onset (beginning)
        for onset in sounds['preferred_onsets']:
            if name_lower.startswith(onset):
                score += 0.15
                break

        for onset in sounds['avoid_onsets']:
            if name_lower.startswith(onset):
                score -= 0.15
                break

        # Check consonants
        preferred_count = sum(1 for c in name_lower if c in sounds['preferred_consonants'])
        avoid_count = sum(1 for c in name_lower if c in sounds.get('avoid_consonants', []))

        score += min(preferred_count * 0.05, 0.2)
        score -= min(avoid_count * 0.05, 0.2)

        # Check vowels
        preferred_vowels = set(sounds['preferred_vowels'])
        vowel_match = sum(1 for c in name_lower if c in preferred_vowels)
        score += min(vowel_match * 0.03, 0.15)

        return min(max(score, 0.0), 1.0)

    def analyze(self, name: str) -> Dict[str, Any]:
        """
        Analyze phonaesthetic properties of a name.

        Returns dict with onset type, vowel character, consonant character, etc.
        """
        name_lower = name.lower()
        result = {
            'onset': '',
            'onset_type': '',
            'dominant_vowels': [],
            'dominant_consonants': [],
            'feel': []
        }

        # Detect onset
        onsets = self._config.get('onsets', {})
        for onset, data in onsets.items():
            if name_lower.startswith(onset):
                result['onset'] = onset
                result['onset_type'] = data.get('brand_feel', [])
                break

        # Count vowels and consonants
        vowels = self._config.get('vowels', {})
        consonants = self._config.get('consonants', {})

        vowel_counts = {}
        for char in name_lower:
            if char in 'aeiou':
                vowel_counts[char] = vowel_counts.get(char, 0) + 1

        if vowel_counts:
            dominant = max(vowel_counts, key=vowel_counts.get)
            if dominant in vowels:
                result['dominant_vowels'] = [dominant]
                result['feel'].extend(vowels[dominant].get('feel', []))

        return result


# =============================================================================
# Industry Profile Manager
# =============================================================================

class IndustryManager:
    """
    Manages industry-specific naming conventions.
    """

    def __init__(self):
        self._config = load_industries()

    def get_profile(self, industry: str) -> Dict[str, Any]:
        """Get the profile for an industry."""
        industries = self._config.get('industries', {})
        return industries.get(industry, {})

    def get_preferred_suffixes(self, industry: str) -> List[str]:
        """Get preferred suffixes for an industry."""
        profile = self.get_profile(industry)
        phonetics = profile.get('phonetics', {})
        return phonetics.get('preferred_suffixes', [])

    def get_ideal_length(self, industry: str) -> Tuple[int, int]:
        """Get ideal name length range for an industry."""
        profile = self.get_profile(industry)
        length = profile.get('length', {})
        return (length.get('ideal_min', 4), length.get('ideal_max', 8))

    def get_preferred_cultures(self, industry: str) -> List[str]:
        """Get preferred cultural sources for an industry."""
        profile = self.get_profile(industry)
        return profile.get('cultural_sources', [])

    def get_archetypes(self, industry: str) -> List[str]:
        """Get recommended archetypes for an industry."""
        profile = self.get_profile(industry)
        return profile.get('archetypes', [])

    def list_industries(self) -> List[str]:
        """List all available industries."""
        return list(self._config.get('industries', {}).keys())


# =============================================================================
# Memorability Scorer
# =============================================================================

class MemorabilityScorer:
    """
    Enhanced scoring that considers memorability factors beyond pronounceability.
    """

    def __init__(self):
        self._hazard_checker = HazardChecker()
        self._syllable_analyzer = SyllableAnalyzer()
        self._phonaesthetics = PhonaestheticsEngine()

    def score(self, name: str, archetype: str = None, industry: str = None) -> Dict[str, float]:
        """
        Comprehensive memorability score.

        Returns dict with component scores and overall score.
        """
        scores = {}

        # Pronounceability (basic)
        scores['pronounceability'] = self._score_pronounceability(name)

        # Length score
        scores['length'] = self._score_length(name)

        # Distinctiveness
        scores['distinctiveness'] = self._score_distinctiveness(name)

        # Rhythm/Stress
        scores['rhythm'] = self._score_rhythm(name)

        # Visual balance
        scores['visual'] = self._score_visual(name)

        # Hazard penalty
        hazard_result = self._hazard_checker.check(name)
        if hazard_result.severity == 'critical':
            scores['hazard_penalty'] = -0.5
        elif hazard_result.severity == 'high':
            scores['hazard_penalty'] = -0.3
        elif hazard_result.severity == 'medium':
            scores['hazard_penalty'] = -0.1
        else:
            scores['hazard_penalty'] = 0.0

        # Archetype fit
        if archetype:
            scores['archetype_fit'] = self._phonaesthetics.score_for_archetype(name, archetype)
        else:
            scores['archetype_fit'] = 0.5

        # Calculate overall
        weights = {
            'pronounceability': 0.25,
            'length': 0.15,
            'distinctiveness': 0.15,
            'rhythm': 0.10,
            'visual': 0.10,
            'archetype_fit': 0.15,
            'hazard_penalty': 0.10
        }

        overall = sum(scores[k] * weights.get(k, 0.1) for k in scores if k != 'hazard_penalty')
        overall += scores['hazard_penalty']
        scores['overall'] = min(max(overall, 0.0), 1.0)

        return scores

    def _score_pronounceability(self, name: str) -> float:
        """Score basic pronounceability."""
        score = 1.0
        name_lower = name.lower()

        # Penalty for difficult clusters
        difficult = ['tsch', 'dsch', 'szcz', 'cks', 'phr', 'thr']
        for d in difficult:
            if d in name_lower:
                score -= 0.15

        # Penalty for double vowels
        for v in 'aeiou':
            if v + v in name_lower:
                score -= 0.4

        # Penalty for triple consonants
        consonants = 'bcdfghjklmnpqrstvwxyz'
        for i in range(len(name_lower) - 2):
            if all(c in consonants for c in name_lower[i:i+3]):
                score -= 0.2
                break

        return max(score, 0.0)

    def _score_length(self, name: str) -> float:
        """Score name length (ideal is 5-7 characters)."""
        length = len(name)
        if 5 <= length <= 7:
            return 1.0
        elif 4 <= length <= 8:
            return 0.8
        elif 3 <= length <= 9:
            return 0.6
        else:
            return 0.4

    def _score_distinctiveness(self, name: str) -> float:
        """Score uniqueness/distinctiveness of sound patterns."""
        score = 0.5
        name_lower = name.lower()

        # Bonus for unique letter combinations
        unique_chars = len(set(name_lower))
        char_ratio = unique_chars / len(name_lower) if name_lower else 0
        score += char_ratio * 0.3

        # Bonus for interesting onsets
        interesting_onsets = ['kr', 'tr', 'st', 'br', 'pr', 'gl', 'fl', 'sp']
        for onset in interesting_onsets:
            if name_lower.startswith(onset):
                score += 0.1
                break

        # Bonus for memorable endings
        memorable_endings = ['ix', 'ex', 'on', 'ar', 'or', 'a', 'ia']
        for ending in memorable_endings:
            if name_lower.endswith(ending):
                score += 0.1
                break

        return min(score, 1.0)

    def _score_rhythm(self, name: str) -> float:
        """Score rhythmic quality."""
        info = self._syllable_analyzer.analyze(name)

        # 2-3 syllables is ideal
        if info.count in [2, 3]:
            score = 1.0
        elif info.count == 1:
            score = 0.7
        elif info.count == 4:
            score = 0.6
        else:
            score = 0.4

        # Bonus for trochaic rhythm (most natural in English/German)
        if info.rhythm_type == 'trochaic':
            score += 0.1

        return min(score, 1.0)

    def _score_visual(self, name: str) -> float:
        """Score visual balance of the written name."""
        score = 1.0

        # Count ascenders (b, d, f, h, k, l, t) and descenders (g, j, p, q, y)
        ascenders = sum(1 for c in name.lower() if c in 'bdfhklt')
        descenders = sum(1 for c in name.lower() if c in 'gjpqy')

        # Balanced is good
        if ascenders > 0 and descenders > 0:
            score += 0.1

        # Too many of either is visually heavy
        if ascenders > 3 or descenders > 2:
            score -= 0.1

        # X, K, V are visually distinctive
        distinctive = sum(1 for c in name.lower() if c in 'xkvz')
        score += distinctive * 0.05

        return min(max(score, 0.0), 1.0)


# =============================================================================
# Base Cultural Generator
# =============================================================================

class CulturalGenerator(ABC):
    """
    Abstract base class for all cultural name generators.

    Provides common functionality:
    - Config loading
    - Hazard checking
    - Syllable analysis
    - Phonaesthetics
    - Industry support
    - Memorability scoring
    """

    def __init__(self, culture: str, seed: int = None):
        if seed:
            random.seed(seed)

        self.culture = culture
        self._config = load_culture_config(culture)
        self._hazard_checker = HazardChecker()
        self._syllable_analyzer = SyllableAnalyzer()
        self._phonaesthetics = PhonaestheticsEngine()
        self._industry_manager = IndustryManager()
        self._memorability = MemorabilityScorer()

        # Build root pool
        self._all_roots = self._build_root_pool()

    def _build_root_pool(self) -> List[Tuple[str, str, str]]:
        """Build list of all roots as (phoneme, meaning, category) tuples."""
        all_roots = []
        roots = self._config.get('roots', {})
        for category, root_list in roots.items():
            for root_data in root_list:
                if len(root_data) >= 2:
                    phoneme, meaning = root_data[0], root_data[1]
                    all_roots.append((phoneme, meaning, category))
        return all_roots

    def generate(self,
                 count: int = 20,
                 categories: List[str] = None,
                 archetype: str = None,
                 industry: str = None,
                 min_length: int = 4,
                 max_length: int = 9,
                 check_hazards: bool = True,
                 markets: List[str] = None) -> List[GeneratedName]:
        """
        Generate brand names.

        Parameters
        ----------
        count : int
            Number of names to generate
        categories : list, optional
            Filter by root categories
        archetype : str, optional
            Brand archetype (power, elegance, speed, etc.)
        industry : str, optional
            Target industry (tech, pharma, luxury, etc.)
        min_length : int
            Minimum name length
        max_length : int
            Maximum name length
        check_hazards : bool
            Whether to filter out hazardous names
        markets : list, optional
            Target markets for hazard checking

        Returns
        -------
        list[GeneratedName]
            Generated names sorted by score
        """
        # Get industry preferences if specified
        if industry:
            profile = self._industry_manager.get_profile(industry)
            if profile:
                length_prefs = profile.get('length', {})
                min_length = length_prefs.get('ideal_min', min_length)
                max_length = length_prefs.get('ideal_max', max_length)
                if not archetype and profile.get('archetypes'):
                    archetype = random.choice(profile['archetypes'])

        names = []
        attempts = 0
        max_attempts = count * 20
        seen = set()

        while len(names) < count and attempts < max_attempts:
            attempts += 1

            # Generate a name
            result = self._generate_one(categories, archetype, industry)
            if result is None:
                continue

            name_str, roots, meanings, method = result

            # Length check
            if len(name_str) < min_length or len(name_str) > max_length:
                continue

            # Uniqueness check
            if name_str.lower() in seen:
                continue
            seen.add(name_str.lower())

            # Hazard check
            hazards = []
            if check_hazards:
                hazard_result = self._hazard_checker.check(name_str, markets)
                if hazard_result.severity in ['high', 'critical']:
                    continue
                hazards = hazard_result.issues

            # Score
            scores = self._memorability.score(name_str, archetype, industry)
            if scores['overall'] < 0.5:
                continue

            # Syllable analysis
            syllable_info = self._syllable_analyzer.analyze(name_str)

            # Phonaesthetics analysis
            phon_analysis = self._phonaesthetics.analyze(name_str)

            names.append(GeneratedName(
                name=name_str.capitalize(),
                roots_used=roots,
                meaning_hints=meanings,
                score=scores['overall'],
                method=method,
                culture=self.culture,
                archetype=archetype or '',
                syllables=syllable_info.count,
                stress_pattern=syllable_info.pattern,
                hazards=hazards,
                phonaesthetics=phon_analysis
            ))

        names.sort(key=lambda n: n.score, reverse=True)
        return names[:count]

    @abstractmethod
    def _generate_one(self,
                      categories: List[str],
                      archetype: str,
                      industry: str) -> Optional[Tuple[str, List[str], List[str], str]]:
        """
        Generate a single name. Must be implemented by subclasses.

        Returns (name_string, roots_used, meanings, method_name) or None.
        """
        pass

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_pool(self, categories: List[str] = None) -> List[Tuple[str, str, str]]:
        """Get root pool, optionally filtered by categories."""
        if categories:
            filtered = [r for r in self._all_roots if r[2] in categories]
            if filtered:
                return filtered
        return self._all_roots

    def _get_suffixes(self, *suffix_types: str) -> List[str]:
        """Get suffix pool from specified types."""
        suffixes = self._config.get('suffixes', {})
        pool = []
        for stype in suffix_types:
            pool.extend(suffixes.get(stype, []))
        return pool if pool else ['a', 'o', 'er', 'en']

    def _get_prefixes(self) -> List[str]:
        """Get prefix pool."""
        prefixes = self._config.get('prefixes', {})
        pool = []
        for ptype, plist in prefixes.items():
            pool.extend(plist)
        return pool if pool else ['', '', '']

    def _get_connector(self, root_end: str, suffix_start: str) -> str:
        """Get appropriate connector between root and suffix."""
        vowels = 'aeiou'
        is_root_vowel = root_end in vowels
        is_suffix_vowel = suffix_start in vowels

        if is_root_vowel and is_suffix_vowel:
            return random.choice(['n', 'r', 'l', 's', 'x'])
        elif not is_root_vowel and not is_suffix_vowel:
            return random.choice(['a', 'i', 'o', 'e'])
        return ''


# =============================================================================
# Cross-Culture Blender
# =============================================================================

class CultureBlender:
    """
    Blends roots and suffixes from multiple cultures.
    """

    def __init__(self, cultures: List[str]):
        self.cultures = cultures
        self._configs = {c: load_culture_config(c) for c in cultures}
        self._hazard_checker = HazardChecker()
        self._memorability = MemorabilityScorer()

    def blend(self,
              count: int = 20,
              archetype: str = None,
              check_hazards: bool = True) -> List[GeneratedName]:
        """
        Generate blended names from multiple cultures.
        """
        names = []
        attempts = 0
        max_attempts = count * 20
        seen = set()

        while len(names) < count and attempts < max_attempts:
            attempts += 1

            # Pick random cultures for root and suffix
            root_culture = random.choice(self.cultures)
            suffix_culture = random.choice(self.cultures)

            root_config = self._configs[root_culture]
            suffix_config = self._configs[suffix_culture]

            # Get random root
            all_roots = []
            for cat, roots in root_config.get('roots', {}).items():
                for r in roots:
                    if len(r) >= 2:
                        all_roots.append((r[0], r[1], cat))

            if not all_roots:
                continue

            root, meaning, cat = random.choice(all_roots)

            # Get random suffix
            all_suffixes = []
            for stype, suffixes in suffix_config.get('suffixes', {}).items():
                all_suffixes.extend(suffixes)

            if not all_suffixes:
                continue

            suffix = random.choice(all_suffixes)

            # Truncate root if needed
            if len(root) > 5:
                root = root[:5]

            # Build name
            vowels = 'aeiou'
            if root[-1] in vowels and suffix[0] in vowels:
                connector = random.choice(['n', 'r', 'l'])
            elif root[-1] not in vowels and suffix[0] not in vowels:
                connector = random.choice(['a', 'i', 'o'])
            else:
                connector = ''

            name_str = root + connector + suffix

            # Checks
            if len(name_str) < 4 or len(name_str) > 9:
                continue

            if name_str.lower() in seen:
                continue
            seen.add(name_str.lower())

            if check_hazards:
                hazard_result = self._hazard_checker.check(name_str)
                if hazard_result.severity in ['high', 'critical']:
                    continue

            scores = self._memorability.score(name_str, archetype)
            if scores['overall'] < 0.5:
                continue

            names.append(GeneratedName(
                name=name_str.capitalize(),
                roots_used=[root],
                meaning_hints=[meaning],
                score=scores['overall'],
                method='blend',
                culture=f"{root_culture}+{suffix_culture}",
                archetype=archetype or ''
            ))

        names.sort(key=lambda n: n.score, reverse=True)
        return names[:count]


# =============================================================================
# Competitive Differentiator
# =============================================================================

class CompetitiveDifferentiator:
    """
    Ensures generated names are distinct from competitors.
    """

    def __init__(self, competitors: List[str] = None):
        self.competitors = [c.lower() for c in (competitors or [])]
        self._build_competitor_patterns()

    def _build_competitor_patterns(self):
        """Build patterns from competitor names."""
        self.competitor_onsets = set()
        self.competitor_endings = set()
        self.competitor_sounds = set()

        for comp in self.competitors:
            # Capture first 2-3 chars
            if len(comp) >= 2:
                self.competitor_onsets.add(comp[:2])
            if len(comp) >= 3:
                self.competitor_onsets.add(comp[:3])

            # Capture last 2-3 chars
            if len(comp) >= 2:
                self.competitor_endings.add(comp[-2:])
            if len(comp) >= 3:
                self.competitor_endings.add(comp[-3:])

            # Add all characters
            self.competitor_sounds.update(comp)

    def is_distinct(self, name: str, threshold: float = 0.5) -> bool:
        """
        Check if name is sufficiently distinct from competitors.

        Returns True if distinct enough, False if too similar.
        """
        if not self.competitors:
            return True

        name_lower = name.lower()
        similarity_score = 0.0

        # Check onset similarity
        for onset in self.competitor_onsets:
            if name_lower.startswith(onset):
                similarity_score += 0.3
                break

        # Check ending similarity
        for ending in self.competitor_endings:
            if name_lower.endswith(ending):
                similarity_score += 0.2
                break

        # Check overall character overlap
        name_chars = set(name_lower)
        overlap = len(name_chars & self.competitor_sounds) / len(name_chars) if name_chars else 0
        similarity_score += overlap * 0.3

        # Check direct substring matches
        for comp in self.competitors:
            if comp in name_lower or name_lower in comp:
                similarity_score += 0.4
                break

        return similarity_score < threshold

    def filter_names(self, names: List[GeneratedName]) -> List[GeneratedName]:
        """Filter out names too similar to competitors."""
        return [n for n in names if self.is_distinct(n.name)]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    'GeneratedName',
    'HazardResult',
    'SyllableInfo',
    # Components
    'HazardChecker',
    'SyllableAnalyzer',
    'PhonaestheticsEngine',
    'IndustryManager',
    'MemorabilityScorer',
    # Base class
    'CulturalGenerator',
    # Utilities
    'CultureBlender',
    'CompetitiveDifferentiator',
    # Loaders
    'load_phonaesthemes',
    'load_hazards',
    'load_industries',
    'load_culture_config',
]
