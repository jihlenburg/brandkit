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
from abc import ABC, abstractmethod

# Use true random from entropy module instead of Python's random
from .entropy import get_rng, EntropyEngine
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


def load_base_generator_config() -> Dict:
    """Load base generator settings (non-phoneme defaults)."""
    return _load_yaml('base_generator.yaml')


def _require_cfg(cfg: Dict[str, Any], key: str, context: str):
    value = cfg.get(key)
    if value is None:
        raise ValueError(f"{context}.{key} must be set in base_generator.yaml")
    return value


# =============================================================================
# Hazard Checker
# =============================================================================

class HazardChecker:
    """
    Checks brand names for cross-linguistic hazards.

    Enhanced with:
    - Syllable-aware matching (hazards at syllable boundaries are more severe)
    - G2P-based phonetic matching for EN/DE markets
    - EN/DE homophone and minimal pair detection
    """

    def __init__(self):
        self._hazards = load_hazards()
        self._cfg = load_base_generator_config().get('hazard', {}) or {}
        rank_map = self._cfg.get('severity_rank')
        if not rank_map:
            raise ValueError("base_generator.hazard.severity_rank must be set in base_generator.yaml")
        self._severity_rank_map = {k: int(v) for k, v in rank_map.items()}
        self._severity_default = _require_cfg(self._cfg, 'severity_default', 'base_generator.hazard')
        self._safe_max_rank = _require_cfg(self._cfg, 'safe_max_rank', 'base_generator.hazard')
        self._severity_map = {v: k for k, v in self._severity_rank_map.items()}
        self._compile_patterns()
        self._compile_en_de_patterns()

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

    def _compile_en_de_patterns(self):
        """Pre-compile EN/DE-specific patterns."""
        self._en_de_patterns = {}
        en_de = self._hazards.get('en_de_patterns', {})
        universal = en_de.get('universal_avoid', [])
        for pattern_data in universal:
            if isinstance(pattern_data, dict) and 'regex' in pattern_data:
                try:
                    self._en_de_patterns[pattern_data['regex']] = {
                        'regex': re.compile(pattern_data['regex'], re.IGNORECASE),
                        'reason': pattern_data.get('reason', ''),
                        'severity': pattern_data.get('severity', 'medium')
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

        # Check EN/DE-specific patterns (universal_avoid)
        for pattern_key, pattern_data in self._en_de_patterns.items():
            if pattern_data['regex'].search(name_lower):
                issues.append({
                    'type': 'en_de_pattern',
                    'pattern': pattern_key,
                    'reason': pattern_data['reason'],
                    'severity': pattern_data['severity']
                })

        # Check EN/DE homophones
        en_de_homophones = self._hazards.get('en_de_homophones', {})
        for word, data in en_de_homophones.items():
            if word in name_lower:
                severity = data.get('severity', 'medium')
                issues.append({
                    'type': 'en_de_homophone',
                    'word': word,
                    'german_meaning': data.get('german_meaning', ''),
                    'english_sounds_like': data.get('english_sounds_like', ''),
                    'severity': severity,
                    'note': data.get('note', '')
                })

        # Syllable-aware hazard detection
        # Check if hazards appear at syllable boundaries (more severe)
        syllable_issues = self._check_syllable_hazards(name_lower)
        issues.extend(syllable_issues)

        # Determine overall severity
        if not issues:
            return HazardResult(is_safe=True, severity='clear', issues=[])

        max_severity = max(
            self._severity_rank(i.get('severity', 'low'))
            for i in issues
        )
        severity_map = self._severity_map

        return HazardResult(
            is_safe=max_severity <= int(self._safe_max_rank),
            severity=severity_map.get(max_severity, self._severity_default),
            issues=issues
        )

    def _severity_rank(self, severity: str) -> int:
        """Convert severity string to numeric rank."""
        return self._severity_rank_map.get(severity, self._severity_rank_map.get(self._severity_default, 1))

    def _sounds_similar(self, name: str, hazard: str) -> bool:
        """Check if name sounds similar to hazard (basic phonetic matching)."""
        # Simple Soundex-like comparison
        def simplify(s):
            # Remove vowels except first, collapse doubles
            sound_cfg = self._cfg.get('sound_similarity', {}) or {}
            vowel_chars = _require_cfg(sound_cfg, 'vowel_chars', 'base_generator.hazard.sound_similarity')
            result = s[0] if s else ''
            for c in s[1:]:
                if c not in vowel_chars and c != result[-1]:
                    result += c
            return result

        return simplify(name) == simplify(hazard) or hazard in name

    def _check_syllable_hazards(self, name: str) -> List[Dict[str, Any]]:
        """
        Check for hazards that appear at syllable boundaries.

        Hazards at syllable boundaries are more severe because they:
        1. Are more likely to be perceived as separate words
        2. May be emphasized during natural speech stress patterns
        3. Stand out more in segmented pronunciation

        Returns list of hazard issues found.
        """
        issues = []
        syllable_cfg = self._cfg.get('syllable', {}) or {}
        vowels = _require_cfg(syllable_cfg, 'vowels', 'base_generator.hazard.syllable')
        boundary_lookahead = int(_require_cfg(syllable_cfg, 'boundary_lookahead', 'base_generator.hazard.syllable'))
        severity_upgrade = _require_cfg(syllable_cfg, 'severity_upgrade', 'base_generator.hazard.syllable')

        # Find syllable boundaries (approximate)
        # Syllable boundaries typically occur before consonant clusters or after vowels
        syllable_starts = [0]
        prev_vowel = False
        for i, char in enumerate(name):
            is_vowel = char in vowels
            # New syllable starts after a vowel followed by a consonant
            if prev_vowel and not is_vowel and i > 0:
                # Check if there's a vowel after this consonant (onset of new syllable)
                for j in range(i + 1, min(i + 1 + boundary_lookahead, len(name))):
                    if name[j] in vowels:
                        syllable_starts.append(i)
                        break
            prev_vowel = is_vowel

        # Add end of word as a boundary
        syllable_starts.append(len(name))

        # Extract syllables
        syllables = []
        for i in range(len(syllable_starts) - 1):
            syl = name[syllable_starts[i]:syllable_starts[i + 1]]
            if syl:
                syllables.append(syl)

        # Check if any syllable matches a known hazard word exactly
        words = self._hazards.get('words', {})
        for syl in syllables:
            if syl in words:
                data = words[syl]
                # Syllable-aligned hazard is more severe
                base_severity = data.get('severity', 'medium')
                # Upgrade severity for syllable alignment
                issues.append({
                    'type': 'syllable_hazard',
                    'syllable': syl,
                    'meaning': data.get('meaning', ''),
                    'language': data.get('language', ''),
                    'severity': severity_upgrade.get(base_severity, 'high'),
                    'note': f"Hazard appears as complete syllable (more severe): {data.get('note', '')}"
                })

        # Check for hazard patterns at syllable boundaries
        patterns = self._hazards.get('patterns', {})
        for pattern, data in patterns.items():
            # Check if pattern matches start or end of a syllable
            for syl in syllables:
                if syl.startswith(pattern) or syl.endswith(pattern):
                    issues.append({
                        'type': 'syllable_pattern',
                        'pattern': pattern,
                        'syllable': syl,
                        'similar_to': data.get('similar_to', ''),
                        'severity': data.get('severity', 'medium'),
                        'note': 'Pattern at syllable boundary'
                    })
                    break  # Only report once per pattern

        # Check word-initial and word-final positions (most prominent)
        # These positions receive natural stress and attention
        prominent_hazards = _require_cfg(syllable_cfg, 'prominent_hazards', 'base_generator.hazard.syllable')
        prominent_severity = _require_cfg(syllable_cfg, 'prominent_severity', 'base_generator.hazard.syllable')
        prominent_note = _require_cfg(syllable_cfg, 'prominent_note', 'base_generator.hazard.syllable')
        for hazard in prominent_hazards:
            if name.startswith(hazard) or name.endswith(hazard):
                issues.append({
                    'type': 'prominent_position',
                    'hazard': hazard,
                    'position': 'start' if name.startswith(hazard) else 'end',
                    'severity': prominent_severity,
                    'note': prominent_note
                })

        return issues


# =============================================================================
# Syllable Analyzer
# =============================================================================

class SyllableAnalyzer:
    """
    Analyzes syllable structure, stress patterns, and rhythm.
    """

    def __init__(self):
        cfg = load_base_generator_config().get('syllable_analyzer', {}) or {}
        vowels = cfg.get('vowels')
        consonants = cfg.get('consonants')
        if not vowels or not consonants:
            raise ValueError("base_generator.syllable_analyzer.vowels/consonants must be set in base_generator.yaml")
        self._vowels = set(vowels)
        self._consonants = set(consonants)
        self._silent_e_min = cfg.get('silent_e_min_count')
        if self._silent_e_min is None:
            raise ValueError("base_generator.syllable_analyzer.silent_e_min_count must be set in base_generator.yaml")
        self._stress_cfg = cfg.get('stress', {}) or {}
        self._weight_cfg = cfg.get('weight', {}) or {}

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
            is_vowel = char in self._vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Handle silent e
        if word.endswith('e') and count >= int(self._silent_e_min):
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
            two_syllable = _require_cfg(self._stress_cfg, 'two_syllable_weak_suffixes', 'base_generator.syllable_analyzer.stress')
            if any(word_lower.endswith(s) for s in two_syllable):
                return "weak-STRONG"
            return "STRONG-weak"

        if syllable_count == 3:
            word_lower = word.lower()
            # Latin-style words often have penultimate stress
            three_syllable = _require_cfg(self._stress_cfg, 'three_syllable_penultimate_suffixes', 'base_generator.syllable_analyzer.stress')
            if any(word_lower.endswith(s) for s in three_syllable):
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
            if char in self._consonants:
                final_consonants += 1
            else:
                break

        superheavy_min = int(_require_cfg(self._weight_cfg, 'superheavy_min', 'base_generator.syllable_analyzer.weight'))
        heavy_min = int(_require_cfg(self._weight_cfg, 'heavy_min', 'base_generator.syllable_analyzer.weight'))
        if final_consonants >= superheavy_min:
            return 'superheavy'
        elif final_consonants >= heavy_min:
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
        base_cfg = load_base_generator_config().get('phonaesthetics', {}) or {}
        self._score_cfg = base_cfg.get('archetype_scoring', {}) or {}

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
            return float(_require_cfg(self._score_cfg, 'neutral_score', 'base_generator.phonaesthetics.archetype_scoring'))

        name_lower = name.lower()
        score = float(_require_cfg(self._score_cfg, 'neutral_score', 'base_generator.phonaesthetics.archetype_scoring'))

        # Check onset (beginning)
        for onset in sounds['preferred_onsets']:
            if name_lower.startswith(onset):
                score += float(_require_cfg(self._score_cfg, 'onset_bonus', 'base_generator.phonaesthetics.archetype_scoring'))
                break

        for onset in sounds['avoid_onsets']:
            if name_lower.startswith(onset):
                score += float(_require_cfg(self._score_cfg, 'onset_penalty', 'base_generator.phonaesthetics.archetype_scoring'))
                break

        # Check consonants
        preferred_count = sum(1 for c in name_lower if c in sounds['preferred_consonants'])
        avoid_count = sum(1 for c in name_lower if c in sounds.get('avoid_consonants', []))

        consonant_bonus_per = float(_require_cfg(self._score_cfg, 'consonant_bonus_per', 'base_generator.phonaesthetics.archetype_scoring'))
        consonant_bonus_cap = float(_require_cfg(self._score_cfg, 'consonant_bonus_cap', 'base_generator.phonaesthetics.archetype_scoring'))
        consonant_penalty_per = float(_require_cfg(self._score_cfg, 'consonant_penalty_per', 'base_generator.phonaesthetics.archetype_scoring'))
        consonant_penalty_cap = float(_require_cfg(self._score_cfg, 'consonant_penalty_cap', 'base_generator.phonaesthetics.archetype_scoring'))
        score += min(preferred_count * consonant_bonus_per, consonant_bonus_cap)
        score -= min(avoid_count * consonant_penalty_per, consonant_penalty_cap)

        # Check vowels
        preferred_vowels = set(sounds['preferred_vowels'])
        vowel_match = sum(1 for c in name_lower if c in preferred_vowels)
        vowel_bonus_per = float(_require_cfg(self._score_cfg, 'vowel_bonus_per', 'base_generator.phonaesthetics.archetype_scoring'))
        vowel_bonus_cap = float(_require_cfg(self._score_cfg, 'vowel_bonus_cap', 'base_generator.phonaesthetics.archetype_scoring'))
        score += min(vowel_match * vowel_bonus_per, vowel_bonus_cap)

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
        self._cfg = load_base_generator_config().get('memorability', {}) or {}
        if not self._cfg:
            raise ValueError("base_generator.memorability must be set in base_generator.yaml")

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
        hazard_penalties = _require_cfg(self._cfg, 'hazard_penalties', 'base_generator.memorability')
        if hazard_result.severity not in hazard_penalties:
            raise ValueError("base_generator.memorability.hazard_penalties must include all severities")
        scores['hazard_penalty'] = float(hazard_penalties[hazard_result.severity])

        # Archetype fit
        if archetype:
            scores['archetype_fit'] = self._phonaesthetics.score_for_archetype(name, archetype)
        else:
            scores['archetype_fit'] = float(_require_cfg(self._cfg, 'archetype_default', 'base_generator.memorability'))

        # Calculate overall
        weights = _require_cfg(self._cfg, 'weights', 'base_generator.memorability')

        overall = sum(scores[k] * weights.get(k, 0.0) for k in scores if k != 'hazard_penalty')
        overall += scores['hazard_penalty']
        scores['overall'] = min(max(overall, 0.0), 1.0)

        return scores

    def _score_pronounceability(self, name: str) -> float:
        """Score basic pronounceability."""
        score = 1.0
        name_lower = name.lower()

        # Penalty for difficult clusters
        pronounce_cfg = _require_cfg(self._cfg, 'pronounceability', 'base_generator.memorability')
        difficult = _require_cfg(pronounce_cfg, 'difficult_clusters', 'base_generator.memorability.pronounceability')
        cluster_penalty = float(_require_cfg(pronounce_cfg, 'cluster_penalty', 'base_generator.memorability.pronounceability'))
        for d in difficult:
            if d in name_lower:
                score += cluster_penalty

        # Penalty for double vowels
        vowel_chars = _require_cfg(pronounce_cfg, 'vowel_chars', 'base_generator.memorability.pronounceability')
        double_vowel_penalty = float(_require_cfg(pronounce_cfg, 'double_vowel_penalty', 'base_generator.memorability.pronounceability'))
        for v in vowel_chars:
            if v + v in name_lower:
                score += double_vowel_penalty

        # Penalty for triple consonants
        consonants = _require_cfg(pronounce_cfg, 'consonant_chars', 'base_generator.memorability.pronounceability')
        triple_penalty = float(_require_cfg(pronounce_cfg, 'triple_consonant_penalty', 'base_generator.memorability.pronounceability'))
        for i in range(len(name_lower) - 2):
            if all(c in consonants for c in name_lower[i:i+3]):
                score += triple_penalty
                break

        return max(score, 0.0)

    def _score_length(self, name: str) -> float:
        """Score name length (ideal is 5-7 characters)."""
        length_cfg = _require_cfg(self._cfg, 'length', 'base_generator.memorability')
        scores_cfg = _require_cfg(length_cfg, 'scores', 'base_generator.memorability.length')
        length = len(name)
        ideal_min = int(_require_cfg(length_cfg, 'ideal_min', 'base_generator.memorability.length'))
        ideal_max = int(_require_cfg(length_cfg, 'ideal_max', 'base_generator.memorability.length'))
        good_min = int(_require_cfg(length_cfg, 'good_min', 'base_generator.memorability.length'))
        good_max = int(_require_cfg(length_cfg, 'good_max', 'base_generator.memorability.length'))
        ok_min = int(_require_cfg(length_cfg, 'ok_min', 'base_generator.memorability.length'))
        ok_max = int(_require_cfg(length_cfg, 'ok_max', 'base_generator.memorability.length'))

        if ideal_min <= length <= ideal_max:
            return float(_require_cfg(scores_cfg, 'ideal', 'base_generator.memorability.length.scores'))
        if good_min <= length <= good_max:
            return float(_require_cfg(scores_cfg, 'good', 'base_generator.memorability.length.scores'))
        if ok_min <= length <= ok_max:
            return float(_require_cfg(scores_cfg, 'ok', 'base_generator.memorability.length.scores'))
        return float(_require_cfg(scores_cfg, 'too_long', 'base_generator.memorability.length.scores'))

    def _score_distinctiveness(self, name: str) -> float:
        """Score uniqueness/distinctiveness of sound patterns."""
        distinct_cfg = _require_cfg(self._cfg, 'distinctiveness', 'base_generator.memorability')
        score = float(_require_cfg(distinct_cfg, 'base_score', 'base_generator.memorability.distinctiveness'))
        name_lower = name.lower()

        # Bonus for unique letter combinations
        unique_chars = len(set(name_lower))
        char_ratio = unique_chars / len(name_lower) if name_lower else 0
        score += char_ratio * float(_require_cfg(distinct_cfg, 'unique_char_weight', 'base_generator.memorability.distinctiveness'))

        # Bonus for interesting onsets
        interesting_onsets = _require_cfg(distinct_cfg, 'interesting_onsets', 'base_generator.memorability.distinctiveness')
        onset_bonus = float(_require_cfg(distinct_cfg, 'interesting_onset_bonus', 'base_generator.memorability.distinctiveness'))
        for onset in interesting_onsets:
            if name_lower.startswith(onset):
                score += onset_bonus
                break

        # Bonus for memorable endings
        memorable_endings = _require_cfg(distinct_cfg, 'memorable_endings', 'base_generator.memorability.distinctiveness')
        ending_bonus = float(_require_cfg(distinct_cfg, 'memorable_ending_bonus', 'base_generator.memorability.distinctiveness'))
        for ending in memorable_endings:
            if name_lower.endswith(ending):
                score += ending_bonus
                break

        return min(score, float(_require_cfg(distinct_cfg, 'max_score', 'base_generator.memorability.distinctiveness')))

    def _score_rhythm(self, name: str) -> float:
        """Score rhythmic quality."""
        info = self._syllable_analyzer.analyze(name)

        # 2-3 syllables is ideal
        rhythm_cfg = _require_cfg(self._cfg, 'rhythm', 'base_generator.memorability')
        syllable_scores = _require_cfg(rhythm_cfg, 'syllable_scores', 'base_generator.memorability.rhythm')
        if info.count in (2, 3):
            score = float(_require_cfg(syllable_scores, str(info.count), 'base_generator.memorability.rhythm.syllable_scores'))
        elif info.count == 1:
            score = float(_require_cfg(syllable_scores, '1', 'base_generator.memorability.rhythm.syllable_scores'))
        elif info.count == 4:
            score = float(_require_cfg(syllable_scores, '4', 'base_generator.memorability.rhythm.syllable_scores'))
        else:
            score = float(_require_cfg(syllable_scores, 'default', 'base_generator.memorability.rhythm.syllable_scores'))

        # Bonus for trochaic rhythm (most natural in English/German)
        if info.rhythm_type == 'trochaic':
            score += float(_require_cfg(rhythm_cfg, 'trochaic_bonus', 'base_generator.memorability.rhythm'))

        return min(score, float(_require_cfg(rhythm_cfg, 'max_score', 'base_generator.memorability.rhythm')))

    def _score_visual(self, name: str) -> float:
        """Score visual balance of the written name."""
        visual_cfg = _require_cfg(self._cfg, 'visual', 'base_generator.memorability')
        score = float(_require_cfg(visual_cfg, 'base_score', 'base_generator.memorability.visual'))

        # Count ascenders (b, d, f, h, k, l, t) and descenders (g, j, p, q, y)
        ascenders_chars = _require_cfg(visual_cfg, 'ascenders', 'base_generator.memorability.visual')
        descenders_chars = _require_cfg(visual_cfg, 'descenders', 'base_generator.memorability.visual')
        ascenders = sum(1 for c in name.lower() if c in ascenders_chars)
        descenders = sum(1 for c in name.lower() if c in descenders_chars)

        # Balanced is good
        if ascenders > 0 and descenders > 0:
            score += float(_require_cfg(visual_cfg, 'balanced_bonus', 'base_generator.memorability.visual'))

        # Too many of either is visually heavy
        ascender_heavy = int(_require_cfg(visual_cfg, 'ascender_heavy_threshold', 'base_generator.memorability.visual'))
        descender_heavy = int(_require_cfg(visual_cfg, 'descender_heavy_threshold', 'base_generator.memorability.visual'))
        if ascenders > ascender_heavy or descenders > descender_heavy:
            score += float(_require_cfg(visual_cfg, 'heavy_penalty', 'base_generator.memorability.visual'))

        # X, K, V are visually distinctive
        distinctive_letters = _require_cfg(visual_cfg, 'distinctive_letters', 'base_generator.memorability.visual')
        distinctive_bonus = float(_require_cfg(visual_cfg, 'distinctive_bonus_per', 'base_generator.memorability.visual'))
        distinctive = sum(1 for c in name.lower() if c in distinctive_letters)
        score += distinctive * distinctive_bonus

        return min(max(score, 0.0), float(_require_cfg(visual_cfg, 'max_score', 'base_generator.memorability.visual')))


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
        # Use true random from entropy module (seed parameter deprecated)
        self._rng = get_rng()
        self._entropy = EntropyEngine()

        self.culture = culture
        self._base_cfg = load_base_generator_config()
        self._gen_cfg = self._base_cfg.get('cultural_generator', {}) or {}
        if not self._gen_cfg:
            raise ValueError("base_generator.cultural_generator must be set in base_generator.yaml")
        self._defaults_cfg = self._gen_cfg.get('defaults', {}) or {}
        self._connector_cfg = self._gen_cfg.get('connector', {}) or {}
        if not self._connector_cfg.get('vowel_vowel') or not self._connector_cfg.get('consonant_consonant'):
            raise ValueError("base_generator.cultural_generator.connector lists must be set in base_generator.yaml")
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
                 count: int = None,
                 categories: List[str] = None,
                 archetype: str = None,
                 industry: str = None,
                 min_length: int = None,
                 max_length: int = None,
                 check_hazards: bool = True,
                 markets: List[str] = None,
                 phonetic_markets: str = "en_de") -> List[GeneratedName]:
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
        if count is None:
            count = _require_cfg(self._gen_cfg, 'default_count', 'base_generator.cultural_generator')
        if min_length is None:
            min_length = _require_cfg(self._gen_cfg, 'default_min_length', 'base_generator.cultural_generator')
        if max_length is None:
            max_length = _require_cfg(self._gen_cfg, 'default_max_length', 'base_generator.cultural_generator')
        if count is None or min_length is None or max_length is None:
            raise ValueError("base_generator.cultural_generator defaults must be set in base_generator.yaml")

        # Get industry preferences if specified
        if industry:
            profile = self._industry_manager.get_profile(industry)
            if profile:
                length_prefs = profile.get('length', {})
                min_length = length_prefs.get('ideal_min', min_length)
                max_length = length_prefs.get('ideal_max', max_length)
                if not archetype and profile.get('archetypes'):
                    archetype = self._rng.choice(profile['archetypes'])

        names = []
        attempts = 0
        max_attempts_multiplier = _require_cfg(self._gen_cfg, 'max_attempts_multiplier', 'base_generator.cultural_generator')
        max_attempts = count * int(max_attempts_multiplier)
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

            # Pronounceability (EN/DE hard gate)
            try:
                from .phonemes import is_pronounceable
                ok, _ = is_pronounceable(name_str, markets=phonetic_markets)
                if not ok:
                    continue
            except Exception:
                pass

            # Hazard check
            hazards = []
            if check_hazards:
                hazard_result = self._hazard_checker.check(name_str, markets)
                hazard_block_levels = set(_require_cfg(self._gen_cfg, 'hazard_block_levels', 'base_generator.cultural_generator'))
                if hazard_result.severity in hazard_block_levels:
                    continue
                hazards = hazard_result.issues

            # Score
            scores = self._memorability.score(name_str, archetype, industry)
            min_mem = _require_cfg(self._gen_cfg, 'min_memorability_score', 'base_generator.cultural_generator')
            if scores['overall'] < float(min_mem):
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
            items = suffixes.get(stype, [])
            # Filter out non-string values (YAML may parse 'on' as boolean True)
            pool.extend(s for s in items if isinstance(s, str))
        default_suffixes = _require_cfg(self._defaults_cfg, 'suffixes', 'base_generator.cultural_generator.defaults')
        return pool if pool else list(default_suffixes)

    def _get_prefixes(self) -> List[str]:
        """Get prefix pool."""
        prefixes = self._config.get('prefixes', {})
        pool = []
        for ptype, plist in prefixes.items():
            pool.extend(plist)
        default_prefixes = _require_cfg(self._defaults_cfg, 'prefixes', 'base_generator.cultural_generator.defaults')
        return pool if pool else list(default_prefixes)

    def _get_consonants(self, ctype: str = 'all') -> List[str]:
        """Get consonants from phonetics config."""
        phonetics = self._config.get('phonetics', {})
        consonants = phonetics.get('consonants', {})
        default_consonants = _require_cfg(self._defaults_cfg, 'consonants', 'base_generator.cultural_generator.defaults')
        return list(consonants.get(ctype, consonants.get('all', list(default_consonants))))

    def _get_vowels(self, vtype: str = 'all') -> List[str]:
        """Get vowels from phonetics config."""
        phonetics = self._config.get('phonetics', {})
        vowels = phonetics.get('vowels', {})
        default_vowels = _require_cfg(self._defaults_cfg, 'vowels', 'base_generator.cultural_generator.defaults')
        return list(vowels.get(vtype, vowels.get('all', list(default_vowels))))

    def _get_connector(self, root_end: str, suffix_start: str) -> str:
        """Get appropriate connector between root and suffix."""
        vowels = _require_cfg(self._connector_cfg, 'vowels', 'base_generator.cultural_generator.connector')
        is_root_vowel = root_end in vowels
        is_suffix_vowel = suffix_start in vowels

        if is_root_vowel and is_suffix_vowel:
            return self._rng.choice(_require_cfg(self._connector_cfg, 'vowel_vowel', 'base_generator.cultural_generator.connector'))
        elif not is_root_vowel and not is_suffix_vowel:
            return self._rng.choice(_require_cfg(self._connector_cfg, 'consonant_consonant', 'base_generator.cultural_generator.connector'))
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
        self._rng = get_rng()
        self._entropy = EntropyEngine()
        self._cfg = load_base_generator_config().get('culture_blender', {}) or {}
        if not self._cfg:
            raise ValueError("base_generator.culture_blender must be set in base_generator.yaml")
        connector_cfg = self._cfg.get('connector', {}) or {}
        if not connector_cfg.get('vowel_vowel') or not connector_cfg.get('consonant_consonant'):
            raise ValueError("base_generator.culture_blender.connector lists must be set in base_generator.yaml")

    def blend(self,
              count: int = None,
              archetype: str = None,
              check_hazards: bool = True) -> List[GeneratedName]:
        """
        Generate blended names from multiple cultures.
        """
        if count is None:
            count = _require_cfg(self._cfg, 'default_count', 'base_generator.culture_blender')
        if count is None:
            raise ValueError("base_generator.culture_blender.default_count must be set in base_generator.yaml")
        names = []
        attempts = 0
        max_attempts_multiplier = _require_cfg(self._cfg, 'max_attempts_multiplier', 'base_generator.culture_blender')
        max_attempts = count * int(max_attempts_multiplier)
        seen = set()

        while len(names) < count and attempts < max_attempts:
            attempts += 1

            # Pick random cultures for root and suffix
            root_culture = self._rng.choice(self.cultures)
            suffix_culture = self._rng.choice(self.cultures)

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

            root, meaning, cat = self._rng.choice(all_roots)

            # Get random suffix
            all_suffixes = []
            for stype, suffixes in suffix_config.get('suffixes', {}).items():
                # Filter out non-string values (YAML may parse 'on' as boolean True)
                all_suffixes.extend(s for s in suffixes if isinstance(s, str))

            if not all_suffixes:
                continue

            suffix = self._rng.choice(all_suffixes)

            # Truncate root at variable point for more variance
            truncate_cfg = _require_cfg(self._cfg, 'truncate_root', 'base_generator.culture_blender')
            trunc_min_len = int(_require_cfg(truncate_cfg, 'min_length', 'base_generator.culture_blender.truncate_root'))
            if len(root) > trunc_min_len:
                cut_min = int(_require_cfg(truncate_cfg, 'cut_min', 'base_generator.culture_blender.truncate_root'))
                cut_max = int(_require_cfg(truncate_cfg, 'cut_max', 'base_generator.culture_blender.truncate_root'))
                cut_point = self._rng.randint(cut_min, min(cut_max, len(root)))
                root = root[:cut_point]

            # Build name with optional morphological operations
            connector_cfg = _require_cfg(self._cfg, 'connector', 'base_generator.culture_blender')
            vowels = _require_cfg(connector_cfg, 'vowels', 'base_generator.culture_blender.connector')
            if root[-1] in vowels and suffix[0] in vowels:
                connector = self._rng.choice(_require_cfg(connector_cfg, 'vowel_vowel', 'base_generator.culture_blender.connector'))
            elif root[-1] not in vowels and suffix[0] not in vowels:
                connector = self._rng.choice(_require_cfg(connector_cfg, 'consonant_consonant', 'base_generator.culture_blender.connector'))
            else:
                connector = ''

            name_str = root + connector + suffix

            # Apply random morphological operation (20% chance)
            morph_cfg = _require_cfg(self._cfg, 'morphology', 'base_generator.culture_blender')
            if self._rng.random() < float(_require_cfg(morph_cfg, 'apply_probability', 'base_generator.culture_blender.morphology')):
                op = self._rng.choice(_require_cfg(morph_cfg, 'operations', 'base_generator.culture_blender.morphology'))
                if op == 'mutate':
                    name_str = self._entropy.mutate(name_str, intensity=float(_require_cfg(morph_cfg, 'mutate_intensity', 'base_generator.culture_blender.morphology')))
                elif op == 'metathesis':
                    name_str = self._entropy.morphology.metathesis(name_str)

            # Checks
            length_cfg = _require_cfg(self._cfg, 'length', 'base_generator.culture_blender')
            min_len = int(_require_cfg(length_cfg, 'min', 'base_generator.culture_blender.length'))
            max_len = int(_require_cfg(length_cfg, 'max', 'base_generator.culture_blender.length'))
            if len(name_str) < min_len or len(name_str) > max_len:
                continue

            if name_str.lower() in seen:
                continue
            seen.add(name_str.lower())

            if check_hazards:
                hazard_result = self._hazard_checker.check(name_str)
                hazard_block_levels = set(_require_cfg(self._cfg, 'hazard_block_levels', 'base_generator.culture_blender'))
                if hazard_result.severity in hazard_block_levels:
                    continue

            scores = self._memorability.score(name_str, archetype)
            min_mem = _require_cfg(self._cfg, 'min_memorability_score', 'base_generator.culture_blender')
            if scores['overall'] < float(min_mem):
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
        self._cfg = load_base_generator_config().get('competitive_differentiator', {}) or {}
        if not self._cfg:
            raise ValueError("base_generator.competitive_differentiator must be set in base_generator.yaml")
        self._build_competitor_patterns()

    def _build_competitor_patterns(self):
        """Build patterns from competitor names."""
        self.competitor_onsets = set()
        self.competitor_endings = set()
        self.competitor_sounds = set()

        onset_lengths = _require_cfg(self._cfg, 'onset_lengths', 'base_generator.competitive_differentiator')
        ending_lengths = _require_cfg(self._cfg, 'ending_lengths', 'base_generator.competitive_differentiator')
        for comp in self.competitors:
            # Capture first N chars
            for length in onset_lengths:
                if len(comp) >= int(length):
                    self.competitor_onsets.add(comp[:int(length)])

            # Capture last N chars
            for length in ending_lengths:
                if len(comp) >= int(length):
                    self.competitor_endings.add(comp[-int(length):])

            # Add all characters
            self.competitor_sounds.update(comp)

    def is_distinct(self, name: str, threshold: float = None) -> bool:
        """
        Check if name is sufficiently distinct from competitors.

        Returns True if distinct enough, False if too similar.
        """
        if not self.competitors:
            return True

        if threshold is None:
            threshold = _require_cfg(self._cfg, 'default_threshold', 'base_generator.competitive_differentiator')

        name_lower = name.lower()
        similarity_score = 0.0
        weights = _require_cfg(self._cfg, 'weights', 'base_generator.competitive_differentiator')
        onset_weight = float(_require_cfg(weights, 'onset', 'base_generator.competitive_differentiator.weights'))
        ending_weight = float(_require_cfg(weights, 'ending', 'base_generator.competitive_differentiator.weights'))
        overlap_weight = float(_require_cfg(weights, 'overlap', 'base_generator.competitive_differentiator.weights'))
        substring_weight = float(_require_cfg(weights, 'substring', 'base_generator.competitive_differentiator.weights'))

        # Check onset similarity
        for onset in self.competitor_onsets:
            if name_lower.startswith(onset):
                similarity_score += onset_weight
                break

        # Check ending similarity
        for ending in self.competitor_endings:
            if name_lower.endswith(ending):
                similarity_score += ending_weight
                break

        # Check overall character overlap
        name_chars = set(name_lower)
        overlap = len(name_chars & self.competitor_sounds) / len(name_chars) if name_chars else 0
        similarity_score += overlap * overlap_weight

        # Check direct substring matches
        for comp in self.competitors:
            if comp in name_lower or name_lower in comp:
                similarity_score += substring_weight
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
    'load_base_generator_config',
    'load_culture_config',
]
