#!/usr/bin/env python3
"""
Brand Name Generator for German/English Markets
================================================
Generates phonetically pleasing brand names based on:
- Klangfarbe (sound color): hard vs soft consonants
- Silbenstruktur (syllable structure): 2-3 syllables optimal
- Konsonant-Vokal-Balance: alternating C-V patterns
- Lautmalerei (onomatopoeia): sound-meaning associations
- Semantische Konnotationen: positive associations

Target market: Camping & recreational vehicles (DC/DC converters)
"""

import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import argparse
import json
from pathlib import Path

from brandkit.generators.phonemes import load_rule_based_config


class SoundQuality(Enum):
    """Klangfarbe - Sound character categories"""
    HARD = "hard"      # k, t, p, g, d, b - energisch, stark
    SOFT = "soft"      # m, n, l, w, j - sanft, warm
    SHARP = "sharp"    # s, z, sch, x - scharf, dynamisch
    FLOWING = "flowing"  # r, f, v - fließend


class VowelQuality(Enum):
    """Vokalcharakter"""
    OPEN = "open"      # a, o - offen, kraftvoll
    CLOSED = "closed"  # i, u - geschlossen, fokussiert
    NEUTRAL = "neutral"  # e - neutral, verbindend


@dataclass
class Phoneme:
    """Ein einzelnes Phonem mit seinen Eigenschaften"""
    sound: str
    quality: SoundQuality | VowelQuality
    is_vowel: bool
    weight: float = 1.0  # Häufigkeitsgewichtung

    # Problematisch in bestimmten Sprachen?
    problematic_de: bool = False
    problematic_en: bool = False


@dataclass
class Syllable:
    """Eine Silbe als Baustein"""
    text: str
    pattern: str  # z.B. "CV", "CVC", "VC"
    stress: bool = False  # Betonung
    semantic_hint: Optional[str] = None  # Semantische Assoziation

    # Klangbewertung
    hardness: float = 0.5  # 0=weich, 1=hart
    brightness: float = 0.5  # 0=dunkel, 1=hell


@dataclass
class SemanticMorpheme:
    """Bedeutungstragende Wortteile"""
    text: str
    meaning: str
    category: str  # energy, travel, nature, tech, transformation
    origin: str  # latin, greek, english, german, fantasy
    syllables: int
    works_as_prefix: bool = True
    works_as_suffix: bool = True
    works_standalone: bool = False


# =============================================================================
# CONFIGURATION
# =============================================================================

_RULE_BASED_CFG = load_rule_based_config()
if not _RULE_BASED_CFG:
    raise ValueError("rule_based.yaml config missing or empty")


def _require(mapping: dict, key: str):
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"rule_based.yaml missing '{key}'")
    return value


def _require_nested(mapping: dict, *keys: str):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"rule_based.yaml missing '{'.'.join(keys)}'")
        current = current[key]
    return current


DEFAULTS = _require(_RULE_BASED_CFG, "defaults")
SCORING = _require(_RULE_BASED_CFG, "scoring")
PRONOUNCEABILITY = _require(_RULE_BASED_CFG, "pronounceability")
PHONETICS = _require(_RULE_BASED_CFG, "phonetics")
GENERATION_CFG = _require(_RULE_BASED_CFG, "generation")
CLI_CFG = _require(_RULE_BASED_CFG, "cli")


def _build_phonemes(entries: dict, quality_enum, is_vowel: bool) -> dict:
    if not entries:
        raise ValueError("phonetics entries missing in rule_based.yaml")
    result = {}
    for symbol, data in entries.items():
        quality = data.get("quality")
        weight = data.get("weight")
        if quality is None or weight is None:
            raise ValueError(f"phoneme '{symbol}' missing quality/weight in rule_based.yaml")
        result[symbol] = Phoneme(
            symbol,
            quality_enum(quality),
            is_vowel,
            float(weight),
        )
    return result


CONSONANTS = _build_phonemes(_require(PHONETICS, "consonants"), SoundQuality, False)
VOWELS = _build_phonemes(_require(PHONETICS, "vowels"), VowelQuality, True)
VOWEL_COMBINATIONS = _require(PHONETICS, "vowel_combinations")
GOOD_CLUSTERS = _require(PHONETICS, "good_clusters")

PROBLEMATIC_DE = _require(PRONOUNCEABILITY, "problematic_de")
PROBLEMATIC_EN = _require(PRONOUNCEABILITY, "problematic_en")
UNUSUAL_VOWELS_DE = _require(PRONOUNCEABILITY, "unusual_vowels_de")
CONSONANT_CLUSTER_MAX = _require(PRONOUNCEABILITY, "consonant_cluster_max")
CONSONANT_CHARS = _require(PRONOUNCEABILITY, "consonant_chars")
VOWEL_CHARS = _require(PRONOUNCEABILITY, "vowel_chars")
UMLAUT_CHARS = _require(PRONOUNCEABILITY, "umlaut_chars")

BLACKLIST_WORDS = _require(_RULE_BASED_CFG, "blacklist_words")
EXISTING_BRANDS = _require(_RULE_BASED_CFG, "existing_brands")
SEMANTIC_MORPHEMES = [SemanticMorpheme(**m) for m in _require(_RULE_BASED_CFG, "semantic_morphemes")]


# =============================================================================
# SILBEN-GENERATOR
# =============================================================================

class SyllableGenerator:
    """Generiert wohlklingende Silben nach phonetischen Regeln"""

    def __init__(self, target_hardness: Optional[float] = None):
        """
        target_hardness: 0.0 = sehr weich, 1.0 = sehr hart
        """
        if target_hardness is None:
            target_hardness = _require_nested(DEFAULTS, "target_hardness")
        self.target_hardness = target_hardness

    def _select_consonant(self, prefer_hard: bool = None) -> str:
        """Wählt einen Konsonanten basierend auf gewünschter Härte"""
        if prefer_hard is None:
            prefer_hard = random.random() < self.target_hardness

        if prefer_hard:
            candidates = [k for k, v in CONSONANTS.items()
                         if v.quality == SoundQuality.HARD]
        else:
            candidates = [k for k, v in CONSONANTS.items()
                         if v.quality in (SoundQuality.SOFT, SoundQuality.FLOWING)]

        weights = [CONSONANTS[c].weight for c in candidates]
        return random.choices(candidates, weights=weights)[0]

    def _select_vowel(self, prefer_open: bool = None) -> str:
        """Wählt einen Vokal"""
        if prefer_open is None:
            prefer_open_threshold = _require_nested(GENERATION_CFG, "prefer_open_threshold")
            prefer_open = random.random() > prefer_open_threshold

        # Manchmal Diphthong verwenden für weicheren Klang
        diphthong_prob = _require_nested(GENERATION_CFG, "diphthong_probability")
        if random.random() < diphthong_prob:
            return random.choice(VOWEL_COMBINATIONS)

        if prefer_open:
            candidates = [k for k, v in VOWELS.items()
                         if v.quality == VowelQuality.OPEN]
        else:
            candidates = list(VOWELS.keys())

        weights = [VOWELS[c].weight for c in candidates]
        return random.choices(candidates, weights=weights)[0]

    def generate_syllable(self, pattern: str = None) -> Syllable:
        """
        Generiert eine Silbe nach Muster.
        Patterns: CV, CVC, VC, V, CCV, CCVC
        """
        if pattern is None:
            patterns_cfg = _require_nested(GENERATION_CFG, "syllable_patterns")
            patterns = _require_nested(patterns_cfg, "patterns")
            weights = _require_nested(patterns_cfg, "weights")
            pattern = random.choices(patterns, weights=weights)[0]

        text = ""
        hardness_sum = 0
        consonant_count = 0

        for char in pattern:
            if char == 'C':
                c = self._select_consonant()
                text += c
                if CONSONANTS[c].quality == SoundQuality.HARD:
                    hardness_sum += 1
                consonant_count += 1
            elif char == 'V':
                text += self._select_vowel()

        avg_hardness = hardness_sum / max(consonant_count, 1)

        # Berechne Helligkeit basierend auf Vokalen
        brightness = _require_nested(GENERATION_CFG, "brightness_default")
        cv_vowels = _require_nested(SCORING, "cv_alternation", "vowels")
        vowels_in_text = [c for c in text if c in VOWELS or c in cv_vowels]
        if vowels_in_text:
            bright_vowels = sum(1 for v in vowels_in_text if v in 'ie')
            brightness = bright_vowels / len(vowels_in_text)

        return Syllable(
            text=text,
            pattern=pattern,
            hardness=avg_hardness,
            brightness=brightness
        )

    def generate_cluster_syllable(self) -> Syllable:
        """Generiert Silbe mit Konsonantencluster am Anfang"""
        onset = random.choice(GOOD_CLUSTERS['onset'])
        vowel = self._select_vowel()

        # Optional: Coda hinzufügen
        coda_prob = _require_nested(GENERATION_CFG, "cluster_coda_probability")
        if random.random() < coda_prob:
            coda = random.choice([c for c in CONSONANTS.keys()
                                 if CONSONANTS[c].quality != SoundQuality.HARD])
            text = onset + vowel + coda
            pattern = "CCVC"
        else:
            text = onset + vowel
            pattern = "CCV"

        cluster_hardness = _require_nested(GENERATION_CFG, "cluster_hardness_default")
        return Syllable(text=text, pattern=pattern, hardness=cluster_hardness)


# =============================================================================
# SCORING SYSTEM
# =============================================================================

@dataclass
class NameScore:
    """Bewertung eines generierten Namens"""
    name: str
    total_score: float

    # Einzelbewertungen (0-1)
    pronounceability_de: float = 0.0
    pronounceability_en: float = 0.0
    memorability: float = 0.0
    euphony: float = 0.0  # Wohlklang
    rhythm: float = 0.0
    semantic_fit: float = 0.0
    uniqueness: float = 0.0

    issues: list = field(default_factory=list)
    semantic_associations: list = field(default_factory=list)


class NameScorer:
    """Bewertet generierte Namen nach verschiedenen Kriterien"""

    def __init__(self):
        self.existing_brands = self._load_existing_brands()

    def _load_existing_brands(self) -> set:
        """Lädt bekannte Markennamen zum Vergleich"""
        return set(b.lower() for b in EXISTING_BRANDS)

    def _check_pronounceability_de(self, name: str) -> float:
        """Prüft Aussprechbarkeit für Deutsche"""
        pron_cfg = _require_nested(SCORING, "pronounceability")
        score = _require_nested(pron_cfg, "base")
        name_lower = name.lower()

        # Problematische Cluster für Deutsche
        de_cfg = _require_nested(pron_cfg, "de")
        de_penalty = _require_nested(de_cfg, "problematic_penalty")
        for prob in PROBLEMATIC_DE:
            if prob in name_lower:
                score += de_penalty

        # Sehr lange Konsonantencluster
        cluster_penalty = _require_nested(de_cfg, "cluster_penalty")
        cluster_pattern = rf'[{CONSONANT_CHARS}]{{{CONSONANT_CLUSTER_MAX},}}'
        if re.search(cluster_pattern, name_lower):
            score += cluster_penalty

        # Doppelvokale die im Deutschen unüblich sind
        unusual_penalty = _require_nested(de_cfg, "unusual_vowel_penalty")
        for uv in UNUSUAL_VOWELS_DE:
            if uv in name_lower:
                score += unusual_penalty

        return max(0, score)

    def _check_pronounceability_en(self, name: str) -> float:
        """Prüft Aussprechbarkeit für Englischsprachige"""
        pron_cfg = _require_nested(SCORING, "pronounceability")
        score = _require_nested(pron_cfg, "base")
        name_lower = name.lower()

        # Problematische Cluster für Englischsprachige
        en_cfg = _require_nested(pron_cfg, "en")
        en_penalty = _require_nested(en_cfg, "problematic_penalty")
        for prob in PROBLEMATIC_EN:
            if prob in name_lower:
                score += en_penalty

        # Deutsche Umlaute (sollten nicht vorkommen, aber sicher ist sicher)
        umlaut_penalty = _require_nested(en_cfg, "umlaut_penalty")
        if any(c in name_lower for c in UMLAUT_CHARS):
            score += umlaut_penalty

        return max(0, score)

    def _check_memorability(self, name: str) -> float:
        """Bewertet Einprägsamkeit"""
        mem_cfg = _require_nested(SCORING, "memorability")
        score = _require_nested(mem_cfg, "base")

        # Optimale Länge: 4-7 Zeichen (STRENGER)
        length = len(name)
        length_cfg = _require_nested(SCORING, "length")
        if length < 3:
            score += _require_nested(length_cfg, "penalty_lt3")
        elif length < 4:
            score += _require_nested(length_cfg, "penalty_lt4")
        elif length <= 6:
            score += _require_nested(length_cfg, "bonus_4_6")
        elif length == 7:
            score += _require_nested(length_cfg, "bonus_7")
        elif length == 8:
            score += _require_nested(length_cfg, "penalty_8")
        elif length == 9:
            score += _require_nested(length_cfg, "penalty_9")
        elif length >= 10:
            score += _require_nested(length_cfg, "penalty_ge10_base") + _require_nested(length_cfg, "penalty_ge10_per_char") * (length - 10)

        # Silbenzahl (2 optimal, 3 okay)
        syllable_count = self._count_syllables(name)
        syll_cfg = _require_nested(SCORING, "syllables")
        if syllable_count == 2:
            score += _require_nested(syll_cfg, "bonus_2")
        elif syllable_count == 3:
            score += _require_nested(syll_cfg, "penalty_3")
        elif syllable_count >= 4:
            score += _require_nested(syll_cfg, "penalty_ge4")

        # Wiederholende Elemente erhöhen Einprägsamkeit
        if self._has_repetition(name):
            score += _require_nested(mem_cfg, "repetition_bonus")

        return min(1.0, max(0, score))

    def _count_syllables(self, name: str) -> int:
        """Zählt ungefähre Silbenzahl"""
        name_lower = name.lower()
        # Vereinfachte Zählung: Vokale zählen, aufeinanderfolgende als eine
        vowels = VOWEL_CHARS
        count = 0
        prev_was_vowel = False

        for char in name_lower:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        return max(1, count)

    def _has_repetition(self, name: str) -> bool:
        """Prüft auf einprägsame Wiederholungen (wie Coca-Cola)"""
        name_lower = name.lower()
        repetition_cfg = _require_nested(SCORING, "repetition")
        min_length = _require_nested(repetition_cfg, "min_length")
        vowels = _require_nested(repetition_cfg, "vowels")

        # Alliteration
        if len(name_lower) >= min_length:
            first_consonant = None
            for c in name_lower:
                if c not in vowels:
                    if first_consonant is None:
                        first_consonant = c
                    elif c == first_consonant:
                        return True
                    break

        # Silbenwiederholung
        if len(name_lower) >= min_length:
            half = len(name_lower) // 2
            if name_lower[:half] == name_lower[half:half*2]:
                return True

        return False

    def _check_euphony(self, name: str) -> float:
        """Bewertet Wohlklang"""
        euph_cfg = _require_nested(SCORING, "euphony")
        score = _require_nested(euph_cfg, "base")
        name_lower = name.lower()

        # Gute Konsonant-Vokal-Abwechslung
        cv_alternation = self._calculate_cv_alternation(name_lower)
        score += cv_alternation * _require_nested(euph_cfg, "cv_alternation_weight")

        # Weiche Konsonanten für angenehmeren Klang
        soft_consonants = sum(1 for c in name_lower
                             if c in CONSONANTS and
                             CONSONANTS[c].quality in (SoundQuality.SOFT, SoundQuality.FLOWING))
        total_consonants = sum(1 for c in name_lower if c in CONSONANTS)
        if total_consonants > 0:
            soft_ratio = soft_consonants / total_consonants
            score += soft_ratio * _require_nested(euph_cfg, "soft_consonant_weight")

        # Offene Vokale klingen voller
        open_vowel_chars = _require_nested(euph_cfg, "open_vowels")
        vowel_chars = _require_nested(euph_cfg, "vowel_chars")
        open_vowels = sum(1 for c in name_lower if c in open_vowel_chars)
        total_vowels = sum(1 for c in name_lower if c in vowel_chars)
        if total_vowels > 0:
            open_ratio = open_vowels / total_vowels
            score += open_ratio * _require_nested(euph_cfg, "open_vowel_weight")

        return min(1.0, score)

    def _calculate_cv_alternation(self, name: str) -> float:
        """Berechnet wie gut Konsonanten und Vokale abwechseln"""
        cv_cfg = _require_nested(SCORING, "cv_alternation")
        short_name_score = _require_nested(cv_cfg, "short_name_score")
        vowels = _require_nested(cv_cfg, "vowels")
        if len(name) < 2:
            return short_name_score

        alternations = 0

        for i in range(1, len(name)):
            prev_is_vowel = name[i-1] in vowels
            curr_is_vowel = name[i] in vowels
            if prev_is_vowel != curr_is_vowel:
                alternations += 1

        return alternations / (len(name) - 1)

    def _check_rhythm(self, name: str) -> float:
        """Bewertet rhythmische Qualität"""
        rhythm_cfg = _require_nested(SCORING, "rhythm")
        syllables = self._count_syllables(name)
        length = len(name)

        # 2 Silben = optimal
        if syllables == 2:
            return _require_nested(rhythm_cfg, "two_syllables")
        elif syllables == 1:
            return _require_nested(rhythm_cfg, "one_syllable")
        elif syllables == 3:
            # 3 Silben nur gut wenn Name kurz genug
            max_short = _require_nested(rhythm_cfg, "max_short_length")
            if length <= max_short:
                return _require_nested(rhythm_cfg, "three_syllables_short")
            return _require_nested(rhythm_cfg, "three_syllables_long")
        elif syllables == 4:
            return _require_nested(rhythm_cfg, "four_syllables")
        return _require_nested(rhythm_cfg, "five_plus")

    def _check_semantic_fit(self, name: str, morphemes_used: list) -> tuple[float, list]:
        """Bewertet semantische Passung zum Camping/Energie-Kontext"""
        sem_cfg = _require_nested(SCORING, "semantic_fit")
        score = _require_nested(sem_cfg, "base")
        category_weights = _require_nested(sem_cfg, "category_weights")
        associations = []

        for morpheme in morphemes_used:
            if isinstance(morpheme, SemanticMorpheme):
                if morpheme.category in category_weights:
                    score += category_weights[morpheme.category]
                    associations.append(f"{morpheme.text}: {morpheme.meaning}")

        return min(1.0, score), associations

    def _check_uniqueness(self, name: str) -> float:
        """Prüft ob der Name einzigartig ist"""
        uniq_cfg = _require_nested(SCORING, "uniqueness")
        name_lower = name.lower()

        # Exakte Übereinstimmung mit bekannter Marke
        if name_lower in self.existing_brands:
            return _require_nested(uniq_cfg, "exact_match_score")

        # Ähnlichkeit zu bekannten Marken (einfacher Check)
        for brand in self.existing_brands:
            if name_lower in brand or brand in name_lower:
                return _require_nested(uniq_cfg, "substring_score")
            # Levenshtein wäre besser, aber für Einfachheit:
            overlap_threshold = _require_nested(uniq_cfg, "overlap_threshold")
            overlap_score = _require_nested(uniq_cfg, "overlap_score")
            if len(set(name_lower) & set(brand)) / max(len(name_lower), len(brand)) > overlap_threshold:
                return overlap_score

        return _require_nested(uniq_cfg, "default_score")

    def _check_blacklist(self, name: str) -> list:
        """Prüft auf problematische Wörter"""
        issues = []
        name_lower = name.lower()

        for word in BLACKLIST_WORDS:
            if word in name_lower:
                issues.append(f"Contains problematic word: '{word}'")

        return issues

    def score_name(self, name: str, morphemes_used: list = None) -> NameScore:
        """Bewertet einen Namen vollständig"""
        if morphemes_used is None:
            morphemes_used = []

        issues = self._check_blacklist(name)
        # Cross-linguistic pronounceability gate (EN/DE)
        try:
            from brandkit.generators.phonemes import is_pronounceable
            ok, reason = is_pronounceable(name, markets='en_de')
            if not ok:
                issues.append(f"pronounceability:{reason}")
        except Exception:
            pass

        pronounce_de = self._check_pronounceability_de(name)
        pronounce_en = self._check_pronounceability_en(name)
        memorability = self._check_memorability(name)
        euphony = self._check_euphony(name)
        rhythm = self._check_rhythm(name)
        semantic_fit, associations = self._check_semantic_fit(name, morphemes_used)
        uniqueness = self._check_uniqueness(name)

        # Gewichtete Gesamtbewertung (Memorability & Rhythm wichtiger!)
        weights = _require_nested(SCORING, "weights")
        total = (
            pronounce_de * _require_nested(weights, "pronounceability_de") +
            pronounce_en * _require_nested(weights, "pronounceability_en") +
            memorability * _require_nested(weights, "memorability") +
            euphony * _require_nested(weights, "euphony") +
            rhythm * _require_nested(weights, "rhythm") +
            semantic_fit * _require_nested(weights, "semantic_fit") +
            uniqueness * _require_nested(weights, "uniqueness")
        )

        # Malus für Blacklist-Treffer
        if issues:
            total *= _require_nested(SCORING, "blacklist_penalty_multiplier")

        return NameScore(
            name=name,
            total_score=total,
            pronounceability_de=pronounce_de,
            pronounceability_en=pronounce_en,
            memorability=memorability,
            euphony=euphony,
            rhythm=rhythm,
            semantic_fit=semantic_fit,
            uniqueness=uniqueness,
            issues=issues,
            semantic_associations=associations
        )


# =============================================================================
# NAME GENERATOR
# =============================================================================

class BrandNameGenerator:
    """Hauptklasse für die Markennamen-Generierung"""

    def __init__(self,
                 target_hardness: Optional[float] = None,
                 prefer_semantic: Optional[bool] = None):
        if target_hardness is None:
            target_hardness = _require_nested(DEFAULTS, "target_hardness")
        if prefer_semantic is None:
            prefer_semantic = _require_nested(DEFAULTS, "prefer_semantic")
        self.syllable_gen = SyllableGenerator(target_hardness)
        self.scorer = NameScorer()
        self.prefer_semantic = prefer_semantic

    def _get_morphemes_by_category(self, category: str) -> list:
        """Holt Morpheme einer bestimmten Kategorie"""
        return [m for m in SEMANTIC_MORPHEMES if m.category == category]

    def _get_suffix_morphemes(self) -> list:
        """Holt alle als Suffix verwendbaren Morpheme"""
        return [m for m in SEMANTIC_MORPHEMES if m.works_as_suffix]

    def _get_prefix_morphemes(self) -> list:
        """Holt alle als Präfix verwendbaren Morpheme"""
        return [m for m in SEMANTIC_MORPHEMES if m.works_as_prefix]

    def generate_semantic_name(self, categories: list = None) -> tuple[str, list]:
        """Generiert Namen basierend auf semantischen Morphemen"""
        if categories is None:
            categories = _require_nested(GENERATION_CFG, "categories_default")

        morphemes_used = []

        # Strategie wählen (prefix_suffix bevorzugt = kürzere Namen)
        strategy_weights = _require_nested(GENERATION_CFG, "semantic_strategy_weights")
        strategies = list(strategy_weights.keys())
        weights = [strategy_weights[s] for s in strategies]
        strategy = random.choices(strategies, weights=weights)[0]

        if strategy == 'prefix_suffix':
            # Präfix + Suffix (z.B. "Voltix", "Trekora")
            fantasy_category = _require_nested(GENERATION_CFG, "fantasy_category")
            prefix_pool = [m for m in self._get_prefix_morphemes()
                         if m.category in categories or m.category == fantasy_category]
            suffix_pool = self._get_suffix_morphemes()

            prefix = random.choice(prefix_pool)
            suffix = random.choice([s for s in suffix_pool if s.category == fantasy_category])

            name = prefix.text.capitalize() + suffix.text
            morphemes_used = [prefix, suffix]

        elif strategy == 'double_morpheme':
            # Zwei bedeutungstragende Morpheme (z.B. "Solflux", "Volterra")
            # NUR kurze Morpheme verwenden (max 4 Zeichen pro Teil)
            double_cfg = _require_nested(GENERATION_CFG, "semantic_double_morpheme")
            max_first_len = _require_nested(double_cfg, "max_first_length")
            max_second_len = _require_nested(double_cfg, "max_second_length")
            first_pool = [m for m in self._get_prefix_morphemes()
                         if m.category in categories and len(m.text) <= max_first_len]
            second_pool = [m for m in self._get_suffix_morphemes()
                          if m.category in categories and len(m.text) <= max_second_len]

            if not first_pool or not second_pool:
                # Fallback zu prefix_suffix
                return self.generate_semantic_name(categories)

            first = random.choice(first_pool)
            second = random.choice([s for s in second_pool if s != first])

            # Verbindung glätten
            name = self._smooth_join(first.text, second.text)

            # Wenn Name zu lang, nochmal versuchen
            max_name_len = _require_nested(double_cfg, "max_name_length")
            if len(name) > max_name_len:
                return self.generate_semantic_name(categories)

            morphemes_used = [first, second]

        elif strategy == 'morpheme_fantasy':
            # Morphem + Fantasy-Silbe
            fantasy_category = _require_nested(GENERATION_CFG, "fantasy_category")
            morpheme = random.choice([m for m in SEMANTIC_MORPHEMES
                                     if m.category in categories])
            fantasy_syl = self.syllable_gen.generate_syllable()

            if morpheme.works_as_prefix:
                name = morpheme.text.capitalize() + fantasy_syl.text
            else:
                name = fantasy_syl.text.capitalize() + morpheme.text

            morphemes_used = [morpheme]

        else:  # standalone_modified
            # Standalone Morphem leicht modifiziert
            standalone = [m for m in SEMANTIC_MORPHEMES
                         if m.works_standalone and m.category in categories]
            if standalone:
                base = random.choice(standalone)
                # Leichte Modifikation
                modifications = _require_nested(GENERATION_CFG, "standalone_modifications")
                mod = random.choice(modifications)
                name = self._apply_modification(base.text, mod)
                morphemes_used = [base]
            else:
                # Fallback
                return self.generate_phonetic_name()

        return name, morphemes_used

    def _smooth_join(self, first: str, second: str) -> str:
        """Verbindet zwei Morpheme fließend"""
        # Wenn erstes mit Vokal endet und zweites mit Vokal beginnt
        vowels = _require_nested(SCORING, "repetition", "vowels")
        if first[-1] in vowels and second[0] in vowels:
            # Einen Vokal weglassen
            return first[:-1].capitalize() + second

        # Wenn erstes mit Konsonant endet und zweites mit Konsonant beginnt
        if first[-1] not in vowels and second[0] not in vowels:
            # Bindevokal einfügen
            connectors = _require_nested(GENERATION_CFG, "join_connectors")
            connector = random.choice(connectors)
            return first.capitalize() + connector + second

        return first.capitalize() + second

    def _apply_modification(self, text: str, mod: dict) -> str:
        mod_type = mod.get("type")
        value = mod.get("value", "")
        if not mod_type:
            raise ValueError("standalone_modifications entries must include type")
        if mod_type == "suffix":
            result = text + value
        elif mod_type == "suffix_capitalize":
            result = text.capitalize() + value
        elif mod_type == "prefix":
            result = value + text
        else:
            raise ValueError(f"Unknown modification type: {mod_type}")
        return result.capitalize()

    def generate_phonetic_name(self, syllable_count: int = None) -> tuple[str, list]:
        """Generiert rein phonetisch basierten Namen"""
        if syllable_count is None:
            phonetic_cfg = _require_nested(GENERATION_CFG, "phonetic")
            syllable_weights = _require_nested(phonetic_cfg, "syllable_count_weights")
            counts = _require_nested(syllable_weights, "counts")
            weights = _require_nested(syllable_weights, "weights")
            syllable_count = random.choices(counts, weights=weights)[0]

        syllables = []

        # Erste Silbe: oft mit Cluster für markanten Anfang
        phonetic_cfg = _require_nested(GENERATION_CFG, "phonetic")
        cluster_start_probability = _require_nested(phonetic_cfg, "cluster_start_probability")
        first_pattern = _require_nested(phonetic_cfg, "first_syllable_pattern")
        if random.random() < cluster_start_probability:
            syllables.append(self.syllable_gen.generate_cluster_syllable())
        else:
            syllables.append(self.syllable_gen.generate_syllable(first_pattern))

        # Mittlere Silben
        for _ in range(syllable_count - 2):
            syllables.append(self.syllable_gen.generate_syllable())

        # Letzte Silbe: oft offen (endet auf Vokal) für weichen Ausklang
        if syllable_count > 1:
            last_patterns = _require_nested(phonetic_cfg, "last_syllable_patterns")
            patterns = _require_nested(last_patterns, "patterns")
            weights = _require_nested(last_patterns, "weights")
            last_pattern = random.choices(patterns, weights=weights)[0]
            last_syl = self.syllable_gen.generate_syllable(last_pattern)
            syllables.append(last_syl)

        name = ''.join(s.text for s in syllables).capitalize()
        return name, []

    def generate_hybrid_name(self) -> tuple[str, list]:
        """Kombiniert semantische und phonetische Elemente"""
        hybrid_cfg = _require_nested(GENERATION_CFG, "hybrid")
        suffix_probability = _require_nested(hybrid_cfg, "phonetic_suffix_probability")
        if random.random() < suffix_probability:
            # Phonetisch + Suffix
            syllable_count = _require_nested(hybrid_cfg, "phonetic_syllable_count")
            phonetic, _ = self.generate_phonetic_name(syllable_count=syllable_count)
            suffix_category = _require_nested(hybrid_cfg, "suffix_category")
            suffix = random.choice([m for m in SEMANTIC_MORPHEMES
                                   if m.category == suffix_category and m.works_as_suffix])
            name = phonetic[:-1] + suffix.text  # Letzten Buchstaben ersetzen
            return name.capitalize(), [suffix]
        else:
            # Präfix + Phonetisch
            prefix_syllables = _require_nested(hybrid_cfg, "prefix_syllables")
            prefix = random.choice([m for m in SEMANTIC_MORPHEMES
                                   if m.works_as_prefix and m.syllables == prefix_syllables])
            phonetic_pattern = _require_nested(hybrid_cfg, "phonetic_pattern")
            phonetic_syl = self.syllable_gen.generate_syllable(phonetic_pattern)
            name = prefix.text + phonetic_syl.text
            return name.capitalize(), [prefix]

    def generate_name(self) -> tuple[str, list]:
        """Generiert einen Namen mit gemischter Strategie"""
        if self.prefer_semantic:
            weights_cfg = _require_nested(GENERATION_CFG, "prefer_semantic_strategy_weights")
        else:
            weights_cfg = _require_nested(GENERATION_CFG, "no_semantic_strategy_weights")
        strategies = list(weights_cfg.keys())
        weights = [weights_cfg[s] for s in strategies]
        strategy = random.choices(strategies, weights=weights)[0]

        if strategy == 'semantic':
            return self.generate_semantic_name()
        elif strategy == 'phonetic':
            return self.generate_phonetic_name()
        else:
            return self.generate_hybrid_name()

    def generate_batch(self,
                       count: Optional[int] = None,
                       min_score: Optional[float] = None,
                       categories: list = None) -> list[NameScore]:
        """Generiert eine Batch von Namen und filtert nach Score"""
        if categories is None:
            categories = _require_nested(GENERATION_CFG, "categories_default")
        if count is None:
            count = _require_nested(GENERATION_CFG, "batch_count")
        if min_score is None:
            min_score = _require_nested(DEFAULTS, "min_score")
        results = []
        attempts = 0
        max_attempts_multiplier = _require_nested(GENERATION_CFG, "max_attempts_multiplier")
        max_attempts = count * max_attempts_multiplier

        while len(results) < count and attempts < max_attempts:
            attempts += 1

            semantic_bias_probability = _require_nested(GENERATION_CFG, "semantic_bias_probability")
            if categories and random.random() < semantic_bias_probability:
                name, morphemes = self.generate_semantic_name(categories)
            else:
                name, morphemes = self.generate_name()

            # Duplikate vermeiden
            if any(r.name.lower() == name.lower() for r in results):
                continue

            # Harter Längenfilter: Max 9 Zeichen
            max_name_length = _require_nested(GENERATION_CFG, "max_name_length")
            if len(name) > max_name_length:
                continue

            score = self.scorer.score_name(name, morphemes)

            if score.total_score >= min_score and not score.issues:
                results.append(score)

        # Nach Score sortieren
        results.sort(key=lambda x: x.total_score, reverse=True)
        return results


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate phonetically pleasing brand names for DE/EN markets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --count 20 --min-score 0.7
  %(prog)s --categories energy travel --hardness 0.3
  %(prog)s --output names.json --format json
        """
    )

    parser.add_argument(
        '-n', '--count',
        type=int,
        default=_require_nested(CLI_CFG, "default_count"),
        help='Number of names to generate'
    )

    parser.add_argument(
        '-m', '--min-score',
        type=float,
        default=_require_nested(CLI_CFG, "default_min_score"),
        help='Minimum score threshold (0-1)'
    )

    parser.add_argument(
        '-c', '--categories',
        nargs='+',
        choices=_require_nested(CLI_CFG, "available_categories"),
        default=_require_nested(CLI_CFG, "default_categories"),
        help='Semantic categories to focus on'
    )

    parser.add_argument(
        '--hardness',
        type=float,
        default=_require_nested(CLI_CFG, "default_hardness"),
        help='Sound hardness (0=soft, 1=hard)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path'
    )

    parser.add_argument(
        '-f', '--format',
        choices=_require_nested(CLI_CFG, "formats"),
        default=_require_nested(CLI_CFG, "default_format"),
        help='Output format'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed scoring information'
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Generator erstellen
    generator = BrandNameGenerator(
        target_hardness=args.hardness,
        prefer_semantic=True
    )

    print(f"Generating {args.count} brand names...")
    print(f"Categories: {', '.join(args.categories)}")
    print(f"Sound hardness: {args.hardness}")
    print(f"Minimum score: {args.min_score}")
    print("-" * 60)

    # Namen generieren
    results = generator.generate_batch(
        count=args.count,
        min_score=args.min_score,
        categories=args.categories
    )

    # Ausgabe formatieren
    if args.format == 'json':
        output_data = []
        for r in results:
            output_data.append({
                'name': r.name,
                'total_score': round(r.total_score, 3),
                'scores': {
                    'pronounceability_de': round(r.pronounceability_de, 3),
                    'pronounceability_en': round(r.pronounceability_en, 3),
                    'memorability': round(r.memorability, 3),
                    'euphony': round(r.euphony, 3),
                    'rhythm': round(r.rhythm, 3),
                    'semantic_fit': round(r.semantic_fit, 3),
                    'uniqueness': round(r.uniqueness, 3),
                },
                'associations': r.semantic_associations
            })

        output_str = json.dumps(output_data, indent=2, ensure_ascii=False)

    elif args.format == 'csv':
        lines = ['name,total_score,pronounce_de,pronounce_en,memorability,euphony,rhythm,semantic,unique']
        for r in results:
            lines.append(
                f'{r.name},{r.total_score:.3f},{r.pronounceability_de:.3f},'
                f'{r.pronounceability_en:.3f},{r.memorability:.3f},{r.euphony:.3f},'
                f'{r.rhythm:.3f},{r.semantic_fit:.3f},{r.uniqueness:.3f}'
            )
        output_str = '\n'.join(lines)

    else:  # text
        lines = []
        for i, r in enumerate(results, 1):
            if args.verbose:
                lines.append(f"\n{i:2}. {r.name:<15} Score: {r.total_score:.2f}")
                lines.append(f"    Aussprechbarkeit DE: {r.pronounceability_de:.2f}  EN: {r.pronounceability_en:.2f}")
                lines.append(f"    Einprägsamkeit: {r.memorability:.2f}  Wohlklang: {r.euphony:.2f}")
                lines.append(f"    Rhythmus: {r.rhythm:.2f}  Semantik: {r.semantic_fit:.2f}")
                if r.semantic_associations:
                    lines.append(f"    Assoziationen: {', '.join(r.semantic_associations)}")
            else:
                assoc = f" ({', '.join(r.semantic_associations)})" if r.semantic_associations else ""
                lines.append(f"{i:2}. {r.name:<15} {r.total_score:.2f}{assoc}")

        output_str = '\n'.join(lines)

    # Ausgabe
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_str, encoding='utf-8')
        print(f"\nResults saved to: {args.output}")
    else:
        print(output_str)

    print(f"\n{'=' * 60}")
    print(f"Generated {len(results)} names with score >= {args.min_score}")


if __name__ == '__main__':
    main()
