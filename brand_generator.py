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
# PHONETISCHE BAUSTEINE
# =============================================================================

# Konsonanten mit Eigenschaften
CONSONANTS = {
    # Harte Konsonanten (energisch, stark)
    'k': Phoneme('k', SoundQuality.HARD, False, 1.0),
    't': Phoneme('t', SoundQuality.HARD, False, 1.2),
    'p': Phoneme('p', SoundQuality.HARD, False, 0.9),
    'g': Phoneme('g', SoundQuality.HARD, False, 0.8),
    'd': Phoneme('d', SoundQuality.HARD, False, 1.0),
    'b': Phoneme('b', SoundQuality.HARD, False, 0.9),

    # Weiche Konsonanten (sanft, warm)
    'm': Phoneme('m', SoundQuality.SOFT, False, 1.1),
    'n': Phoneme('n', SoundQuality.SOFT, False, 1.3),
    'l': Phoneme('l', SoundQuality.SOFT, False, 1.2),
    'w': Phoneme('w', SoundQuality.SOFT, False, 0.7),
    'j': Phoneme('j', SoundQuality.SOFT, False, 0.5),

    # Scharfe Konsonanten (dynamisch)
    's': Phoneme('s', SoundQuality.SHARP, False, 1.1),
    'z': Phoneme('z', SoundQuality.SHARP, False, 0.6),
    'x': Phoneme('x', SoundQuality.SHARP, False, 0.4),

    # Fließende Konsonanten
    'r': Phoneme('r', SoundQuality.FLOWING, False, 1.0),
    'f': Phoneme('f', SoundQuality.FLOWING, False, 0.8),
    'v': Phoneme('v', SoundQuality.FLOWING, False, 0.7),
    'h': Phoneme('h', SoundQuality.FLOWING, False, 0.5),
}

# Vokale mit Eigenschaften
VOWELS = {
    'a': Phoneme('a', VowelQuality.OPEN, True, 1.3),
    'o': Phoneme('o', VowelQuality.OPEN, True, 1.0),
    'i': Phoneme('i', VowelQuality.CLOSED, True, 1.2),
    'u': Phoneme('u', VowelQuality.CLOSED, True, 0.8),
    'e': Phoneme('e', VowelQuality.NEUTRAL, True, 1.4),
}

# Diphthonge und Vokalverbindungen (für weicheren Klang)
VOWEL_COMBINATIONS = ['ai', 'au', 'ei', 'eu', 'ou', 'oo', 'ee', 'ia', 'io', 'ea']

# Konsonantencluster die gut klingen (DE+EN kompatibel)
GOOD_CLUSTERS = {
    'onset': ['tr', 'kr', 'pr', 'br', 'dr', 'gr', 'fr', 'fl', 'bl', 'gl', 'pl', 'kl', 'sl', 'sp', 'st', 'sk'],
    'coda': ['nt', 'nd', 'nk', 'lt', 'ld', 'rt', 'rd', 'rk', 'st', 'sk', 'mp', 'lm', 'rm'],
}

# Problematische Kombinationen
PROBLEMATIC_DE = ['th', 'ck', 'sh']  # Schwer für Deutsche oder anders ausgesprochen
PROBLEMATIC_EN = ['ch', 'sch', 'pf', 'tsch']  # Schwer für Englischsprachige

# Wörter die in DE oder EN negativ konnotiert sind
BLACKLIST_WORDS = [
    'mist', 'gift', 'bad', 'tot', 'kot', 'piss', 'dick', 'ass', 'fart', 'crap',
    'nazi', 'pedo', 'rape', 'kill', 'die', 'dead', 'shit', 'fuck', 'damn',
    'nova',  # "no va" = geht nicht (Spanisch, aber bekannt)
    'aids', 'hiv', 'std',
]


# =============================================================================
# SEMANTISCHE MORPHEME FÜR CAMPING/ENERGIE-KONTEXT
# =============================================================================

SEMANTIC_MORPHEMES = [
    # ENERGIE & TRANSFORMATION
    SemanticMorpheme('volt', 'electrical unit', 'energy', 'latin', 1, True, True, True),
    SemanticMorpheme('amp', 'current', 'energy', 'latin', 1, True, True, True),
    SemanticMorpheme('flux', 'flow', 'energy', 'latin', 1, True, True, True),
    SemanticMorpheme('dyn', 'power', 'energy', 'greek', 1, True, False, False),
    SemanticMorpheme('erg', 'energy/work', 'energy', 'greek', 1, False, True, False),
    SemanticMorpheme('trans', 'across/change', 'transformation', 'latin', 1, True, False, False),
    SemanticMorpheme('morph', 'form/change', 'transformation', 'greek', 1, True, True, False),
    SemanticMorpheme('vert', 'turn', 'transformation', 'latin', 1, False, True, False),
    SemanticMorpheme('pulse', 'beat/push', 'energy', 'latin', 1, True, True, True),
    SemanticMorpheme('spark', 'ignition', 'energy', 'english', 1, True, True, True),
    SemanticMorpheme('flow', 'movement', 'energy', 'english', 1, True, True, True),
    SemanticMorpheme('wave', 'oscillation', 'energy', 'english', 1, True, True, True),

    # REISE & BEWEGUNG
    SemanticMorpheme('via', 'way/path', 'travel', 'latin', 2, True, True, True),
    SemanticMorpheme('trek', 'journey', 'travel', 'dutch', 1, True, True, True),
    SemanticMorpheme('nav', 'navigation', 'travel', 'latin', 1, True, False, False),
    SemanticMorpheme('voy', 'voyage', 'travel', 'french', 1, True, False, False),
    SemanticMorpheme('road', 'path', 'travel', 'english', 1, True, True, True),
    SemanticMorpheme('trail', 'path', 'travel', 'english', 1, True, True, True),
    SemanticMorpheme('quest', 'search/journey', 'travel', 'latin', 1, True, True, True),
    SemanticMorpheme('roam', 'wander', 'travel', 'english', 1, True, True, True),
    SemanticMorpheme('nomad', 'wanderer', 'travel', 'greek', 2, True, True, True),
    SemanticMorpheme('wander', 'travel', 'travel', 'german', 2, True, True, True),

    # NATUR & OUTDOOR
    SemanticMorpheme('terra', 'earth', 'nature', 'latin', 2, True, True, True),
    SemanticMorpheme('sol', 'sun', 'nature', 'latin', 1, True, True, True),
    SemanticMorpheme('lux', 'light', 'nature', 'latin', 1, True, True, True),
    SemanticMorpheme('aura', 'air/atmosphere', 'nature', 'latin', 2, True, True, True),
    SemanticMorpheme('aqua', 'water', 'nature', 'latin', 2, True, True, True),
    SemanticMorpheme('silva', 'forest', 'nature', 'latin', 2, True, True, True),
    SemanticMorpheme('mont', 'mountain', 'nature', 'latin', 1, True, False, False),
    SemanticMorpheme('sky', 'heaven', 'nature', 'english', 1, True, True, True),
    SemanticMorpheme('peak', 'summit', 'nature', 'english', 1, True, True, True),
    SemanticMorpheme('wild', 'untamed', 'nature', 'english', 1, True, True, True),
    SemanticMorpheme('aurora', 'dawn', 'nature', 'latin', 3, True, True, True),
    SemanticMorpheme('zephyr', 'gentle wind', 'nature', 'greek', 2, True, True, True),

    # TECHNIK & ZUVERLÄSSIGKEIT
    SemanticMorpheme('core', 'center', 'tech', 'latin', 1, True, True, True),
    SemanticMorpheme('max', 'maximum', 'tech', 'latin', 1, True, True, True),
    SemanticMorpheme('pro', 'professional', 'tech', 'latin', 1, True, False, False),
    SemanticMorpheme('neo', 'new', 'tech', 'greek', 2, True, False, False),
    SemanticMorpheme('prime', 'first/best', 'tech', 'latin', 1, True, True, True),
    SemanticMorpheme('apex', 'peak/top', 'tech', 'latin', 2, True, True, True),
    SemanticMorpheme('link', 'connection', 'tech', 'english', 1, True, True, True),
    SemanticMorpheme('sync', 'synchronize', 'tech', 'greek', 1, True, True, True),
    SemanticMorpheme('hub', 'center', 'tech', 'english', 1, True, True, True),

    # FANTASIE-SILBEN (wohlklingend, ohne Bedeutung)
    SemanticMorpheme('ix', 'suffix (tech feel)', 'fantasy', 'fantasy', 1, False, True, False),
    SemanticMorpheme('ia', 'suffix (place)', 'fantasy', 'fantasy', 2, False, True, False),
    SemanticMorpheme('on', 'suffix (tech)', 'fantasy', 'fantasy', 1, False, True, False),
    SemanticMorpheme('ex', 'suffix (dynamic)', 'fantasy', 'fantasy', 1, False, True, False),
    SemanticMorpheme('um', 'suffix (latin feel)', 'fantasy', 'fantasy', 1, False, True, False),
    SemanticMorpheme('us', 'suffix (latin)', 'fantasy', 'fantasy', 1, False, True, False),
    SemanticMorpheme('or', 'suffix (agent)', 'fantasy', 'fantasy', 1, False, True, False),
    SemanticMorpheme('er', 'suffix (agent)', 'fantasy', 'fantasy', 1, False, True, False),
    SemanticMorpheme('al', 'suffix (quality)', 'fantasy', 'fantasy', 1, False, True, False),
    SemanticMorpheme('ar', 'suffix (quality)', 'fantasy', 'fantasy', 1, False, True, False),
    SemanticMorpheme('an', 'suffix (belonging)', 'fantasy', 'fantasy', 1, False, True, False),
    SemanticMorpheme('en', 'suffix (made of)', 'fantasy', 'fantasy', 1, False, True, False),
]


# =============================================================================
# SILBEN-GENERATOR
# =============================================================================

class SyllableGenerator:
    """Generiert wohlklingende Silben nach phonetischen Regeln"""

    def __init__(self, target_hardness: float = 0.5):
        """
        target_hardness: 0.0 = sehr weich, 1.0 = sehr hart
        """
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
            prefer_open = random.random() > 0.5

        # Manchmal Diphthong verwenden für weicheren Klang
        if random.random() < 0.15:
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
            pattern = random.choices(
                ['CV', 'CVC', 'VC', 'CCV'],
                weights=[0.4, 0.3, 0.15, 0.15]
            )[0]

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
        brightness = 0.5
        vowels_in_text = [c for c in text if c in VOWELS or c in 'aeiou']
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
        if random.random() < 0.3:
            coda = random.choice([c for c in CONSONANTS.keys()
                                 if CONSONANTS[c].quality != SoundQuality.HARD])
            text = onset + vowel + coda
            pattern = "CCVC"
        else:
            text = onset + vowel
            pattern = "CCV"

        return Syllable(text=text, pattern=pattern, hardness=0.4)


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
        # Einige bekannte Camping/RV/Tech-Marken
        return {
            'dometic', 'truma', 'fiamma', 'thule', 'webasto', 'victron',
            'renogy', 'ecoflow', 'bluetti', 'jackery', 'goal zero',
            'tesla', 'volta', 'ampere', 'watt', 'ohm',
        }

    def _check_pronounceability_de(self, name: str) -> float:
        """Prüft Aussprechbarkeit für Deutsche"""
        score = 1.0
        name_lower = name.lower()

        # Problematische Cluster für Deutsche
        for prob in PROBLEMATIC_DE:
            if prob in name_lower:
                score -= 0.15

        # Sehr lange Konsonantencluster
        if re.search(r'[bcdfghjklmnpqrstvwxz]{4,}', name_lower):
            score -= 0.3

        # Doppelvokale die im Deutschen unüblich sind
        unusual_vowels = ['oo', 'ee', 'aa']
        for uv in unusual_vowels:
            if uv in name_lower:
                score -= 0.05

        return max(0, score)

    def _check_pronounceability_en(self, name: str) -> float:
        """Prüft Aussprechbarkeit für Englischsprachige"""
        score = 1.0
        name_lower = name.lower()

        # Problematische Cluster für Englischsprachige
        for prob in PROBLEMATIC_EN:
            if prob in name_lower:
                score -= 0.2

        # Deutsche Umlaute (sollten nicht vorkommen, aber sicher ist sicher)
        if any(c in name_lower for c in 'äöüß'):
            score -= 0.4

        return max(0, score)

    def _check_memorability(self, name: str) -> float:
        """Bewertet Einprägsamkeit"""
        score = 1.0

        # Optimale Länge: 4-7 Zeichen (STRENGER)
        length = len(name)
        if length < 3:
            score -= 0.4
        elif length < 4:
            score -= 0.1
        elif length <= 6:
            score += 0.15  # Sweet spot: 4-6 Zeichen
        elif length == 7:
            score += 0.05
        elif length == 8:
            score -= 0.1
        elif length == 9:
            score -= 0.25
        elif length >= 10:
            score -= 0.4 + 0.1 * (length - 10)  # Starke Strafe für Wortmonster

        # Silbenzahl (2 optimal, 3 okay)
        syllable_count = self._count_syllables(name)
        if syllable_count == 2:
            score += 0.15
        elif syllable_count == 3:
            score -= 0.05  # Leichte Strafe statt Bonus
        elif syllable_count >= 4:
            score -= 0.3  # Starke Strafe für 4+ Silben

        # Wiederholende Elemente erhöhen Einprägsamkeit
        if self._has_repetition(name):
            score += 0.1

        return min(1.0, max(0, score))

    def _count_syllables(self, name: str) -> int:
        """Zählt ungefähre Silbenzahl"""
        name_lower = name.lower()
        # Vereinfachte Zählung: Vokale zählen, aufeinanderfolgende als eine
        vowels = 'aeiouäöü'
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

        # Alliteration
        if len(name_lower) >= 4:
            first_consonant = None
            for c in name_lower:
                if c not in 'aeiou':
                    if first_consonant is None:
                        first_consonant = c
                    elif c == first_consonant:
                        return True
                    break

        # Silbenwiederholung
        if len(name_lower) >= 4:
            half = len(name_lower) // 2
            if name_lower[:half] == name_lower[half:half*2]:
                return True

        return False

    def _check_euphony(self, name: str) -> float:
        """Bewertet Wohlklang"""
        score = 0.7  # Baseline
        name_lower = name.lower()

        # Gute Konsonant-Vokal-Abwechslung
        cv_alternation = self._calculate_cv_alternation(name_lower)
        score += cv_alternation * 0.2

        # Weiche Konsonanten für angenehmeren Klang
        soft_consonants = sum(1 for c in name_lower
                             if c in CONSONANTS and
                             CONSONANTS[c].quality in (SoundQuality.SOFT, SoundQuality.FLOWING))
        total_consonants = sum(1 for c in name_lower if c in CONSONANTS)
        if total_consonants > 0:
            soft_ratio = soft_consonants / total_consonants
            score += soft_ratio * 0.1

        # Offene Vokale klingen voller
        open_vowels = sum(1 for c in name_lower if c in 'ao')
        total_vowels = sum(1 for c in name_lower if c in 'aeiou')
        if total_vowels > 0:
            open_ratio = open_vowels / total_vowels
            score += open_ratio * 0.05

        return min(1.0, score)

    def _calculate_cv_alternation(self, name: str) -> float:
        """Berechnet wie gut Konsonanten und Vokale abwechseln"""
        if len(name) < 2:
            return 0.5

        vowels = 'aeiouäöü'
        alternations = 0

        for i in range(1, len(name)):
            prev_is_vowel = name[i-1] in vowels
            curr_is_vowel = name[i] in vowels
            if prev_is_vowel != curr_is_vowel:
                alternations += 1

        return alternations / (len(name) - 1)

    def _check_rhythm(self, name: str) -> float:
        """Bewertet rhythmische Qualität"""
        syllables = self._count_syllables(name)
        length = len(name)

        # 2 Silben = optimal
        if syllables == 2:
            return 0.95
        elif syllables == 1:
            return 0.75
        elif syllables == 3:
            # 3 Silben nur gut wenn Name kurz genug
            if length <= 7:
                return 0.8
            else:
                return 0.6
        elif syllables == 4:
            return 0.4  # Viel zu lang
        else:
            return 0.2  # 5+ Silben = inakzeptabel

    def _check_semantic_fit(self, name: str, morphemes_used: list) -> tuple[float, list]:
        """Bewertet semantische Passung zum Camping/Energie-Kontext"""
        score = 0.5
        associations = []

        for morpheme in morphemes_used:
            if isinstance(morpheme, SemanticMorpheme):
                if morpheme.category in ['energy', 'transformation', 'travel', 'nature']:
                    score += 0.15
                    associations.append(f"{morpheme.text}: {morpheme.meaning}")
                elif morpheme.category == 'tech':
                    score += 0.1
                    associations.append(f"{morpheme.text}: {morpheme.meaning}")

        return min(1.0, score), associations

    def _check_uniqueness(self, name: str) -> float:
        """Prüft ob der Name einzigartig ist"""
        name_lower = name.lower()

        # Exakte Übereinstimmung mit bekannter Marke
        if name_lower in self.existing_brands:
            return 0.0

        # Ähnlichkeit zu bekannten Marken (einfacher Check)
        for brand in self.existing_brands:
            if name_lower in brand or brand in name_lower:
                return 0.3
            # Levenshtein wäre besser, aber für Einfachheit:
            if len(set(name_lower) & set(brand)) / max(len(name_lower), len(brand)) > 0.8:
                return 0.5

        return 1.0

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

        pronounce_de = self._check_pronounceability_de(name)
        pronounce_en = self._check_pronounceability_en(name)
        memorability = self._check_memorability(name)
        euphony = self._check_euphony(name)
        rhythm = self._check_rhythm(name)
        semantic_fit, associations = self._check_semantic_fit(name, morphemes_used)
        uniqueness = self._check_uniqueness(name)

        # Gewichtete Gesamtbewertung (Memorability & Rhythm wichtiger!)
        total = (
            pronounce_de * 0.12 +
            pronounce_en * 0.12 +
            memorability * 0.30 +  # WICHTIG: Kurze Namen bevorzugen
            euphony * 0.15 +
            rhythm * 0.18 +        # WICHTIG: 2 Silben bevorzugen
            semantic_fit * 0.05 +  # Weniger wichtig
            uniqueness * 0.08
        )

        # Malus für Blacklist-Treffer
        if issues:
            total *= 0.1

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
                 target_hardness: float = 0.5,
                 prefer_semantic: bool = True):
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
            categories = ['energy', 'travel', 'nature', 'tech']

        morphemes_used = []

        # Strategie wählen (prefix_suffix bevorzugt = kürzere Namen)
        strategy = random.choices(
            ['prefix_suffix', 'double_morpheme', 'morpheme_fantasy', 'standalone_modified'],
            weights=[0.45, 0.10, 0.25, 0.20]  # double_morpheme selten (erzeugt lange Namen)
        )[0]

        if strategy == 'prefix_suffix':
            # Präfix + Suffix (z.B. "Voltix", "Trekora")
            prefix_pool = [m for m in self._get_prefix_morphemes()
                         if m.category in categories or m.category == 'fantasy']
            suffix_pool = self._get_suffix_morphemes()

            prefix = random.choice(prefix_pool)
            suffix = random.choice([s for s in suffix_pool if s.category == 'fantasy'])

            name = prefix.text.capitalize() + suffix.text
            morphemes_used = [prefix, suffix]

        elif strategy == 'double_morpheme':
            # Zwei bedeutungstragende Morpheme (z.B. "Solflux", "Volterra")
            # NUR kurze Morpheme verwenden (max 4 Zeichen pro Teil)
            first_pool = [m for m in self._get_prefix_morphemes()
                         if m.category in categories and len(m.text) <= 4]
            second_pool = [m for m in self._get_suffix_morphemes()
                          if m.category in categories and len(m.text) <= 3]

            if not first_pool or not second_pool:
                # Fallback zu prefix_suffix
                return self.generate_semantic_name(categories)

            first = random.choice(first_pool)
            second = random.choice([s for s in second_pool if s != first])

            # Verbindung glätten
            name = self._smooth_join(first.text, second.text)

            # Wenn Name zu lang, nochmal versuchen
            if len(name) > 8:
                return self.generate_semantic_name(categories)

            morphemes_used = [first, second]

        elif strategy == 'morpheme_fantasy':
            # Morphem + Fantasy-Silbe
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
                modifications = [
                    lambda x: x + 'a',
                    lambda x: x + 'o',
                    lambda x: x + 'i',
                    lambda x: x[0].upper() + x[1:] + 'ex',
                    lambda x: x[0].upper() + x[1:] + 'on',
                    lambda x: 'e' + x,
                ]
                mod = random.choice(modifications)
                name = mod(base.text).capitalize()
                morphemes_used = [base]
            else:
                # Fallback
                return self.generate_phonetic_name()

        return name, morphemes_used

    def _smooth_join(self, first: str, second: str) -> str:
        """Verbindet zwei Morpheme fließend"""
        # Wenn erstes mit Vokal endet und zweites mit Vokal beginnt
        vowels = 'aeiou'
        if first[-1] in vowels and second[0] in vowels:
            # Einen Vokal weglassen
            return first[:-1].capitalize() + second

        # Wenn erstes mit Konsonant endet und zweites mit Konsonant beginnt
        if first[-1] not in vowels and second[0] not in vowels:
            # Bindevokal einfügen
            connector = random.choice(['a', 'i', 'o'])
            return first.capitalize() + connector + second

        return first.capitalize() + second

    def generate_phonetic_name(self, syllable_count: int = None) -> tuple[str, list]:
        """Generiert rein phonetisch basierten Namen"""
        if syllable_count is None:
            syllable_count = random.choices([2, 3], weights=[0.6, 0.4])[0]

        syllables = []

        # Erste Silbe: oft mit Cluster für markanten Anfang
        if random.random() < 0.4:
            syllables.append(self.syllable_gen.generate_cluster_syllable())
        else:
            syllables.append(self.syllable_gen.generate_syllable('CV'))

        # Mittlere Silben
        for _ in range(syllable_count - 2):
            syllables.append(self.syllable_gen.generate_syllable())

        # Letzte Silbe: oft offen (endet auf Vokal) für weichen Ausklang
        if syllable_count > 1:
            last_syl = self.syllable_gen.generate_syllable(
                random.choice(['CV', 'CV', 'CVC'])
            )
            syllables.append(last_syl)

        name = ''.join(s.text for s in syllables).capitalize()
        return name, []

    def generate_hybrid_name(self) -> tuple[str, list]:
        """Kombiniert semantische und phonetische Elemente"""
        if random.random() < 0.5:
            # Phonetisch + Suffix
            phonetic, _ = self.generate_phonetic_name(syllable_count=2)
            suffix = random.choice([m for m in SEMANTIC_MORPHEMES
                                   if m.category == 'fantasy' and m.works_as_suffix])
            name = phonetic[:-1] + suffix.text  # Letzten Buchstaben ersetzen
            return name.capitalize(), [suffix]
        else:
            # Präfix + Phonetisch
            prefix = random.choice([m for m in SEMANTIC_MORPHEMES
                                   if m.works_as_prefix and m.syllables == 1])
            phonetic_syl = self.syllable_gen.generate_syllable('CV')
            name = prefix.text + phonetic_syl.text
            return name.capitalize(), [prefix]

    def generate_name(self) -> tuple[str, list]:
        """Generiert einen Namen mit gemischter Strategie"""
        strategy = random.choices(
            ['semantic', 'phonetic', 'hybrid'],
            weights=[0.5, 0.25, 0.25] if self.prefer_semantic else [0.25, 0.5, 0.25]
        )[0]

        if strategy == 'semantic':
            return self.generate_semantic_name()
        elif strategy == 'phonetic':
            return self.generate_phonetic_name()
        else:
            return self.generate_hybrid_name()

    def generate_batch(self,
                       count: int = 50,
                       min_score: float = 0.6,
                       categories: list = None) -> list[NameScore]:
        """Generiert eine Batch von Namen und filtert nach Score"""
        results = []
        attempts = 0
        max_attempts = count * 10  # Prevent infinite loops

        while len(results) < count and attempts < max_attempts:
            attempts += 1

            if categories and random.random() < 0.7:
                name, morphemes = self.generate_semantic_name(categories)
            else:
                name, morphemes = self.generate_name()

            # Duplikate vermeiden
            if any(r.name.lower() == name.lower() for r in results):
                continue

            # Harter Längenfilter: Max 9 Zeichen
            if len(name) > 9:
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
        default=30,
        help='Number of names to generate (default: 30)'
    )

    parser.add_argument(
        '-m', '--min-score',
        type=float,
        default=0.65,
        help='Minimum score threshold (0-1, default: 0.65)'
    )

    parser.add_argument(
        '-c', '--categories',
        nargs='+',
        choices=['energy', 'travel', 'nature', 'tech', 'transformation'],
        default=['energy', 'travel', 'nature'],
        help='Semantic categories to focus on'
    )

    parser.add_argument(
        '--hardness',
        type=float,
        default=0.4,
        help='Sound hardness (0=soft, 1=hard, default: 0.4)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path'
    )

    parser.add_argument(
        '-f', '--format',
        choices=['text', 'json', 'csv'],
        default='text',
        help='Output format (default: text)'
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
