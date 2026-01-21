#!/usr/bin/env python3
"""
Phoneme Configuration Loader
============================
Loads phoneme configurations from YAML files for brand name generators.

Usage:
    from brandkit.generators.phonemes import (
        load_greek, load_turkic, load_nordic,
        load_japanese, load_latin, load_celtic, load_celestial,
        load_strategies, load_phonaesthemes, load_hazards, load_industries
    )

    greek = load_greek()
    hazards = load_hazards()
    industries = load_industries()
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from functools import lru_cache


# =============================================================================
# Configuration Path
# =============================================================================

PHONEMES_DIR = Path(__file__).parent


# =============================================================================
# Data Classes for Typed Access
# =============================================================================

@dataclass
class PhonemeConfig:
    """Container for loaded phoneme configuration."""
    roots: Dict[str, List[Tuple[str, str]]]
    suffixes: Dict[str, List[str]]
    prefixes: Dict[str, List[str]]
    patterns: Dict[str, List[str]]
    phonetics: Dict[str, Any]
    avoid: Dict[str, List[str]]
    raw: Dict[str, Any]  # Full raw config

    def get_all_roots(self) -> List[Tuple[str, str, str]]:
        """Get all roots as (phoneme, meaning, category) tuples."""
        all_roots = []
        for category, roots in self.roots.items():
            for root_data in roots:
                if len(root_data) >= 2:
                    phoneme, meaning = root_data[0], root_data[1]
                    all_roots.append((phoneme, meaning, category))
        return all_roots

    def get_roots_by_category(self, *categories: str) -> List[Tuple[str, str, str]]:
        """Get roots filtered by category names."""
        all_roots = []
        for category in categories:
            if category in self.roots:
                for root_data in self.roots[category]:
                    if len(root_data) >= 2:
                        phoneme, meaning = root_data[0], root_data[1]
                        all_roots.append((phoneme, meaning, category))
        return all_roots

    def get_suffix_pool(self, *suffix_types: str) -> List[str]:
        """Get combined suffix list from specified types."""
        pool = []
        for stype in suffix_types:
            if stype in self.suffixes:
                pool.extend(self.suffixes[stype])
        return pool


@dataclass
class StrategiesConfig:
    """Container for generation strategies configuration."""
    sound_symbolism: Dict[str, Any]
    phonotactics: Dict[str, Any]
    vowel_harmony: Dict[str, Any]
    syllables: Dict[str, Any]
    strategies: Dict[str, Any]
    junctions: Dict[str, Any]
    scoring: Dict[str, Any]
    euphony: Dict[str, Any]
    archetypes: Dict[str, Any]
    pronounceability: Dict[str, Any]
    raw: Dict[str, Any]

    def get_archetype(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific brand archetype configuration."""
        return self.archetypes.get(name)

    def get_scoring_params(self) -> Dict[str, Any]:
        """Get scoring parameters."""
        return self.scoring

    def get_bad_clusters(self) -> List[str]:
        """Get all bad clusters to avoid."""
        clusters = []
        bad = self.scoring.get('bad_clusters', {})
        for category in bad.values():
            if isinstance(category, list):
                clusters.extend(category)
        return clusters

    def get_impossible_endings(self) -> set:
        """Get all impossible word-final clusters."""
        endings = set()
        impossible = self.pronounceability.get('impossible_endings', {})
        for category in impossible.values():
            if isinstance(category, list):
                endings.update(category)
        return endings

    def get_valid_final_consonants(self) -> set:
        """Get valid word-final single consonants."""
        return set(self.pronounceability.get('valid_final_consonants', []))

    def get_valid_final_vowels(self) -> set:
        """Get valid word-final vowels."""
        return set(self.pronounceability.get('valid_final_vowels', ['a', 'e', 'i', 'o', 'u']))

    def get_valid_final_chars(self) -> set:
        """Get all valid word-final single characters (vowels + consonants)."""
        return self.get_valid_final_vowels() | self.get_valid_final_consonants()

    def get_valid_final_clusters(self) -> set:
        """Get all valid word-final clusters."""
        clusters = set()
        valid = self.pronounceability.get('valid_final_clusters', {})
        for category in valid.values():
            if isinstance(category, list):
                clusters.update(category)
        return clusters

    def get_impossible_internal(self) -> set:
        """Get impossible internal clusters."""
        return set(self.pronounceability.get('impossible_internal', []))

    def get_triple_reject(self) -> set:
        """Get triple patterns to reject."""
        return set(self.pronounceability.get('triple_reject', []))

    def get_awkward_initial_clusters(self) -> set:
        """Get awkward initial clusters to reject."""
        return set(self.pronounceability.get('awkward_initial_clusters', []))

    def get_chemical_endings(self) -> set:
        """Get chemical/pharmaceutical endings to reject."""
        return set(self.pronounceability.get('chemical_endings', []))

    def get_rhythm_config(self) -> Dict[str, Any]:
        """Get rhythm/stress pattern configuration."""
        return self.raw.get('rhythm', {})

    def get_phonaesthetic_config(self) -> Dict[str, Any]:
        """Get phonaesthetic quality configuration."""
        return self.raw.get('phonaesthetic_quality', {})

    # --- Locale-specific methods ---

    def get_locale_pronounceability(self, locale: str) -> Dict[str, Any]:
        """Get locale-specific pronounceability rules (en, de)."""
        key = f'pronounceability_{locale}'
        return self.raw.get(key, {})

    def get_locale_allowed_initial(self, locale: str) -> set:
        """Get allowed initial clusters for a specific locale."""
        config = self.get_locale_pronounceability(locale)
        return set(config.get('allowed_initial_clusters', []))

    def get_locale_forbidden_initial(self, locale: str) -> set:
        """Get forbidden initial clusters for a specific locale."""
        config = self.get_locale_pronounceability(locale)
        forbidden = config.get('forbidden_clusters', {})
        return set(forbidden.get('initial', []))

    def get_locale_forbidden_final(self, locale: str) -> set:
        """Get forbidden final clusters for a specific locale."""
        config = self.get_locale_pronounceability(locale)
        forbidden = config.get('forbidden_clusters', {})
        return set(forbidden.get('final', []))

    def get_locale_forbidden_internal(self, locale: str) -> set:
        """Get forbidden internal clusters for a specific locale."""
        config = self.get_locale_pronounceability(locale)
        forbidden = config.get('forbidden_clusters', {})
        return set(forbidden.get('internal', []))

    def get_german_normalization(self) -> Dict[str, Any]:
        """Get German normalization rules (umlauts, eszett)."""
        return self.raw.get('german_normalization', {})

    def get_market_weights(self, market: str) -> Dict[str, Any]:
        """Get market-specific phonaesthetic weights (en, de, en_de)."""
        weights = self.raw.get('market_weights', {})
        return weights.get(market, {})


@dataclass
class PhonaesthemesConfig:
    """Container for phonaesthemes (sound symbolism) configuration."""
    onsets: Dict[str, Dict[str, Any]]
    codas: Dict[str, Dict[str, Any]]
    consonants: Dict[str, Any]
    vowels: Dict[str, Any]
    archetypes: Dict[str, Dict[str, Any]]
    semantic_mappings: Dict[str, Any]
    raw: Dict[str, Any]

    def get_sounds_for_archetype(self, archetype: str) -> Dict[str, List[str]]:
        """Get preferred sounds for a brand archetype."""
        arch = self.archetypes.get(archetype, {})
        return {
            'onsets': arch.get('preferred_onsets', []),
            'codas': arch.get('preferred_codas', []),
            'consonants': arch.get('consonant_qualities', []),
            'vowels': arch.get('vowel_qualities', []),
        }

    def get_onset_meaning(self, onset: str) -> Optional[str]:
        """Get the meaning associated with an onset cluster."""
        return self.onsets.get(onset, {}).get('meaning')

    def get_semantic_dimension(self, dimension: str) -> Dict[str, List[str]]:
        """Get sounds associated with a semantic dimension."""
        return self.semantic_mappings.get(dimension, {})


@dataclass
class HazardsConfig:
    """Container for cross-linguistic hazards configuration."""
    exact_words: Dict[str, List[Dict[str, Any]]]
    sound_alikes: Dict[str, List[Dict[str, Any]]]
    phonetic_patterns: Dict[str, List[Dict[str, Any]]]
    acronyms: Dict[str, Any]
    cultural_religious: Dict[str, List[str]]
    raw: Dict[str, Any]

    def get_hazards_for_market(self, market: str) -> List[Dict[str, Any]]:
        """Get hazards specific to a market."""
        hazards = []
        # Exact words
        for word, entries in self.exact_words.items():
            for entry in entries:
                if entry.get('market') == market or entry.get('market') == 'universal':
                    hazards.append({'word': word, **entry})
        return hazards

    def get_all_exact_hazards(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all exact word hazards."""
        return self.exact_words

    def get_sound_alikes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get sound-alike hazards."""
        return self.sound_alikes

    def get_phonetic_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get phonetic pattern hazards (regex-based)."""
        return self.phonetic_patterns


@dataclass
class IndustriesConfig:
    """Container for industry-specific naming conventions."""
    industries: Dict[str, Dict[str, Any]]
    raw: Dict[str, Any]

    def get_profile(self, industry: str) -> Optional[Dict[str, Any]]:
        """Get naming profile for a specific industry."""
        return self.industries.get(industry)

    def get_all_industries(self) -> List[str]:
        """Get list of all available industries."""
        return list(self.industries.keys())

    def get_preferred_sounds(self, industry: str) -> Dict[str, List[str]]:
        """Get preferred sounds for an industry."""
        profile = self.industries.get(industry, {})
        return {
            'consonants': profile.get('preferred_sounds', []),
            'vowels': profile.get('vowels', []),
            'suffixes': profile.get('suffixes', []),
        }

    def get_cultural_sources(self, industry: str) -> List[str]:
        """Get recommended cultural sources for an industry."""
        profile = self.industries.get(industry, {})
        return profile.get('cultural_sources', [])


# =============================================================================
# Loader Functions
# =============================================================================

def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML file from the phonemes directory."""
    filepath = PHONEMES_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Phoneme config not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_greek() -> PhonemeConfig:
    """Load Greek mythology phoneme configuration."""
    raw = _load_yaml('greek.yaml')

    # Convert roots from YAML format to tuples
    roots = {}
    for category, root_list in raw.get('roots', {}).items():
        roots[category] = [tuple(r) for r in root_list]

    return PhonemeConfig(
        roots=roots,
        suffixes=raw.get('suffixes', {}),
        prefixes=raw.get('prefixes', {}),
        patterns=raw.get('patterns', {}),
        phonetics=raw.get('phonetics', {}),
        avoid=raw.get('avoid', {}),
        raw=raw,
    )


@lru_cache(maxsize=1)
def load_turkic() -> PhonemeConfig:
    """Load Turkic phoneme configuration."""
    raw = _load_yaml('turkic.yaml')

    # Convert roots from YAML format to tuples
    roots = {}
    for category, root_list in raw.get('roots', {}).items():
        roots[category] = [tuple(r) for r in root_list]

    return PhonemeConfig(
        roots=roots,
        suffixes=raw.get('suffixes', {}),
        prefixes=raw.get('prefixes', {}),
        patterns=raw.get('patterns', {}),
        phonetics=raw.get('phonetics', {}),
        avoid=raw.get('avoid', {}),
        raw=raw,
    )


@lru_cache(maxsize=1)
def load_nordic() -> PhonemeConfig:
    """Load Nordic/Scandinavian phoneme configuration."""
    raw = _load_yaml('nordic.yaml')

    # Convert roots from YAML format to tuples
    roots = {}
    for category, root_list in raw.get('roots', {}).items():
        roots[category] = [tuple(r) for r in root_list]

    return PhonemeConfig(
        roots=roots,
        suffixes=raw.get('suffixes', {}),
        prefixes=raw.get('prefixes', {}),
        patterns=raw.get('patterns', {}),
        phonetics=raw.get('phonetics', {}),
        avoid=raw.get('avoid', {}),
        raw=raw,
    )


@lru_cache(maxsize=1)
def load_japanese() -> PhonemeConfig:
    """Load Japanese-inspired phoneme configuration."""
    raw = _load_yaml('japanese.yaml')

    roots = {}
    for category, root_list in raw.get('roots', {}).items():
        roots[category] = [tuple(r) for r in root_list]

    return PhonemeConfig(
        roots=roots,
        suffixes=raw.get('suffixes', {}),
        prefixes=raw.get('prefixes', {}),
        patterns=raw.get('patterns', {}),
        phonetics=raw.get('phonetics', {}),
        avoid=raw.get('avoid', {}),
        raw=raw,
    )


@lru_cache(maxsize=1)
def load_latin() -> PhonemeConfig:
    """Load Latin/Romance phoneme configuration."""
    raw = _load_yaml('latin.yaml')

    roots = {}
    for category, root_list in raw.get('roots', {}).items():
        roots[category] = [tuple(r) for r in root_list]

    return PhonemeConfig(
        roots=roots,
        suffixes=raw.get('suffixes', {}),
        prefixes=raw.get('prefixes', {}),
        patterns=raw.get('patterns', {}),
        phonetics=raw.get('phonetics', {}),
        avoid=raw.get('avoid', {}),
        raw=raw,
    )


@lru_cache(maxsize=1)
def load_celtic() -> PhonemeConfig:
    """Load Celtic phoneme configuration."""
    raw = _load_yaml('celtic.yaml')

    roots = {}
    for category, root_list in raw.get('roots', {}).items():
        roots[category] = [tuple(r) for r in root_list]

    return PhonemeConfig(
        roots=roots,
        suffixes=raw.get('suffixes', {}),
        prefixes=raw.get('prefixes', {}),
        patterns=raw.get('patterns', {}),
        phonetics=raw.get('phonetics', {}),
        avoid=raw.get('avoid', {}),
        raw=raw,
    )


@lru_cache(maxsize=1)
def load_celestial() -> PhonemeConfig:
    """Load Celestial/Space phoneme configuration."""
    raw = _load_yaml('celestial.yaml')

    roots = {}
    for category, root_list in raw.get('roots', {}).items():
        roots[category] = [tuple(r) for r in root_list]

    return PhonemeConfig(
        roots=roots,
        suffixes=raw.get('suffixes', {}),
        prefixes=raw.get('prefixes', {}),
        patterns=raw.get('patterns', {}),
        phonetics=raw.get('phonetics', {}),
        avoid=raw.get('avoid', {}),
        raw=raw,
    )


@lru_cache(maxsize=1)
def load_phonaesthemes() -> PhonaesthemesConfig:
    """Load phonaesthemes (sound symbolism) configuration."""
    raw = _load_yaml('phonaesthemes.yaml')

    return PhonaesthemesConfig(
        onsets=raw.get('onsets', {}),
        codas=raw.get('codas', {}),
        consonants=raw.get('consonants', {}),
        vowels=raw.get('vowels', {}),
        archetypes=raw.get('archetypes', {}),
        semantic_mappings=raw.get('semantic_mappings', {}),
        raw=raw,
    )


@lru_cache(maxsize=1)
def load_hazards() -> HazardsConfig:
    """Load cross-linguistic hazards configuration."""
    raw = _load_yaml('hazards.yaml')

    return HazardsConfig(
        exact_words=raw.get('exact_words', {}),
        sound_alikes=raw.get('sound_alikes', {}),
        phonetic_patterns=raw.get('phonetic_patterns', {}),
        acronyms=raw.get('acronyms', {}),
        cultural_religious=raw.get('cultural_religious', {}),
        raw=raw,
    )


@lru_cache(maxsize=1)
def load_industries() -> IndustriesConfig:
    """Load industry-specific naming conventions."""
    raw = _load_yaml('industries.yaml')

    return IndustriesConfig(
        industries=raw.get('industries', {}),
        raw=raw,
    )


@lru_cache(maxsize=1)
def load_strategies() -> StrategiesConfig:
    """Load generation strategies configuration."""
    raw = _load_yaml('strategies.yaml')

    return StrategiesConfig(
        sound_symbolism=raw.get('sound_symbolism', {}),
        phonotactics=raw.get('phonotactics', {}),
        vowel_harmony=raw.get('vowel_harmony', {}),
        syllables=raw.get('syllables', {}),
        strategies=raw.get('strategies', {}),
        junctions=raw.get('junctions', {}),
        scoring=raw.get('scoring', {}),
        euphony=raw.get('euphony', {}),
        archetypes=raw.get('archetypes', {}),
        pronounceability=raw.get('pronounceability', {}),
        raw=raw,
    )


def reload_configs():
    """Clear cached configs and reload from disk."""
    load_greek.cache_clear()
    load_turkic.cache_clear()
    load_nordic.cache_clear()
    load_japanese.cache_clear()
    load_latin.cache_clear()
    load_celtic.cache_clear()
    load_celestial.cache_clear()
    load_strategies.cache_clear()
    load_phonaesthemes.cache_clear()
    load_hazards.cache_clear()
    load_industries.cache_clear()


def get_all_cultures() -> Dict[str, PhonemeConfig]:
    """Load all cultural phoneme configurations."""
    return {
        'greek': load_greek(),
        'turkic': load_turkic(),
        'nordic': load_nordic(),
        'japanese': load_japanese(),
        'latin': load_latin(),
        'celtic': load_celtic(),
        'celestial': load_celestial(),
    }


def get_culture(name: str) -> Optional[PhonemeConfig]:
    """Load a specific cultural phoneme configuration by name."""
    loaders = {
        'greek': load_greek,
        'turkic': load_turkic,
        'nordic': load_nordic,
        'japanese': load_japanese,
        'latin': load_latin,
        'celtic': load_celtic,
        'celestial': load_celestial,
    }
    loader = loaders.get(name.lower())
    return loader() if loader else None


# =============================================================================
# German Orthography Normalization
# =============================================================================

def normalize_de(text: str, apply_final_devoicing: bool = True) -> str:
    """
    Normalize German orthography for phonetic comparison.

    Applies:
    - Umlaut expansion: ä→ae, ö→oe, ü→ue
    - Eszett expansion: ß→ss
    - Optional final devoicing: b→p, d→t, g→k, v→f, z→s at word end

    Parameters
    ----------
    text : str
        Text to normalize
    apply_final_devoicing : bool
        Whether to apply final devoicing (default True)

    Returns
    -------
    str
        Normalized text
    """
    # Umlaut mappings
    result = text.replace('ä', 'ae').replace('Ä', 'Ae')
    result = result.replace('ö', 'oe').replace('Ö', 'Oe')
    result = result.replace('ü', 'ue').replace('Ü', 'Ue')
    # Eszett
    result = result.replace('ß', 'ss')

    # Final devoicing (for similarity matching)
    if apply_final_devoicing and result:
        final_devoicing = {'b': 'p', 'd': 't', 'g': 'k', 'v': 'f', 'z': 's'}
        if result[-1].lower() in final_devoicing:
            if result[-1].isupper():
                result = result[:-1] + final_devoicing[result[-1].lower()].upper()
            else:
                result = result[:-1] + final_devoicing[result[-1]]

    return result


# =============================================================================
# Lightweight G2P (Grapheme-to-Phoneme Approximation)
# =============================================================================

def approx_g2p_en(text: str) -> str:
    """
    Approximate grapheme-to-phoneme conversion for English.

    Uses ~40 orthographic rules to approximate pronunciation.
    Returns simplified phonetic representation (not IPA).

    Rules handle:
    - Digraphs: th, sh, ch, ph, wh, ck, ng, gh
    - Silent letters: kn-, gn-, wr-, -mb, -mn
    - Vowel digraphs: ee, ea, oo, ou, oi, ow, aw
    - Magic e: VCe → long vowel

    Parameters
    ----------
    text : str
        Text to convert

    Returns
    -------
    str
        Simplified phonetic representation
    """
    text = text.lower()

    # === CONSONANT DIGRAPHS ===
    # Order matters - longer patterns first
    replacements = [
        # Digraphs
        ('tch', 'CH'),      # match, watch
        ('sch', 'SH'),      # school (in loanwords)
        ('th', 'TH'),       # the, think (simplify to TH)
        ('sh', 'SH'),       # ship, fish
        ('ch', 'CH'),       # chip, church
        ('ph', 'F'),        # phone, graph
        ('wh', 'W'),        # what, when
        ('ck', 'K'),        # back, clock
        ('ng', 'NG'),       # ring, song
        ('dg', 'J'),        # edge, judge
        ('gh', ''),         # silent: night, though (complex, simplify)
        ('wr', 'R'),        # write, wrong
        ('kn', 'N'),        # knife, know
        ('gn', 'N'),        # gnome, gnat
        ('mb', 'M'),        # lamb, comb (final mb)
        ('mn', 'M'),        # autumn, column (final mn)

        # Vowel digraphs
        ('ee', 'EE'),       # see, tree (long e)
        ('ea', 'EE'),       # eat, sea (usually long e)
        ('oo', 'OO'),       # food, moon
        ('ou', 'OW'),       # out, house
        ('ow', 'OW'),       # cow, how (before consonant or end)
        ('oi', 'OY'),       # oil, coin
        ('oy', 'OY'),       # boy, toy
        ('aw', 'AW'),       # saw, law
        ('au', 'AW'),       # caught, fault
        ('ie', 'EE'),       # field, piece (usually)
        ('ai', 'AY'),       # rain, wait
        ('ay', 'AY'),       # day, play
    ]

    result = text
    for pattern, replacement in replacements:
        result = result.replace(pattern, replacement)

    # === FINAL SILENT E (Magic E) ===
    # VCe pattern - vowel becomes long
    import re
    # Simple approximation: remove final e after consonant
    if len(result) > 2 and result[-1] == 'e' and result[-2] not in 'aeiouAEIOUYW':
        result = result[:-1]

    return result


def approx_g2p_de(text: str) -> str:
    """
    Approximate grapheme-to-phoneme conversion for German.

    Uses ~50 orthographic rules to approximate pronunciation.
    Returns simplified phonetic representation (not IPA).

    Rules handle:
    - German digraphs: sch, ch, pf, tz, st, sp
    - Umlauts: ä, ö, ü (if not already normalized)
    - Vowel length markers: eh, ah, ie, oo
    - Final devoicing: -b→p, -d→t, -g→k

    Parameters
    ----------
    text : str
        Text to convert

    Returns
    -------
    str
        Simplified phonetic representation
    """
    # First normalize umlauts
    text = normalize_de(text, apply_final_devoicing=False)
    text = text.lower()

    # === CONSONANT DIGRAPHS ===
    replacements = [
        # German-specific digraphs
        ('tsch', 'CH'),     # deutsch, Tschüss
        ('sch', 'SH'),      # schön, Schule
        ('ch', 'CH'),       # ich, auch (simplify - actually /ç/ vs /x/)
        ('pf', 'PF'),       # Pferd, Kopf
        ('tz', 'TS'),       # Katze, Platz
        ('ck', 'K'),        # Lack, dick

        # Initial clusters (German pronounces these)
        # st- and sp- at start are /ʃt/ and /ʃp/ in German
        # We'll mark them specially

        # Vowel length markers (long vowels)
        ('eh', 'EH'),       # sehr, mehr (long e)
        ('ah', 'AH'),       # Jahr, Bahn (long a)
        ('oh', 'OH'),       # Sohn, Bohne (long o)
        ('ie', 'EE'),       # Bier, Spiel (long i)
        ('ee', 'EE'),       # See, Fee (long e)
        ('aa', 'AA'),       # Saal (long a)
        ('oo', 'OO'),       # Boot (long o)

        # Common endings
        ('ung', 'UNG'),     # Hoffnung
        ('heit', 'HAIT'),   # Freiheit
        ('keit', 'KAIT'),   # Möglichkeit
        ('lich', 'LIKH'),   # freundlich
        ('ig', 'IKH'),      # fertig (final -ig = /ɪç/)
    ]

    result = text
    for pattern, replacement in replacements:
        result = result.replace(pattern, replacement)

    # === INITIAL ST/SP ===
    # German pronounces initial st- as /ʃt/ and sp- as /ʃp/
    if result.startswith('st'):
        result = 'SHT' + result[2:]
    elif result.startswith('sp'):
        result = 'SHP' + result[2:]

    # === FINAL DEVOICING ===
    final_devoicing = {'b': 'P', 'd': 'T', 'g': 'K', 'v': 'F', 'z': 'S'}
    if result and result[-1] in final_devoicing:
        result = result[:-1] + final_devoicing[result[-1]]

    return result


# =============================================================================
# Utility Functions
# =============================================================================

def get_connector(root_end: str, suffix_start: str, strategies: StrategiesConfig = None) -> str:
    """
    Get appropriate connector between root and suffix.

    Uses junction rules from strategies config if available.
    """
    if strategies is None:
        strategies = load_strategies()

    junctions = strategies.junctions

    vowels = 'aeiou'
    is_root_vowel = root_end in vowels
    is_suffix_vowel = suffix_start in vowels

    # Vowel-vowel junction
    if is_root_vowel and is_suffix_vowel:
        vv = junctions.get('vowel_vowel', {})
        inserts = vv.get('strategies', [{}])[0].get('insert', ['n', 'r', 'x'])
        import random
        return random.choice(inserts)

    # Consonant-consonant junction
    if not is_root_vowel and not is_suffix_vowel:
        cc = junctions.get('consonant_consonant', {})
        inserts = cc.get('strategies', [{}])[0].get('insert', ['a', 'i', 'o'])
        import random
        return random.choice(inserts)

    # Mixed - no connector needed
    return ''


def is_pronounceable(name: str, strategies: StrategiesConfig = None,
                     markets: str = 'en_de') -> tuple[bool, str]:
    """
    Check if a name is pronounceable using linguistic rules from config.

    Returns (is_pronounceable, reason).

    Uses the Sonority Sequencing Principle and phonotactic constraints
    for English/German cross-linguistic pronounceability.
    Rules are loaded from strategies.yaml pronounceability section.

    Parameters
    ----------
    name : str
        The name to check
    strategies : StrategiesConfig, optional
        Strategies config. Loads default if None.
    markets : str
        Target market(s): 'en', 'de', or 'en_de' (default).
        For 'en_de', name must be pronounceable in BOTH languages.
    """
    if strategies is None:
        strategies = load_strategies()

    name_lower = name.lower()

    if len(name_lower) < 2:
        return False, "too_short"

    # Determine which locales to check
    if markets == 'en':
        locales = ['en']
    elif markets == 'de':
        locales = ['de']
    else:  # en_de or default
        locales = ['en', 'de']

    # Load shared rules from config (language-agnostic)
    impossible_endings = strategies.get_impossible_endings()
    valid_final_chars = strategies.get_valid_final_chars()
    valid_final_clusters = strategies.get_valid_final_clusters()
    impossible_internal = strategies.get_impossible_internal()
    triple_reject = strategies.get_triple_reject()
    awkward_initial = strategies.get_awkward_initial_clusters()
    chemical_endings = strategies.get_chemical_endings()

    # Check for awkward initial clusters (shared rules)
    for cluster in awkward_initial:
        if name_lower.startswith(cluster):
            return False, f"awkward_start:{cluster}"

    # Check for chemical/pharmaceutical endings
    for ending in chemical_endings:
        if name_lower.endswith(ending):
            return False, f"chemical_ending:{ending}"

    # === LOCALE-SPECIFIC CHECKS ===
    # For each target locale, check forbidden clusters
    for locale in locales:
        # Forbidden initial clusters for this locale
        forbidden_initial = strategies.get_locale_forbidden_initial(locale)
        for cluster in forbidden_initial:
            if name_lower.startswith(cluster):
                return False, f"forbidden_start_{locale}:{cluster}"

        # Forbidden final clusters for this locale
        forbidden_final = strategies.get_locale_forbidden_final(locale)
        for cluster in forbidden_final:
            if name_lower.endswith(cluster):
                return False, f"forbidden_end_{locale}:{cluster}"

        # Forbidden internal clusters for this locale
        forbidden_internal = strategies.get_locale_forbidden_internal(locale)
        for cluster in forbidden_internal:
            if cluster in name_lower:
                return False, f"forbidden_internal_{locale}:{cluster}"

    # Check ending (shared rules)
    last1 = name_lower[-1]
    last2 = name_lower[-2:] if len(name_lower) >= 2 else ''
    last3 = name_lower[-3:] if len(name_lower) >= 3 else ''

    # Check for impossible endings first
    if last2 in impossible_endings:
        return False, f"unpronounceable_ending:{last2}"
    if last3 and last3[-2:] in impossible_endings:
        return False, f"unpronounceable_ending:{last3[-2:]}"

    # Check if ending is valid
    ending_ok = (
        last1 in valid_final_chars or
        last2 in valid_final_clusters or
        last3 in valid_final_clusters
    )

    if not ending_ok:
        # Final character isn't in valid list
        return False, f"unusual_ending:{last1}"

    # Check for impossible internal clusters (shared)
    for cluster in impossible_internal:
        if cluster in name_lower:
            return False, f"impossible_cluster:{cluster}"

    # Check for triple patterns to reject
    for pattern in triple_reject:
        if pattern in name_lower:
            return False, f"triple_pattern:{pattern}"

    # === REPETITIVE PATTERNS ===
    # Detect stuttering like "fafa", "baba" - but only consonant-initiated
    # Vowel-consonant patterns like "oror" in "Sorora" sound fine
    vowels = set('aeiou')
    for i in range(len(name_lower) - 3):
        chunk = name_lower[i:i+2]
        if chunk == name_lower[i+2:i+4]:
            # Only reject if pattern starts with consonant (CV-CV stuttering)
            if chunk[0] not in vowels:
                return False, f"stuttering_pattern:{chunk}{chunk}"

    # Check for 3-char repetition (these are always awkward)
    for i in range(len(name_lower) - 5):
        chunk = name_lower[i:i+3]
        if chunk == name_lower[i+3:i+6]:
            return False, f"stuttering_pattern:{chunk}{chunk}"

    return True, "ok"


def score_name(name: str, strategies: StrategiesConfig = None,
               category: str = None, markets: str = 'en_de') -> float:
    """
    Score a name using linguistic pronounceability + phonaesthetic criteria.

    Returns float between 0.0 and 1.0.

    Uses a three-phase approach:
    1. Hard gate: Must be pronounceable (if not, score = 0)
    2. Basic scoring: Length, CV balance, patterns
    3. Phonaesthetic scoring: Research-based beauty assessment
    4. Market-specific adjustments: Apply EN/DE-specific weights

    Parameters
    ----------
    name : str
        The name to score
    strategies : StrategiesConfig, optional
        Strategies config. Loads default if None.
    category : str, optional
        Category for sound-fit scoring ('tech', 'luxury', 'power', 'nature', 'speed')
    markets : str
        Target market(s): 'en', 'de', or 'en_de' (default).

    Returns
    -------
    float
        Score between 0.0 and 1.0
    """
    if strategies is None:
        strategies = load_strategies()

    name_lower = name.lower()

    # === PHASE 1: PRONOUNCEABILITY GATE ===
    is_ok, reason = is_pronounceable(name, strategies, markets)
    if not is_ok:
        return 0.0  # Hard fail - unpronounceable names get 0

    # === PHASE 2: BASIC SCORING ===
    scoring = strategies.scoring
    basic_score = 0.6  # Base score for pronounceable names

    # Length scoring
    length_params = scoring.get('length', {})
    name_len = len(name)
    ideal_min = length_params.get('ideal_min', 5)
    ideal_max = length_params.get('ideal_max', 7)
    abs_min = length_params.get('absolute_min', 4)
    abs_max = length_params.get('absolute_max', 9)

    if name_len < abs_min:
        basic_score -= 0.2
    elif name_len > abs_max:
        basic_score -= 0.15
    elif ideal_min <= name_len <= ideal_max:
        basic_score += 0.1

    # Penalize double vowels (still awkward even if pronounceable)
    for v in 'aeiou':
        if v + v in name_lower:
            basic_score -= 0.15

    # Reward strong openings
    if name_lower[0] in 'kptbdgvrstlm':
        basic_score += 0.05

    # Reward clean endings
    if name_lower[-1] in 'aeiox':
        basic_score += 0.05
    elif name_lower[-2:] in {'us', 'is', 'on', 'an', 'ar', 'or', 'er', 'ix', 'ex'}:
        basic_score += 0.05

    basic_score = min(max(basic_score, 0.0), 1.0)

    # === PHASE 3: PHONAESTHETIC SCORING ===
    phon_result = phonaesthetic_score(name, strategies, category)
    phon_score = phon_result['score']

    # Combine: 40% basic, 60% phonaesthetic
    final_score = basic_score * 0.4 + phon_score * 0.6

    # === PHASE 4: MARKET-SPECIFIC ADJUSTMENTS ===
    market_weights = strategies.get_market_weights(markets)
    if market_weights:
        consonant_adj = market_weights.get('consonant_adjustments', {})
        # Apply adjustments based on consonants present in name
        for consonant, adjustment in consonant_adj.items():
            if consonant in name_lower:
                final_score += adjustment

        # Check ending preferences
        ending_adj = market_weights.get('ending_adjustments', {})
        for ending, adjustment in ending_adj.items():
            if name_lower.endswith(ending):
                final_score += adjustment
                break  # Only apply one ending adjustment

    return min(max(final_score, 0.0), 1.0)


def syllabify(name: str) -> List[str]:
    """
    Split a name into syllables using onset maximization principle.

    Uses a simplified but linguistically-grounded approach:
    1. Find vowel nuclei
    2. Assign consonants to onsets (maximizing onsets)
    3. Remaining consonants go to codas

    Returns list of syllable strings.
    """
    name_lower = name.lower()
    vowels = set('aeiou')
    consonants = set('bcdfghjklmnpqrstvwxyz')

    # Find vowel positions (syllable nuclei)
    vowel_positions = [i for i, c in enumerate(name_lower) if c in vowels]

    if not vowel_positions:
        return [name_lower]  # No vowels, return as single syllable

    syllables = []
    prev_end = 0

    for idx, vpos in enumerate(vowel_positions):
        # Find where this syllable starts
        if idx == 0:
            start = 0
        else:
            # Split consonants between syllables (onset maximization)
            prev_vowel = vowel_positions[idx - 1]
            consonant_run = name_lower[prev_vowel + 1:vpos]

            if len(consonant_run) == 0:
                start = prev_end
            elif len(consonant_run) == 1:
                start = vpos  # Single C goes to onset
            else:
                # Split: keep 1-2 consonants for onset, rest to coda
                # Common onsets: bl, br, cl, cr, dr, fl, fr, gl, gr, pl, pr, sc, sk, sl, sm, sn, sp, st, sw, tr
                valid_onsets = {'bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr',
                                'sc', 'sk', 'sl', 'sm', 'sn', 'sp', 'st', 'sw', 'tr', 'tw', 'wr',
                                'str', 'spr', 'scr'}
                # Try to find valid onset
                for onset_len in [3, 2, 1]:
                    if onset_len <= len(consonant_run):
                        potential_onset = consonant_run[-onset_len:]
                        if potential_onset in valid_onsets or onset_len == 1:
                            start = vpos - onset_len + (prev_vowel + 1 - prev_end)
                            break
                else:
                    start = prev_end

            start = max(prev_end, vpos - 1)  # Simplified: at least 1 consonant to onset

        # Find where this syllable ends
        if idx == len(vowel_positions) - 1:
            end = len(name_lower)
        else:
            next_vowel = vowel_positions[idx + 1]
            # Syllable ends before next syllable starts
            end = vpos + 1  # Will be adjusted by next iteration

        # For now, use a simpler approach: split at consonant clusters
        if idx < len(vowel_positions) - 1:
            next_v = vowel_positions[idx + 1]
            between = name_lower[vpos + 1:next_v]
            if len(between) <= 1:
                end = next_v
            else:
                end = vpos + 1 + (len(between) // 2)

        syllables.append(name_lower[prev_end:end])
        prev_end = end

    # Add final segment
    if prev_end < len(name_lower):
        syllables[-1] += name_lower[prev_end:]

    # Clean up empty syllables
    syllables = [s for s in syllables if s]

    return syllables if syllables else [name_lower]


def get_syllable_weight(syllable: str) -> str:
    """
    Determine if a syllable is Heavy (H) or Light (L).

    Heavy: closed syllable (ends in consonant) or has long vowel/diphthong
    Light: open syllable with short vowel (CV pattern)
    """
    vowels = set('aeiou')
    consonants = set('bcdfghjklmnpqrstvwxyz')

    if not syllable:
        return 'L'

    # Check if syllable ends in consonant (closed = heavy)
    if syllable[-1] in consonants:
        return 'H'

    # Check for diphthongs (heavy)
    diphthongs = ['ai', 'au', 'ei', 'eu', 'oi', 'ou', 'ae', 'oe']
    for diph in diphthongs:
        if diph in syllable:
            return 'H'

    # Open syllable with single short vowel = light
    return 'L'


def get_stress_pattern(syllables: List[str]) -> str:
    """
    Infer likely stress pattern from syllable weights.

    Uses weight-to-stress principle:
    - Heavy syllables attract stress (S)
    - Light syllables typically unstressed (U)
    - First syllable bias for Germanic languages

    Returns string like "SU", "SUU", "US", etc.
    """
    if not syllables:
        return ""

    weights = [get_syllable_weight(s) for s in syllables]
    pattern = []

    for i, w in enumerate(weights):
        if i == 0:
            # First syllable: stressed if heavy, or if name is 2 syllables
            if w == 'H' or len(syllables) <= 2:
                pattern.append('S')
            else:
                pattern.append('U')
        elif w == 'H':
            # Heavy syllables attract stress
            # But avoid stress clash (two S in a row)
            if pattern and pattern[-1] == 'S':
                pattern.append('U')  # Avoid clash
            else:
                pattern.append('S')
        else:
            pattern.append('U')

    # Ensure at least one stress
    if 'S' not in pattern and pattern:
        pattern[0] = 'S'

    return ''.join(pattern)


def analyze_rhythm(name: str, strategies: StrategiesConfig = None) -> Dict[str, Any]:
    """
    Analyze the rhythmic properties of a name.

    Returns dict with:
    - syllables: list of syllable strings
    - syllable_count: number of syllables
    - weights: list of H/L for each syllable
    - stress_pattern: string like "SU", "SUU"
    - rhythm_type: "trochaic", "iambic", "dactylic", etc.
    - rhythm_score: 0.0 to 1.0
    """
    if strategies is None:
        strategies = load_strategies()

    rhythm_config = strategies.get_rhythm_config()
    name_lower = name.lower()

    # Syllabify
    syllables = syllabify(name_lower)
    weights = [get_syllable_weight(s) for s in syllables]
    stress_pattern = get_stress_pattern(syllables)

    # Determine rhythm type
    rhythm_type = "unknown"
    if stress_pattern.startswith('S'):
        if stress_pattern == 'SU' or stress_pattern.startswith('SU'):
            rhythm_type = "trochaic"
        elif stress_pattern == 'SUU':
            rhythm_type = "dactylic"
    elif stress_pattern.startswith('U'):
        if stress_pattern == 'US' or stress_pattern.startswith('US'):
            rhythm_type = "iambic"
    if stress_pattern in ['SS', 'SSS']:
        rhythm_type = "spondaic"
    if all(c == 'U' for c in stress_pattern):
        rhythm_type = "unstressed"

    # Calculate rhythm score from config
    stress_patterns = rhythm_config.get('stress_patterns', {})
    rhythm_score = 0.5  # Default

    # Look up score in config
    for category, patterns in stress_patterns.items():
        if isinstance(patterns, dict) and stress_pattern in patterns:
            rhythm_score = patterns[stress_pattern]
            break

    # Apply bonuses/penalties from config
    bonuses = rhythm_config.get('bonuses', {})
    penalties = rhythm_config.get('penalties', {})

    # Trochaic 2-syllable bonus
    if rhythm_type == 'trochaic' and len(syllables) == 2:
        rhythm_score += bonuses.get('trochaic_2syl', 0)

    # Heavy first syllable bonus
    if weights and weights[0] == 'H':
        rhythm_score += bonuses.get('heavy_first_syl', 0)

    # Too many syllables penalty
    ideal = rhythm_config.get('ideal_syllable_count', {})
    if len(syllables) > ideal.get('maximum', 3):
        rhythm_score += penalties.get('too_many_syllables', 0)

    # Iambic start penalty
    if rhythm_type == 'iambic':
        rhythm_score += penalties.get('iambic_start', 0)

    # Clamp score
    rhythm_score = min(max(rhythm_score, 0.0), 1.0)

    return {
        'syllables': syllables,
        'syllable_count': len(syllables),
        'weights': weights,
        'weight_pattern': ''.join(weights),
        'stress_pattern': stress_pattern,
        'rhythm_type': rhythm_type,
        'rhythm_score': rhythm_score,
    }


def _get_cv_pattern(name: str) -> str:
    """Get consonant-vowel pattern of a name (e.g., 'Voltix' -> 'CVCCVC')."""
    vowels = set('aeiou')
    pattern = []
    for char in name.lower():
        if char in vowels:
            pattern.append('V')
        elif char.isalpha():
            pattern.append('C')
    return ''.join(pattern)


def phonaesthetic_score(name: str, strategies: StrategiesConfig = None,
                        category: str = None) -> Dict[str, Any]:
    """
    Score a name based on phonaesthetic research.

    Evaluates how pleasant/beautiful a name sounds based on:
    - Consonant quality (Crystal's research on beautiful sounds)
    - Vowel quality (front vs back vowels)
    - Processing fluency (CV balance, mild repetition)
    - Phonotactic naturalness
    - Rhythm (from analyze_rhythm)

    Parameters
    ----------
    name : str
        The name to score
    strategies : StrategiesConfig, optional
        Strategies config. Loads default if None.
    category : str, optional
        Category for sound-fit scoring ('tech', 'luxury', 'power', 'nature', 'speed')

    Returns
    -------
    dict
        Detailed breakdown with:
        - score: float (0.0 to 1.0)
        - consonant_score: float
        - vowel_score: float
        - fluency_score: float
        - naturalness_score: float
        - rhythm_score: float
        - details: dict with explanations
    """
    import re

    if strategies is None:
        strategies = load_strategies()

    config = strategies.get_phonaesthetic_config()
    name_lower = name.lower()

    details = {}
    scores = {}

    # === CONSONANT QUALITY ===
    consonant_config = config.get('consonant_scores', {})

    # Pleasant consonants (l, m, n, s, r)
    pleasant_cons = set(consonant_config.get('pleasant', {}).get('consonants', ['l', 'm', 'n', 's', 'r']))
    pleasant_bonus = consonant_config.get('pleasant', {}).get('score_bonus', 0.02)

    # Powerful consonants (plosives)
    powerful_cons = set(consonant_config.get('powerful', {}).get('consonants', ['k', 't', 'p', 'b', 'd', 'g']))
    powerful_bonus = consonant_config.get('powerful', {}).get('score_bonus', 0.01)

    # Awkward consonants
    awkward_cons = set(consonant_config.get('awkward', {}).get('consonants', ['x', 'z', 'q']))
    awkward_penalty = consonant_config.get('awkward', {}).get('score_penalty', -0.03)

    # Avoid patterns
    avoid_patterns = consonant_config.get('avoid', {}).get('patterns', ['zh', 'dj'])
    avoid_penalty = consonant_config.get('avoid', {}).get('score_penalty', -0.05)

    # Calculate consonant score
    cons_score = 0.5  # Base
    pleasant_count = sum(1 for c in name_lower if c in pleasant_cons)
    powerful_count = sum(1 for c in name_lower if c in powerful_cons)
    awkward_count = sum(1 for c in name_lower if c in awkward_cons)

    cons_score += pleasant_count * pleasant_bonus
    cons_score += powerful_count * powerful_bonus
    cons_score += awkward_count * awkward_penalty

    # Check avoid patterns
    for pattern in avoid_patterns:
        if pattern in name_lower:
            cons_score += avoid_penalty

    cons_score = min(max(cons_score, 0.0), 1.0)
    scores['consonant'] = cons_score
    details['consonant'] = {
        'pleasant_count': pleasant_count,
        'powerful_count': powerful_count,
        'awkward_count': awkward_count,
    }

    # === VOWEL QUALITY ===
    vowel_config = config.get('vowel_scores', {})

    front_vowels = set(vowel_config.get('front', {}).get('vowels', ['e', 'i']))
    front_bonus = vowel_config.get('front', {}).get('score_bonus', 0.02)

    back_vowels = set(vowel_config.get('back', {}).get('vowels', ['o', 'u']))
    back_bonus = vowel_config.get('back', {}).get('score_bonus', 0.0)

    central_vowels = set(vowel_config.get('central', {}).get('vowels', ['a']))
    central_bonus = vowel_config.get('central', {}).get('score_bonus', 0.01)

    vowel_score = 0.5  # Base
    front_count = sum(1 for c in name_lower if c in front_vowels)
    back_count = sum(1 for c in name_lower if c in back_vowels)
    central_count = sum(1 for c in name_lower if c in central_vowels)

    vowel_score += front_count * front_bonus
    vowel_score += back_count * back_bonus
    vowel_score += central_count * central_bonus

    vowel_score = min(max(vowel_score, 0.0), 1.0)
    scores['vowel'] = vowel_score
    details['vowel'] = {
        'front_count': front_count,
        'back_count': back_count,
        'central_count': central_count,
    }

    # === PROCESSING FLUENCY ===
    fluency_config = config.get('fluency', {})
    fluency_score = 0.5  # Base

    # CV ratio
    cv_config = fluency_config.get('cv_ratio', {})
    vowels_all = set('aeiou')
    n_vowels = sum(1 for c in name_lower if c in vowels_all)
    n_consonants = sum(1 for c in name_lower if c.isalpha() and c not in vowels_all)

    if n_vowels > 0:
        cv_ratio = n_consonants / n_vowels
        ideal_min = cv_config.get('ideal_min', 0.8)
        ideal_max = cv_config.get('ideal_max', 1.5)
        ratio_bonus = cv_config.get('bonus', 0.05)
        ratio_penalty = cv_config.get('penalty', -0.05)

        if ideal_min <= cv_ratio <= ideal_max:
            fluency_score += ratio_bonus
            details['cv_ratio'] = f"{cv_ratio:.2f} (ideal range)"
        else:
            fluency_score += ratio_penalty
            details['cv_ratio'] = f"{cv_ratio:.2f} (outside ideal)"
    else:
        details['cv_ratio'] = "no vowels"

    # Mild repetition bonus
    repetition_config = fluency_config.get('mild_repetition', {})
    for pattern_info in repetition_config.get('patterns', []):
        pattern = pattern_info.get('pattern', '')
        bonus = pattern_info.get('bonus', 0)
        if pattern and re.search(pattern, name_lower):
            fluency_score += bonus

    # CV alternation
    cv_alt_config = fluency_config.get('cv_alternation', {})
    cv_pattern = _get_cv_pattern(name)
    cv_alt_patterns = cv_alt_config.get('patterns', ['CVCV', 'CVCVC', 'VCVCV'])
    cv_alt_bonus = cv_alt_config.get('bonus', 0.03)

    if cv_pattern in cv_alt_patterns:
        fluency_score += cv_alt_bonus
        details['cv_pattern'] = f"{cv_pattern} (ideal alternation)"
    else:
        details['cv_pattern'] = cv_pattern

    fluency_score = min(max(fluency_score, 0.0), 1.0)
    scores['fluency'] = fluency_score

    # === PHONOTACTIC NATURALNESS ===
    naturalness_config = config.get('naturalness', {})
    naturalness_score = 0.5  # Base

    for pattern_info in naturalness_config.get('natural_patterns', []):
        pattern = pattern_info.get('pattern', '')
        bonus = pattern_info.get('bonus', 0)
        if pattern and re.search(pattern, name_lower):
            naturalness_score += bonus

    naturalness_score = min(max(naturalness_score, 0.0), 1.0)
    scores['naturalness'] = naturalness_score

    # === RHYTHM ===
    rhythm_info = analyze_rhythm(name, strategies)
    scores['rhythm'] = rhythm_info['rhythm_score']
    details['rhythm'] = rhythm_info

    # === CLUSTER QUALITY ===
    # Penalize harsh consonant clusters even if pronounceable
    cluster_config = config.get('cluster_quality', {})
    cluster_score = 0.6  # Base score

    # Check for harsh clusters
    harsh_clusters = cluster_config.get('harsh_clusters', ['sph', 'gry', 'bry', 'scr', 'spr'])
    penalty_per_cluster = cluster_config.get('penalty_per_cluster', -0.05)

    harsh_found = []
    for cluster in harsh_clusters:
        if cluster in name_lower:
            cluster_score += penalty_per_cluster
            harsh_found.append(cluster)

    # Check for triple consonants (e.g., "lll", "nnn")
    triple_penalty = cluster_config.get('triple_consonant_penalty', -0.08)
    consonants_str = 'bcdfghjklmnpqrstvwxyz'
    triple_cons_found = []
    for c in consonants_str:
        if c * 3 in name_lower:
            cluster_score += triple_penalty
            triple_cons_found.append(c * 3)

    cluster_score = min(max(cluster_score, 0.0), 1.0)
    scores['cluster_quality'] = cluster_score
    details['cluster_quality'] = {
        'harsh_clusters_found': harsh_found,
        'triple_consonants_found': triple_cons_found,
    }

    # === ENDING QUALITY ===
    # Good endings enhance memorability, bad ones detract
    ending_config = config.get('ending_quality', {})
    ending_score = 0.5  # Base score

    good_endings = ending_config.get('good_endings', {})
    good_bonus = ending_config.get('good_bonus', 0.04)
    bad_endings = ending_config.get('bad_endings', [])
    bad_penalty = ending_config.get('bad_penalty', -0.06)

    # Check good endings (tier1, tier2, tier3)
    tier1 = good_endings.get('tier1', ['a', 'o', 'ia', 'io'])
    tier2 = good_endings.get('tier2', ['is', 'us', 'on', 'an', 'or', 'er'])
    tier3 = good_endings.get('tier3', ['ix', 'ex', 'ax'])

    # Ensure all endings are strings and sort by length (longer first for specificity)
    all_good_endings = [str(e) for e in tier1 + tier2 + tier3 if e]
    all_good_endings.sort(key=len, reverse=True)

    ending_match = None
    # Check in order of specificity (longer endings first)
    for ending in all_good_endings:
        if name_lower.endswith(ending):
            ending_score += good_bonus
            ending_match = ending
            break

    # Check bad endings
    bad_match = None
    bad_endings_str = [str(e) for e in bad_endings if e]
    for ending in bad_endings_str:
        if name_lower.endswith(ending):
            ending_score += bad_penalty
            bad_match = ending
            break

    ending_score = min(max(ending_score, 0.0), 1.0)
    scores['ending_quality'] = ending_score
    details['ending_quality'] = {
        'good_ending_match': ending_match,
        'bad_ending_match': bad_match,
    }

    # === CATEGORY SOUND FIT (optional) ===
    if category:
        category_config = config.get('category_sound_fit', {}).get(category, {})
        preferred = set(category_config.get('preferred', []))
        if preferred:
            fit_count = sum(1 for c in name_lower if c in preferred)
            fit_ratio = fit_count / len(name_lower) if name_lower else 0
            scores['category_fit'] = min(fit_ratio * 2, 1.0)  # Scale up
            details['category_fit'] = f"{category}: {fit_count} matching sounds"

    # === MEMORABILITY ===
    # Based on cognitive psychology research:
    # - Optimal length (4-7 characters) aids recall
    # - Strong initial consonants create distinctiveness
    # - 2-3 syllables is easiest to remember
    # - Repetition/alliteration aids encoding
    memorability_score = 0.5  # Base

    # Length factor: optimal 4-7 characters
    name_len = len(name_lower)
    if 4 <= name_len <= 7:
        memorability_score += 0.15  # Optimal length
    elif name_len <= 3:
        memorability_score += 0.05  # Very short - easy but maybe too simple
    elif name_len <= 9:
        memorability_score += 0.08  # Acceptable
    else:
        memorability_score -= 0.05  # Too long

    # Syllable count: 2-3 is optimal
    syllable_count = rhythm_info.get('syllable_count', 2)
    if 2 <= syllable_count <= 3:
        memorability_score += 0.12
    elif syllable_count == 1:
        memorability_score += 0.05  # Very short
    elif syllable_count == 4:
        memorability_score += 0.03  # Acceptable
    else:
        memorability_score -= 0.05  # Too many syllables

    # Strong initial consonant (plosives, fricatives)
    strong_initials = set('bcdgkptvf')
    if name_lower and name_lower[0] in strong_initials:
        memorability_score += 0.08

    # Alliteration bonus (repeated initial sounds within word)
    syllables = rhythm_info.get('syllables', [])
    has_alliteration = False
    if len(syllables) >= 2:
        first_sounds = [s[0] for s in syllables if s]
        if len(first_sounds) >= 2 and first_sounds[0] == first_sounds[1]:
            memorability_score += 0.06  # Alliteration
            has_alliteration = True

    # Assonance bonus (repeated vowels)
    vowel_counts = {}
    for v in 'aeiou':
        cnt = name_lower.count(v)
        if cnt >= 2:
            vowel_counts[v] = cnt
    if vowel_counts:
        memorability_score += 0.05  # Has vowel repetition

    memorability_score = min(max(memorability_score, 0.0), 1.0)
    scores['memorability'] = memorability_score
    details['memorability'] = {
        'length': name_len,
        'syllable_count': syllable_count,
        'strong_initial': name_lower[0] if name_lower else '',
        'has_alliteration': has_alliteration,
    }

    # === WEIGHTED FINAL SCORE ===
    weights = config.get('weights', {})
    w_cons = weights.get('consonant_quality', 0.22)
    w_vowel = weights.get('vowel_quality', 0.15)
    w_fluency = weights.get('fluency', 0.20)
    w_natural = weights.get('naturalness', 0.13)
    w_rhythm = weights.get('rhythm', 0.10)
    w_cluster = weights.get('cluster_quality', 0.10)
    w_ending = weights.get('ending_quality', 0.10)

    final_score = (
        scores['consonant'] * w_cons +
        scores['vowel'] * w_vowel +
        scores['fluency'] * w_fluency +
        scores['naturalness'] * w_natural +
        scores['rhythm'] * w_rhythm +
        scores['cluster_quality'] * w_cluster +
        scores['ending_quality'] * w_ending
    )

    # If category fit was calculated, blend it in
    if 'category_fit' in scores:
        # Give category fit 10% weight, reduce others proportionally
        final_score = final_score * 0.9 + scores['category_fit'] * 0.1

    # Determine quality tier
    thresholds = config.get('thresholds', {})
    if final_score >= thresholds.get('excellent', 0.85):
        quality = 'excellent'
    elif final_score >= thresholds.get('good', 0.70):
        quality = 'good'
    elif final_score >= thresholds.get('acceptable', 0.50):
        quality = 'acceptable'
    else:
        quality = 'poor'

    return {
        'score': final_score,
        'quality': quality,
        'consonant_score': scores['consonant'],
        'vowel_score': scores['vowel'],
        'fluency_score': scores['fluency'],
        'naturalness_score': scores['naturalness'],
        'rhythm_score': scores['rhythm'],
        'cluster_quality_score': scores['cluster_quality'],
        'ending_quality_score': scores['ending_quality'],
        'memorability_score': scores['memorability'],
        'category_fit_score': scores.get('category_fit'),
        'details': details,
    }


def apply_vowel_harmony(word: str, use_back: bool = True) -> str:
    """
    Apply Turkic-style vowel harmony to a word.

    If use_back is True, uses back vowels (a, o, u).
    If False, uses front vowels (e, i).
    """
    strategies = load_strategies()
    harmony = strategies.vowel_harmony

    back = set(harmony.get('back_vowels', ['a', 'o', 'u']))
    front = set(harmony.get('front_vowels', ['e', 'i']))

    result = []
    for char in word.lower():
        if char in back and not use_back:
            # Map back to front
            mapping = {'a': 'e', 'o': 'i', 'u': 'i'}
            result.append(mapping.get(char, char))
        elif char in front and use_back:
            # Map front to back
            mapping = {'e': 'a', 'i': 'u'}
            result.append(mapping.get(char, char))
        else:
            result.append(char)

    return ''.join(result)


def generate_from_pattern(pattern: str, phonetics: Dict[str, Any] = None,
                          use_back_vowels: bool = None) -> str:
    """
    Generate a name from a phonetic pattern.

    Pattern language:
        C = any consonant
        V = any vowel
        B = back vowel (a, o, u)
        F = front vowel (e, i)
        K = strong/plosive consonant (k, t, p, b, d, g)
        S = soft consonant (s, l, m, n)
        R = 'r' literal
        N = nasal (m, n)
        L = liquid (l, r)
        P = plosive (k, p, t, b, d, g)
        H = 'h' literal
        X = 'x' literal
        Y = 'y' literal
        Other lowercase = literal character

    Parameters
    ----------
    pattern : str
        Pattern string (e.g., "CVCVC", "KVCan")
    phonetics : dict, optional
        Phonetics configuration. If None, loads from strategies.
    use_back_vowels : bool, optional
        If True, V uses back vowels. If False, V uses front vowels.
        If None, chooses randomly.

    Returns
    -------
    str
        Generated name string
    """
    import random

    if phonetics is None:
        strategies = load_strategies()
        phonetics = {
            'consonants': {
                'all': list('klmnprst'),
                'strong': list('kptbdg'),
                'soft': list('slmn'),
            },
            'vowels': {
                'all': list('aeiou'),
                'back': list('aou'),
                'front': list('ei'),
            },
            'nasals': list('mn'),
            'liquids': list('lr'),
            'plosives': list('kptbdg'),
            'fricatives': list('sfvz'),
        }

    # Default phoneme sets
    consonants = phonetics.get('consonants', {}).get('all', list('klmnprst'))
    if isinstance(phonetics.get('consonants'), dict):
        consonants_strong = phonetics['consonants'].get('strong', list('kptbdg'))
        consonants_soft = phonetics['consonants'].get('soft', list('slmn'))
    else:
        consonants_strong = list('kptbdg')
        consonants_soft = list('slmn')

    vowels_all = phonetics.get('vowels', {}).get('all', list('aeiou'))
    vowels_back = phonetics.get('vowels', {}).get('back', list('aou'))
    vowels_front = phonetics.get('vowels', {}).get('front', list('ei'))

    nasals = phonetics.get('nasals', list('mn'))
    liquids = phonetics.get('liquids', list('lr'))
    plosives = phonetics.get('plosives', list('kptbdg'))
    fricatives = phonetics.get('fricatives', list('sfvz'))

    # Determine vowel harmony
    if use_back_vowels is None:
        use_back_vowels = random.choice([True, False])

    vowel_set = vowels_back if use_back_vowels else vowels_front

    result = []
    i = 0
    while i < len(pattern):
        char = pattern[i]

        if char == 'C':
            result.append(random.choice(consonants))
        elif char == 'V':
            result.append(random.choice(vowel_set))
        elif char == 'B':
            result.append(random.choice(vowels_back))
        elif char == 'F':
            result.append(random.choice(vowels_front))
        elif char == 'K':
            result.append(random.choice(consonants_strong))
        elif char == 'S':
            result.append(random.choice(consonants_soft))
        elif char == 'N':
            result.append(random.choice(nasals))
        elif char == 'L':
            result.append(random.choice(liquids))
        elif char == 'P':
            result.append(random.choice(plosives))
        elif char == 'Z':
            result.append(random.choice(fricatives))
        elif char in 'RHXY':
            # Literal uppercase
            result.append(char.lower())
        else:
            # Literal character (lowercase)
            result.append(char)

        i += 1

    return ''.join(result)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Cultural Phoneme Loaders
    'load_greek',
    'load_turkic',
    'load_nordic',
    'load_japanese',
    'load_latin',
    'load_celtic',
    'load_celestial',
    # Meta/Strategy Loaders
    'load_strategies',
    'load_phonaesthemes',
    'load_hazards',
    'load_industries',
    # Utility Loaders
    'reload_configs',
    'get_all_cultures',
    'get_culture',
    # Data classes
    'PhonemeConfig',
    'StrategiesConfig',
    'PhonaesthemesConfig',
    'HazardsConfig',
    'IndustriesConfig',
    # Utilities
    'get_connector',
    'is_pronounceable',
    'score_name',
    'phonaesthetic_score',
    'syllabify',
    'get_syllable_weight',
    'get_stress_pattern',
    'analyze_rhythm',
    'apply_vowel_harmony',
    'generate_from_pattern',
]
