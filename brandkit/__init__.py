#!/usr/bin/env python3
"""
BrandKit - Brand Name Generator & Validator
============================================

A comprehensive toolkit for generating and validating brand names
for German and English markets, with trademark and domain checking.

Quick Start
-----------
    from brandkit import BrandKit

    kit = BrandKit()

    # Generate names
    names = kit.generate(count=10)

    # Check availability (with Nice class filtering)
    result = kit.check("Voltix", nice_classes="camping_rv")

    # Or with specific classes
    result = kit.check("Voltix", nice_classes=[9, 12])

Modules
-------
    brandkit.generators - Name generation (rule-based, Markov, LLM)
    brandkit.checkers   - Validation (trademark, domain, similarity)
    brandkit.db         - SQLite database for name management
    brandkit.config     - Configuration and Nice class profiles

CLI Usage
---------
    python -m brandkit generate -n 10
    python -m brandkit check "Voltix" --profile camping_rv --full
    python -m brandkit profiles
"""

__version__ = "0.1.0"
__author__ = "Joern Ihlenburg"

import sys
from pathlib import Path

# Ensure parent directory is in path for imports
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

# =============================================================================
# Submodule Imports
# =============================================================================

from . import generators
from . import checkers
from . import db
from . import config

# =============================================================================
# Generator Imports
# =============================================================================

from .generators import (
    # Main generators
    BrandGenerator,
    BrandNameGenerator,
    RuleBasedGenerator,
    LLMGenerator,
    TurkicGenerator,
    GreekGenerator,
    NordicGenerator,
    # Cultural generators
    JapaneseGenerator,
    LatinGenerator,
    CelticGenerator,
    CelestialGenerator,
    AnimalsGenerator,
    MythologyGenerator,
    LandmarksGenerator,
    # Base generator framework
    CulturalGenerator,
    CulturalGeneratedName,
    HazardChecker,
    HazardResult,
    SyllableAnalyzer,
    SyllableInfo,
    PhonaestheticsEngine,
    IndustryManager,
    MemorabilityScorer,
    CultureBlender,
    CompetitiveDifferentiator,
    # Data classes
    GeneratedName,
    NameScore,
    TurkicName,
    GreekName,
    NordicName,
)

# =============================================================================
# Checker Imports
# =============================================================================

from .checkers import (
    # Unified checker
    TrademarkChecker,
    # Individual checkers
    EUIPOChecker,
    RapidAPIChecker,
    DomainChecker,
    SimilarityChecker,
    # Convenience functions
    check_domain,
    check_similarity,
)

# =============================================================================
# Database Imports
# =============================================================================

from .db import (
    NameDB,
    BrandNameDB,
    NameStatus,
    NameRecord,
    BrandName,
    get_db,
    get_namedb,
)

# =============================================================================
# Config Imports
# =============================================================================

from .config import (
    Config,
    get_config,
    load_env,
    # Nice class profiles
    NICE_PROFILES,
    get_nice_classes,
    list_profiles,
)

from .quality import filter_and_rank
from .settings import get_setting

# =============================================================================
# BrandKit Main Class
# =============================================================================

class BrandKit:
    """
    Main interface for brand name generation and validation.

    Combines all generators and checkers into a unified workflow with
    database persistence and Nice class filtering support.

    Attributes
    ----------
    db : NameDB
        Database for storing and managing names
    config : Config
        Application configuration with API keys

    Examples
    --------
    Basic usage:

        >>> kit = BrandKit()
        >>> names = kit.generate(count=5)
        >>> for name in names:
        ...     print(f"{name.name}: {name.total_score:.2f}")

    With trademark checking:

        >>> result = kit.check("Voltix", nice_classes="camping_rv")
        >>> if result['available']:
        ...     kit.save(name)

    Using specific Nice classes:

        >>> result = kit.check("Voltix", nice_classes=[9, 12])
    """

    def __init__(self, db_path: str = None):
        """
        Initialize BrandKit with all components.

        Parameters
        ----------
        db_path : str, optional
            Path to SQLite database. Defaults to 'brandnames.db' in
            the current directory.
        """
        # Load configuration from .env
        self._config = get_config()

        # Initialize database
        self._db = NameDB(db_path) if db_path else get_db()

        # Initialize generators (lazy-loaded where expensive)
        self._rule_gen = BrandGenerator()
        self._turkic_gen = TurkicGenerator()
        self._greek_gen = GreekGenerator()
        self._nordic_gen = NordicGenerator()
        self._llm_gen = None
        if self._config.has_anthropic:
            self._llm_gen = LLMGenerator(self._config.anthropic_api_key)

        # New cultural generators (lazy-loaded)
        self._japanese_gen = None
        self._latin_gen = None
        self._celtic_gen = None
        self._celestial_gen = None
        self._animals_gen = None
        self._mythology_gen = None
        self._landmarks_gen = None

        # Advanced features
        self._hazard_checker = HazardChecker()
        self._phonaesthetics = PhonaestheticsEngine()
        self._industry_manager = IndustryManager()
        self._memorability_scorer = MemorabilityScorer()

        # Initialize checkers
        self._trademark = TrademarkChecker(
            euipo_client_id=self._config.euipo_client_id,
            euipo_client_secret=self._config.euipo_client_secret,
            rapidapi_key=self._config.rapidapi_key,
        )
        self._domain = DomainChecker()
        self._similarity = SimilarityChecker()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def db(self) -> NameDB:
        """Access the name database."""
        return self._db

    @property
    def config(self) -> Config:
        """Access configuration."""
        return self._config

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def _get_japanese_gen(self):
        """Lazy-load the Japanese generator."""
        if self._japanese_gen is None:
            self._japanese_gen = JapaneseGenerator()
        return self._japanese_gen

    def _get_latin_gen(self):
        """Lazy-load the Latin generator."""
        if self._latin_gen is None:
            self._latin_gen = LatinGenerator()
        return self._latin_gen

    def _get_celtic_gen(self):
        """Lazy-load the Celtic generator."""
        if self._celtic_gen is None:
            self._celtic_gen = CelticGenerator()
        return self._celtic_gen

    def _get_celestial_gen(self):
        """Lazy-load the Celestial generator."""
        if self._celestial_gen is None:
            self._celestial_gen = CelestialGenerator()
        return self._celestial_gen

    def _get_animals_gen(self):
        """Lazy-load the Animals generator."""
        if self._animals_gen is None:
            self._animals_gen = AnimalsGenerator()
        return self._animals_gen

    def _get_mythology_gen(self):
        """Lazy-load the Mythology generator."""
        if self._mythology_gen is None:
            self._mythology_gen = MythologyGenerator()
        return self._mythology_gen

    def _get_landmarks_gen(self):
        """Lazy-load the Landmarks generator."""
        if self._landmarks_gen is None:
            self._landmarks_gen = LandmarksGenerator()
        return self._landmarks_gen

    def generate(self,
                 count: int = None,
                 method: str = None,
                 **kwargs) -> list:
        """
        Generate brand names using the specified method.

        Parameters
        ----------
        count : int
            Number of names to generate (default from app.yaml)
        method : str
            Generation method. One of:
            - "rule_based": Phonetic rules and morpheme combinations
            - "greek": Greek mythology-inspired names
            - "nordic": Nordic/Scandinavian patterns
            - "japanese": Japanese-inspired phonetics
            - "latin": Latin/Romance morphemes
            - "celtic": Celtic patterns
            - "celestial": Space/cosmic themes
            - "turkic": VW/Nissan style vowel harmony
            - "blend": Cross-culture blending
            - "llm": Claude-powered creative generation (optional)
        **kwargs
            Additional arguments passed to the generator

        Returns
        -------
        list
            List of generated names (NameScore or LLMGeneratedName)

        Raises
        ------
        ValueError
            If method is unknown or LLM requested without API key

        Examples
        --------
            >>> names = kit.generate(count=5, method="rule_based")
            >>> names = kit.generate(count=5, method="greek")
            >>> names = kit.generate(count=5, method="llm",
            ...                      context="Solar panels")
        """
        gen_defaults = get_setting("generation.defaults", {}) or {}
        if count is None:
            count = gen_defaults.get("count")
        if method is None:
            method = gen_defaults.get("method")
        if count is None or method is None:
            raise ValueError("generation.defaults must be set in app.yaml")

        quality_cfg = get_setting("quality_filter", {}) or {}
        quality_enabled = kwargs.pop('post_filter', None)
        if quality_enabled is None:
            quality_enabled = quality_cfg.get("enabled")
        quality_min_score = kwargs.pop('quality_min_score', None)
        if quality_min_score is None:
            quality_min_score = quality_cfg.get("min_score")
        quality_similarity = kwargs.pop('quality_similarity', None)
        if quality_similarity is None:
            quality_similarity = quality_cfg.get("similarity_threshold")
        quality_max_suffix_pct = kwargs.pop('quality_max_suffix_pct', None)
        if quality_max_suffix_pct is None:
            quality_max_suffix_pct = quality_cfg.get("max_suffix_pct")
        quality_max_prefix_pct = kwargs.pop('quality_max_prefix_pct', None)
        if quality_max_prefix_pct is None:
            quality_max_prefix_pct = quality_cfg.get("max_prefix_pct")
        quality_oversample = kwargs.pop('quality_oversample', None)
        if quality_oversample is None:
            quality_oversample = quality_cfg.get("oversample")

        if (quality_enabled is None or quality_min_score is None or
                quality_similarity is None or quality_max_suffix_pct is None or
                quality_max_prefix_pct is None or quality_oversample is None):
            raise ValueError("quality_filter settings must be set in app.yaml")

        quality_markets = kwargs.get('phonetic_markets')
        if quality_markets is None:
            quality_markets = gen_defaults.get("markets")
        if quality_markets is None:
            raise ValueError("generation.defaults.markets must be set in app.yaml")

        raw_count = max(count, int(count * quality_oversample)) if quality_enabled else count

        if method == "rule_based":
            names = self._rule_gen.generate_batch(raw_count, **kwargs)
            for n in names:
                if not hasattr(n, 'method') or not getattr(n, 'method', None):
                    try:
                        setattr(n, 'method', 'rule_based')
                    except Exception:
                        pass
            if quality_enabled:
                return filter_and_rank(
                    names,
                    target_count=count,
                    markets=quality_markets,
                    min_score=quality_min_score,
                    similarity_threshold=quality_similarity,
                    max_suffix_pct=quality_max_suffix_pct,
                    max_prefix_pct=quality_max_prefix_pct,
                )
            return names

        elif method == "llm":
            if not self._llm_gen:
                raise ValueError("LLM generation requires ANTHROPIC_API_KEY")
            result = self._llm_gen.generate(count=raw_count, **kwargs)
            if result.error:
                raise ValueError(f"LLM generation failed: {result.error}")
            names = result.names
            for n in names:
                if not hasattr(n, 'method') or not getattr(n, 'method', None):
                    try:
                        setattr(n, 'method', 'llm')
                    except Exception:
                        pass
            if quality_enabled:
                return filter_and_rank(
                    names,
                    target_count=count,
                    markets=quality_markets,
                    min_score=quality_min_score,
                    similarity_threshold=quality_similarity,
                    max_suffix_pct=quality_max_suffix_pct,
                    max_prefix_pct=quality_max_prefix_pct,
                )
            return names

        elif method == "turkic":
            # Turkic-inspired names (VW/Nissan style)
            names = self._turkic_gen.generate(count=raw_count, **kwargs)
            if quality_enabled:
                return filter_and_rank(
                    names,
                    target_count=count,
                    markets=quality_markets,
                    min_score=quality_min_score,
                    similarity_threshold=quality_similarity,
                    max_suffix_pct=quality_max_suffix_pct,
                    max_prefix_pct=quality_max_prefix_pct,
                )
            return names

        elif method == "greek":
            # Greek mythology-inspired names
            names = self._greek_gen.generate(count=raw_count, **kwargs)
            if quality_enabled:
                return filter_and_rank(
                    names,
                    target_count=count,
                    markets=quality_markets,
                    min_score=quality_min_score,
                    similarity_threshold=quality_similarity,
                    max_suffix_pct=quality_max_suffix_pct,
                    max_prefix_pct=quality_max_prefix_pct,
                )
            return names

        elif method == "nordic":
            # Nordic/Scandinavian-inspired names
            names = self._nordic_gen.generate(count=raw_count, **kwargs)
            if quality_enabled:
                return filter_and_rank(
                    names,
                    target_count=count,
                    markets=quality_markets,
                    min_score=quality_min_score,
                    similarity_threshold=quality_similarity,
                    max_suffix_pct=quality_max_suffix_pct,
                    max_prefix_pct=quality_max_prefix_pct,
                )
            return names

        elif method == "japanese":
            # Japanese-inspired names
            gen = self._get_japanese_gen()
            names = gen.generate(count=raw_count, **kwargs)
            if quality_enabled:
                return filter_and_rank(
                    names,
                    target_count=count,
                    markets=quality_markets,
                    min_score=quality_min_score,
                    similarity_threshold=quality_similarity,
                    max_suffix_pct=quality_max_suffix_pct,
                    max_prefix_pct=quality_max_prefix_pct,
                )
            return names

        elif method == "latin":
            # Latin/Romance-inspired names
            gen = self._get_latin_gen()
            names = gen.generate(count=raw_count, **kwargs)
            if quality_enabled:
                return filter_and_rank(
                    names,
                    target_count=count,
                    markets=quality_markets,
                    min_score=quality_min_score,
                    similarity_threshold=quality_similarity,
                    max_suffix_pct=quality_max_suffix_pct,
                    max_prefix_pct=quality_max_prefix_pct,
                )
            return names

        elif method == "celtic":
            # Celtic-inspired names
            gen = self._get_celtic_gen()
            names = gen.generate(count=raw_count, **kwargs)
            if quality_enabled:
                return filter_and_rank(
                    names,
                    target_count=count,
                    markets=quality_markets,
                    min_score=quality_min_score,
                    similarity_threshold=quality_similarity,
                    max_suffix_pct=quality_max_suffix_pct,
                    max_prefix_pct=quality_max_prefix_pct,
                )
            return names

        elif method == "celestial":
            # Celestial/Space-inspired names
            gen = self._get_celestial_gen()
            names = gen.generate(count=raw_count, **kwargs)
            if quality_enabled:
                return filter_and_rank(
                    names,
                    target_count=count,
                    markets=quality_markets,
                    min_score=quality_min_score,
                    similarity_threshold=quality_similarity,
                    max_suffix_pct=quality_max_suffix_pct,
                    max_prefix_pct=quality_max_prefix_pct,
                )
            return names

        elif method == "animals":
            # English wildlife-inspired names
            gen = self._get_animals_gen()
            names = gen.generate(count=raw_count, **kwargs)
            if quality_enabled:
                return filter_and_rank(
                    names,
                    target_count=count,
                    markets=quality_markets,
                    min_score=quality_min_score,
                    similarity_threshold=quality_similarity,
                    max_suffix_pct=quality_max_suffix_pct,
                    max_prefix_pct=quality_max_prefix_pct,
                )
            return names

        elif method == "mythology":
            # Modern mythology-inspired names
            gen = self._get_mythology_gen()
            names = gen.generate(count=raw_count, **kwargs)
            if quality_enabled:
                return filter_and_rank(
                    names,
                    target_count=count,
                    markets=quality_markets,
                    min_score=quality_min_score,
                    similarity_threshold=quality_similarity,
                    max_suffix_pct=quality_max_suffix_pct,
                    max_prefix_pct=quality_max_prefix_pct,
                )
            return names

        elif method == "landmarks":
            # Landmarks and natural wonders-inspired names
            gen = self._get_landmarks_gen()
            names = gen.generate(count=raw_count, **kwargs)
            if quality_enabled:
                return filter_and_rank(
                    names,
                    target_count=count,
                    markets=quality_markets,
                    min_score=quality_min_score,
                    similarity_threshold=quality_similarity,
                    max_suffix_pct=quality_max_suffix_pct,
                    max_prefix_pct=quality_max_prefix_pct,
                )
            return names

        elif method == "blend":
            # Cross-culture blending
            cultures = kwargs.pop('cultures', None)
            if cultures is None:
                cultures = get_setting("generation.blend_default_cultures")
            if not cultures:
                raise ValueError("generation.blend_default_cultures must be set in app.yaml")
            archetype = kwargs.get('archetype')
            blender = CultureBlender(cultures)
            names = blender.blend(count=raw_count, archetype=archetype)
            if quality_enabled:
                return filter_and_rank(
                    names,
                    target_count=count,
                    markets=quality_markets,
                    min_score=quality_min_score,
                    similarity_threshold=quality_similarity,
                    max_suffix_pct=quality_max_suffix_pct,
                    max_prefix_pct=quality_max_prefix_pct,
                )
            return names

        elif method == "all":
            # Mix all cultural methods randomly
            import random
            methods = get_setting("generation.all_methods")
            if not methods:
                raise ValueError("generation.all_methods must be set in app.yaml")
            names = []
            per_method = max(1, count // len(methods))
            remainder = count - (per_method * len(methods))

            for m in methods:
                n = per_method + (1 if remainder > 0 else 0)
                remainder -= 1
                try:
                    batch = self.generate(count=n, method=m, **kwargs)
                    names.extend(batch)
                except Exception:
                    pass  # Skip failed methods

            random.shuffle(names)
            if quality_enabled:
                return filter_and_rank(
                    names,
                    target_count=count,
                    markets=quality_markets,
                    min_score=quality_min_score,
                    similarity_threshold=quality_similarity,
                    max_suffix_pct=quality_max_suffix_pct,
                    max_prefix_pct=quality_max_prefix_pct,
                )
            return names[:count]

        else:
            raise ValueError(f"Unknown method: {method}")

    # -------------------------------------------------------------------------
    # Checking
    # -------------------------------------------------------------------------

    def check(self, name: str, check_all: bool = True, nice_classes=None) -> dict:
        """
        Check a brand name for availability.

        Parameters
        ----------
        name : str
            Brand name to check
        check_all : bool
            If True, check all sources (trademark, domain, similarity).
            If False, only check similarity.
        nice_classes : str, list, or None
            Nice classification filter for trademark search:
            - str: Profile name (e.g., "camping_rv", "electronics")
            - list: Direct class numbers (e.g., [9, 12])
            - None: Search all classes (no filtering)

        Returns
        -------
        dict
            Check results with keys:
            - 'name': The name checked
            - 'available': Overall availability (bool)
            - 'warnings': List of warning messages
            - 'nice_classes': Resolved class list used
            - 'similarity': SimilarityResult object
            - 'domain': DomainCheckResult object (if check_all)
            - 'trademark': TrademarkChecker result dict (if check_all)

        Examples
        --------
            >>> result = kit.check("Voltix", nice_classes="camping_rv")
            >>> if result['available']:
            ...     print("Name is available!")

            >>> result = kit.check("Voltix", nice_classes=[9, 12])
        """
        # Resolve Nice classes from profile name if needed
        resolved_classes = get_nice_classes(nice_classes)

        result = {
            'name': name,
            'available': True,
            'warnings': [],
            'nice_classes': resolved_classes,
        }

        # ----- Similarity Check -----
        sim_result = self._similarity.check(name)
        result['similarity'] = sim_result
        if not sim_result.is_safe:
            result['available'] = False
            similar_names = ', '.join(
                m.known_brand for m in sim_result.similar_brands[:3]
            )
            result['warnings'].append(f"Similar to: {similar_names}")

        if not check_all:
            return result

        # ----- Domain Check -----
        domain_result = self._domain.check(name)
        result['domain'] = domain_result
        if not domain_result.any_available:
            result['warnings'].append("No domains available")

        # ----- Trademark Check -----
        if self._trademark.has_euipo or self._trademark.has_uspto:
            tm_result = self._trademark.check(name, nice_classes=resolved_classes)
            result['trademark'] = tm_result

            if not tm_result['is_available']:
                result['available'] = False

                # Add specific warnings
                if tm_result.get('euipo') and tm_result['euipo'].found:
                    total = (tm_result['euipo'].exact_matches +
                            tm_result['euipo'].similar_matches)
                    result['warnings'].append(f"EU trademark conflicts: {total}")

                if tm_result.get('uspto') and tm_result['uspto'].found:
                    total = (tm_result['uspto'].exact_matches +
                            tm_result['uspto'].similar_matches)
                    result['warnings'].append(f"US trademark conflicts: {total}")

            # Phonetic collision analysis against returned trademark matches
            risk_summary = self._assess_trademark_risk(
                name=name,
                tm_result=tm_result,
                resolved_classes=resolved_classes
            )
            result['trademark_risk'] = risk_summary

            if risk_summary.get('blocking_matches', 0) > 0:
                result['available'] = False
                result['warnings'].append(
                    f"Phonetic collision risk: {risk_summary['blocking_matches']} high/critical matches"
                )
            elif risk_summary.get('medium_matches', 0) > 0:
                result['warnings'].append(
                    f"Phonetic collision risk: {risk_summary['medium_matches']} medium matches (review)"
                )

        return result

    def _assess_trademark_risk(self, name: str, tm_result: dict, resolved_classes=None) -> dict:
        """Assess phonetic collision risk from trademark API results."""
        from .phonetic_similarity import compute_phonetic_similarity, assess_trademark_risk, get_trademark_policy

        def classes_overlap(match_classes):
            if resolved_classes is None:
                return bool(match_classes)
            if not match_classes:
                return True  # conservative when class info is missing
            return any(c in resolved_classes for c in match_classes)

        def risk_for_match(match_name, match_status, is_exact, match_classes):
            phon_sim = compute_phonetic_similarity(name, match_name)
            overlap = classes_overlap(match_classes)
            risk = assess_trademark_risk(
                match_status=match_status,
                is_exact=is_exact,
                phonetic_similarity=phon_sim,
                classes_overlap=overlap,
            )
            return risk, phon_sim

        policy = get_trademark_policy()
        blocking_levels = set(policy.get("blocking_levels", []))

        summary = {
            "critical_matches": 0,
            "high_matches": 0,
            "medium_matches": 0,
            "low_matches": 0,
            "unknown_matches": 0,
            "blocking_matches": 0,
        }

        def process_matches(matches):
            for m in matches or []:
                match_name = getattr(m, "name", None) or str(m)
                match_status = getattr(m, "status", None)
                match_classes = getattr(m, "nice_classes", None)
                is_exact = match_name.lower() == name.lower()
                risk, _ = risk_for_match(match_name, match_status, is_exact, match_classes)

                if risk == "CRITICAL":
                    summary["critical_matches"] += 1
                    if "CRITICAL" in blocking_levels:
                        summary["blocking_matches"] += 1
                elif risk == "HIGH":
                    summary["high_matches"] += 1
                    if "HIGH" in blocking_levels:
                        summary["blocking_matches"] += 1
                elif risk == "MEDIUM":
                    summary["medium_matches"] += 1
                elif risk == "LOW":
                    summary["low_matches"] += 1
                else:
                    summary["unknown_matches"] += 1

        if tm_result.get("uspto") and getattr(tm_result["uspto"], "matches", None):
            process_matches(tm_result["uspto"].matches)
        if tm_result.get("euipo") and getattr(tm_result["euipo"], "matches", None):
            process_matches(tm_result["euipo"].matches)

        return summary

    def store_trademark_matches(self, name: str, check_result: dict) -> int:
        """
        Store trademark matches from a check result with risk assessment.

        Call this after check() to persist conflicting trademarks for later review.
        Computes phonetic similarity and risk level for each match.

        Parameters
        ----------
        name : str
            Brand name that was checked
        check_result : dict
            Result from check() containing 'trademark' field

        Returns
        -------
        int
            Number of matches stored
        """
        from .phonetic_similarity import (
            compute_phonetic_similarity,
            assess_trademark_risk
        )

        tm_data = check_result.get('trademark', {})
        if not tm_data:
            return 0

        resolved_classes = check_result.get('nice_classes')

        def classes_overlap(match_classes):
            if resolved_classes is None:
                return bool(match_classes)
            if not match_classes:
                return True
            return any(c in resolved_classes for c in match_classes)

        def risk_for_match(match_name, match_status, is_exact, match_classes):
            phon_sim = compute_phonetic_similarity(name, match_name)
            overlap = classes_overlap(match_classes)
            risk = assess_trademark_risk(
                match_status=match_status,
                is_exact=is_exact,
                phonetic_similarity=phon_sim,
                classes_overlap=overlap,
            )
            return risk, phon_sim

        total_stored = 0

        # Store USPTO matches
        uspto_result = tm_data.get('uspto')
        if uspto_result and hasattr(uspto_result, 'matches') and uspto_result.matches:
            matches = []
            name_lower = name.lower()
            for m in uspto_result.matches:
                match_name = m.name if hasattr(m, 'name') else str(m)
                is_exact = match_name.lower() == name_lower
                match_status = getattr(m, 'status', None)
                match_classes = getattr(m, 'nice_classes', None)

                # Compute phonetic similarity and risk level
                risk, phon_sim = risk_for_match(match_name, match_status, is_exact, match_classes)

                matches.append({
                    'match_name': match_name,
                    'match_serial': getattr(m, 'serial_number', None) or getattr(m, 'application_number', None),
                    'match_classes': match_classes,
                    'match_status': match_status,
                    'similarity_score': getattr(m, 'similarity_score', None),
                    'is_exact': is_exact,
                    'phonetic_similarity': phon_sim,
                    'risk_level': risk,
                })
            if matches:
                self._db.clear_trademark_matches(name, 'US')
                total_stored += self._db.save_trademark_matches_batch(name, 'US', matches)

        # Store EUIPO matches
        euipo_result = tm_data.get('euipo')
        if euipo_result and hasattr(euipo_result, 'matches') and euipo_result.matches:
            matches = []
            name_lower = name.lower()
            for m in euipo_result.matches:
                match_name = m.name if hasattr(m, 'name') else str(m)
                is_exact = match_name.lower() == name_lower
                match_status = getattr(m, 'status', None)
                match_classes = getattr(m, 'nice_classes', None)

                # Compute phonetic similarity and risk level
                risk, phon_sim = risk_for_match(match_name, match_status, is_exact, match_classes)

                matches.append({
                    'match_name': match_name,
                    'match_serial': getattr(m, 'application_number', None),
                    'match_classes': match_classes,
                    'match_status': match_status,
                    'similarity_score': None,
                    'is_exact': is_exact,
                    'phonetic_similarity': phon_sim,
                    'risk_level': risk,
                })
            if matches:
                self._db.clear_trademark_matches(name, 'EU')
                total_stored += self._db.save_trademark_matches_batch(name, 'EU', matches)

        return total_stored

    def check_and_store(self, name: str, check_all: bool = True, nice_classes=None) -> dict:
        """
        Check a brand name and store trademark matches.

        Combines check() and store_trademark_matches() for convenience.

        Parameters
        ----------
        name : str
            Brand name to check
        check_all : bool
            If True, check all sources
        nice_classes : str, list, or None
            Nice classification filter

        Returns
        -------
        dict
            Check result (same as check())
        """
        result = self.check(name, check_all=check_all, nice_classes=nice_classes)
        if 'trademark' in result:
            matches_stored = self.store_trademark_matches(name, result)
            result['matches_stored'] = matches_stored
        return result

    # -------------------------------------------------------------------------
    # Advanced Features
    # -------------------------------------------------------------------------

    def check_hazards(self, name: str, markets: list = None) -> HazardResult:
        """
        Check a name for cross-linguistic hazards.

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

        Examples
        --------
            >>> result = kit.check_hazards("Voltix")
            >>> if result.is_safe:
            ...     print("Name is safe internationally!")
        """
        return self._hazard_checker.check(name, markets)

    def score_memorability(self, name: str, archetype: str = None,
                           industry: str = None) -> dict:
        """
        Get comprehensive memorability score for a name.

        Parameters
        ----------
        name : str
            Brand name to score
        archetype : str, optional
            Brand archetype (power, elegance, speed, etc.)
        industry : str, optional
            Target industry (tech, pharma, luxury, etc.)

        Returns
        -------
        dict
            Component scores and overall score

        Examples
        --------
            >>> scores = kit.score_memorability("Voltix", archetype="power")
            >>> print(f"Overall: {scores['overall']:.2f}")
        """
        return self._memorability_scorer.score(name, archetype, industry)

    def generate_for_industry(self,
                              industry: str,
                              count: int = None,
                              **kwargs) -> list:
        """
        Generate names optimized for a specific industry.

        Uses industry profiles to select appropriate cultural sources,
        archetypes, and phonetic patterns.

        Parameters
        ----------
        industry : str
            Target industry: 'tech', 'automotive', 'pharma', 'luxury',
            'food_beverage', 'finance', 'energy', 'outdoor', 'wellness',
            'gaming', 'ecommerce'
        count : int
            Number of names to generate
        **kwargs
            Additional arguments passed to generators

        Returns
        -------
        list
            Generated names sorted by score

        Examples
        --------
            >>> names = kit.generate_for_industry("tech", count=10)
            >>> for name in names:
            ...     print(f"{name.name}: {name.score:.2f}")
        """
        profile = self._industry_manager.get_profile(industry)
        if not profile:
            raise ValueError(f"Unknown industry: {industry}. "
                             f"Available: {self._industry_manager.list_industries()}")

        gen_defaults = get_setting("generation.defaults", {}) or {}
        if count is None:
            count = gen_defaults.get("count")
        if count is None:
            raise ValueError("generation.defaults.count must be set in app.yaml")

        fallback_cultures = get_setting("generation.industry_fallback_cultures")
        if not fallback_cultures:
            raise ValueError("generation.industry_fallback_cultures must be set in app.yaml")

        # Get recommended settings from profile
        cultures = profile.get('cultural_sources', fallback_cultures)
        archetypes = profile.get('archetypes', [])
        archetype = kwargs.pop('archetype', None) or (
            archetypes[0] if archetypes else None
        )

        quality_cfg = get_setting("quality_filter", {}) or {}
        quality_enabled = kwargs.pop('post_filter', None)
        if quality_enabled is None:
            quality_enabled = quality_cfg.get("enabled")
        quality_min_score = kwargs.pop('quality_min_score', None)
        if quality_min_score is None:
            quality_min_score = quality_cfg.get("min_score")
        quality_similarity = kwargs.pop('quality_similarity', None)
        if quality_similarity is None:
            quality_similarity = quality_cfg.get("similarity_threshold")
        quality_max_suffix_pct = kwargs.pop('quality_max_suffix_pct', None)
        if quality_max_suffix_pct is None:
            quality_max_suffix_pct = quality_cfg.get("max_suffix_pct")
        quality_max_prefix_pct = kwargs.pop('quality_max_prefix_pct', None)
        if quality_max_prefix_pct is None:
            quality_max_prefix_pct = quality_cfg.get("max_prefix_pct")
        quality_markets = kwargs.get('phonetic_markets')
        if quality_markets is None:
            quality_markets = gen_defaults.get("markets")

        if (quality_enabled is None or quality_min_score is None or
                quality_similarity is None or quality_max_suffix_pct is None or
                quality_max_prefix_pct is None or quality_markets is None):
            raise ValueError("quality_filter settings must be set in app.yaml")

        # Generate from recommended cultures
        all_names = []
        per_culture = max(1, count // len(cultures))

        for culture in cultures:
            try:
                gen_kwargs = dict(kwargs)
                if quality_enabled:
                    gen_kwargs['post_filter'] = False
                batch = self.generate(
                    count=per_culture,
                    method=culture,
                    industry=industry,
                    archetype=archetype,
                    **gen_kwargs
                )
                all_names.extend(batch)
            except Exception:
                pass  # Skip if culture not available

        if quality_enabled:
            return filter_and_rank(
                all_names,
                target_count=count,
                markets=quality_markets,
                min_score=quality_min_score,
                similarity_threshold=quality_similarity,
                max_suffix_pct=quality_max_suffix_pct,
                max_prefix_pct=quality_max_prefix_pct,
            )

        # Sort by score and return
        if all_names:
            all_names.sort(key=lambda n: getattr(n, 'score', 0), reverse=True)
        return all_names[:count]

    def list_industries(self) -> list:
        """List all available industries for generation."""
        return self._industry_manager.list_industries()

    def get_industry_profile(self, industry: str) -> dict:
        """Get the profile for a specific industry."""
        return self._industry_manager.get_profile(industry)

    def analyze_phonaesthetics(self, name: str) -> dict:
        """
        Analyze phonaesthetic properties of a name.

        Returns dict with onset type, vowel character, dominant sounds, etc.
        """
        return self._phonaesthetics.analyze(name)

    def score_for_archetype(self, name: str, archetype: str) -> float:
        """
        Score how well a name fits a brand archetype.

        Archetypes: 'power', 'elegance', 'speed', 'nature', 'tech', 'trust',
                   'innovation'
        """
        return self._phonaesthetics.score_for_archetype(name, archetype)

    def filter_competitors(self, names: list, competitors: list) -> list:
        """
        Filter out names too similar to competitors.

        Parameters
        ----------
        names : list
            Generated names to filter
        competitors : list
            Competitor brand names to avoid

        Returns
        -------
        list
            Names that are distinct from competitors
        """
        differentiator = CompetitiveDifferentiator(competitors)
        return differentiator.filter_names(names)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, name, status: str = "candidate", method: str = None,
             use_llm_meaning: bool = False, industry: str = None) -> NameRecord:
        """
        Save a name to the database with phonaesthetic scores and semantic meaning.

        Parameters
        ----------
        name : GeneratedName, LLMGeneratedName, or str
            The name to save. Can be a generator result object or string.
        status : str
            Initial status. One of: 'new', 'candidate', 'reviewed',
            'finalist', 'rejected', 'blocked'
        method : str, optional
            Generation method (e.g., 'greek', 'japanese'). Extracted from
            name object if available.
        use_llm_meaning : bool
            If True, use Claude LLM to generate richer brand meanings.
            Requires ANTHROPIC_API_KEY. Default is False (template-based).
        industry : str, optional
            Target industry for context in LLM meaning generation.

        Returns
        -------
        NameRecord
            The saved database record

        Examples
        --------
            >>> names = kit.generate(count=5)
            >>> for name in names:
            ...     if kit.check(name.name)['available']:
            ...         kit.save(name)
            >>> # With LLM-enhanced meaning:
            >>> kit.save(name, use_llm_meaning=True, industry='tech')
        """
        from .db import NameStatus
        from .generators.phonemes import phonaesthetic_score
        from .meaning_generator import generate_meaning_with_llm_fallback

        # Extract name string and metadata
        if hasattr(name, 'name'):
            name_str = name.name
            score = getattr(name, 'total_score', None) or \
                    getattr(name, 'score_estimate', 0.5)
            method = method or getattr(name, 'method', None) or "brandkit"
            # Extract semantic info for meaning generation
            roots_used = getattr(name, 'roots_used', [])
            meaning_hints = getattr(name, 'meaning_hints', [])
            culture = getattr(name, 'culture', method)
            category = getattr(name, 'archetype', None)
        else:
            name_str = str(name)
            score = 0.5
            method = method or "brandkit"
            roots_used = []
            meaning_hints = []
            culture = method
            category = None

        # Generate semantic meaning from root components
        roots = list(zip(roots_used, meaning_hints)) if roots_used and meaning_hints else []
        semantic_meaning = generate_meaning_with_llm_fallback(
            name=name_str,
            culture=culture,
            roots=roots,
            category=category,
            use_llm=use_llm_meaning,
            api_key=self._config.anthropic_api_key if self._config.has_anthropic else None,
            industry=industry,
        )

        # Calculate phonaesthetic scores
        phon = phonaesthetic_score(name_str)

        # Convert status string to NameStatus enum
        if isinstance(status, str):
            status = NameStatus(status)

        # Save to database
        name_id = self._db.add(
            name=name_str,
            status=status,
            method=method,
            semantic_meaning=semantic_meaning,
        )

        # Update with phonaesthetic scores if save was successful
        if name_id:
            self._db.update_phonaesthetic_scores(
                name_str,
                overall=phon['score'],
                consonant=phon['consonant_score'],
                vowel=phon['vowel_score'],
                fluency=phon['fluency_score'],
                rhythm=phon['rhythm_score'],
                naturalness=phon['naturalness_score'],
                memorability=phon['memorability_score'],
                cluster_quality=phon['cluster_quality_score'],
                ending_quality=phon['ending_quality_score'],
                quality_tier=phon['quality'],
            )

        return self._db.get(name_str)


# =============================================================================
# Convenience Functions
# =============================================================================

def generate(count: int = None, method: str = None, **kwargs) -> list:
    """
    Quick generation using default BrandKit instance.

    See BrandKit.generate() for full documentation.
    """
    kit = BrandKit()
    return kit.generate(count, method, **kwargs)


def check(name: str, nice_classes=None) -> dict:
    """
    Quick check using default BrandKit instance.

    See BrandKit.check() for full documentation.
    """
    kit = BrandKit()
    return kit.check(name, nice_classes=nice_classes)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    '__version__',

    # Main class
    'BrandKit',

    # Rule-based Generators
    'BrandGenerator',
    'BrandNameGenerator',
    'RuleBasedGenerator',
    'LLMGenerator',
    'GeneratedName',
    'NameScore',

    # Cultural Generators
    'TurkicGenerator',
    'GreekGenerator',
    'NordicGenerator',
    'JapaneseGenerator',
    'LatinGenerator',
    'CelticGenerator',
    'CelestialGenerator',
    'TurkicName',
    'GreekName',
    'NordicName',

    # Base Generator Framework
    'CulturalGenerator',
    'CulturalGeneratedName',
    'HazardChecker',
    'HazardResult',
    'SyllableAnalyzer',
    'SyllableInfo',
    'PhonaestheticsEngine',
    'IndustryManager',
    'MemorabilityScorer',
    'CultureBlender',
    'CompetitiveDifferentiator',

    # Checkers
    'TrademarkChecker',
    'EUIPOChecker',
    'RapidAPIChecker',
    'DomainChecker',
    'SimilarityChecker',
    'check_domain',
    'check_similarity',

    # Database
    'NameDB',
    'BrandNameDB',
    'NameStatus',
    'NameRecord',
    'BrandName',
    'get_db',
    'get_namedb',

    # Config
    'Config',
    'get_config',
    'load_env',
    'NICE_PROFILES',
    'get_nice_classes',
    'list_profiles',

    # Convenience functions
    'generate',
    'check',

    # Submodules
    'generators',
    'checkers',
    'db',
    'config',
]
