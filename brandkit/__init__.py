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

__version__ = "0.3.0"
__author__ = "BrandKit"

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
    MarkovGenerator,
    HybridMarkovGenerator,
    LLMGenerator,
    TurkicGenerator,
    GreekGenerator,
    NordicGenerator,
    # New cultural generators
    JapaneseGenerator,
    LatinGenerator,
    CelticGenerator,
    CelestialGenerator,
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
        self._markov_gen = None  # Lazy-loaded (training takes time)
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

    def _get_markov_gen(self):
        """Lazy-load the Markov generator (training takes time)."""
        if self._markov_gen is None:
            self._markov_gen = HybridMarkovGenerator()
        return self._markov_gen

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

    def generate(self,
                 count: int = 10,
                 method: str = "rule_based",
                 **kwargs) -> list:
        """
        Generate brand names using the specified method.

        Parameters
        ----------
        count : int
            Number of names to generate (default: 10)
        method : str
            Generation method. One of:
            - "rule_based": Phonetic rules and morpheme combinations
            - "markov": Statistical character-level generation
            - "llm": Claude-powered creative generation
            - "hybrid": Mix of rule-based and Markov
        **kwargs
            Additional arguments passed to the generator

        Returns
        -------
        list
            List of generated names. Type depends on method:
            - rule_based/hybrid: List[NameScore]
            - markov: List[str]
            - llm: List[LLMGeneratedName]

        Raises
        ------
        ValueError
            If method is unknown or LLM requested without API key

        Examples
        --------
            >>> names = kit.generate(count=5, method="rule_based")
            >>> names = kit.generate(count=5, method="llm",
            ...                      context="Solar panels")
        """
        if method == "rule_based":
            return self._rule_gen.generate_batch(count, **kwargs)

        elif method == "markov":
            markov_gen = self._get_markov_gen()
            results = markov_gen.generate(count=count, **kwargs)
            return [name for name, _ in results]

        elif method == "llm":
            if not self._llm_gen:
                raise ValueError("LLM generation requires ANTHROPIC_API_KEY")
            result = self._llm_gen.generate(count=count, **kwargs)
            if result.error:
                raise ValueError(f"LLM generation failed: {result.error}")
            return result.names

        elif method == "hybrid":
            # Mix of rule-based and Markov
            rule_count = count // 2
            markov_count = count - rule_count
            names = self._rule_gen.generate_batch(rule_count, **kwargs)
            markov_gen = self._get_markov_gen()
            markov_gen.generate(count=markov_count, **kwargs)
            return sorted(names, key=lambda n: n.total_score, reverse=True)

        elif method == "turkic":
            # Turkic-inspired names (VW/Nissan style)
            return self._turkic_gen.generate(count=count, **kwargs)

        elif method == "greek":
            # Greek mythology-inspired names
            return self._greek_gen.generate(count=count, **kwargs)

        elif method == "nordic":
            # Nordic/Scandinavian-inspired names
            return self._nordic_gen.generate(count=count, **kwargs)

        elif method == "japanese":
            # Japanese-inspired names
            gen = self._get_japanese_gen()
            return gen.generate(count=count, **kwargs)

        elif method == "latin":
            # Latin/Romance-inspired names
            gen = self._get_latin_gen()
            return gen.generate(count=count, **kwargs)

        elif method == "celtic":
            # Celtic-inspired names
            gen = self._get_celtic_gen()
            return gen.generate(count=count, **kwargs)

        elif method == "celestial":
            # Celestial/Space-inspired names
            gen = self._get_celestial_gen()
            return gen.generate(count=count, **kwargs)

        elif method == "blend":
            # Cross-culture blending
            cultures = kwargs.pop('cultures', ['greek', 'latin', 'nordic', 'japanese'])
            archetype = kwargs.get('archetype')
            blender = CultureBlender(cultures)
            return blender.blend(count=count, archetype=archetype)

        elif method == "all":
            # Mix all methods randomly
            import random
            methods = ["rule_based", "markov", "turkic", "greek", "nordic",
                       "japanese", "latin", "celtic", "celestial"]
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
                              count: int = 20,
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

        # Get recommended settings from profile
        cultures = profile.get('cultural_sources', ['greek', 'latin'])
        archetypes = profile.get('archetypes', [])
        archetype = kwargs.pop('archetype', None) or (
            archetypes[0] if archetypes else None
        )

        # Generate from recommended cultures
        all_names = []
        per_culture = max(1, count // len(cultures))

        for culture in cultures:
            try:
                batch = self.generate(
                    count=per_culture,
                    method=culture,
                    industry=industry,
                    archetype=archetype,
                    **kwargs
                )
                all_names.extend(batch)
            except Exception:
                pass  # Skip if culture not available

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

    def save(self, name, status: str = "candidate", method: str = None) -> NameRecord:
        """
        Save a name to the database with phonaesthetic scores.

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
        """
        from .db import NameStatus
        from .generators.phonemes import phonaesthetic_score

        # Extract name string and metadata
        if hasattr(name, 'name'):
            name_str = name.name
            score = getattr(name, 'total_score', None) or \
                    getattr(name, 'score_estimate', 0.5)
            method = method or getattr(name, 'method', None) or "brandkit"
        else:
            name_str = str(name)
            score = 0.5
            method = method or "brandkit"

        # Calculate phonaesthetic scores
        phon = phonaesthetic_score(name_str)

        # Convert status string to NameStatus enum
        if isinstance(status, str):
            status = NameStatus(status)

        # Save to database with phonaesthetic scores
        name_id = self._db.add(
            name=name_str,
            score=score,
            status=status,
            method=method,
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
                quality_tier=phon['quality'],
            )

        return self._db.get(name_str)


# =============================================================================
# Convenience Functions
# =============================================================================

def generate(count: int = 10, method: str = "rule_based", **kwargs) -> list:
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

    # Legacy Generators
    'BrandGenerator',
    'BrandNameGenerator',
    'RuleBasedGenerator',
    'MarkovGenerator',
    'HybridMarkovGenerator',
    'LLMGenerator',
    'TurkicGenerator',
    'GreekGenerator',
    'NordicGenerator',
    'GeneratedName',
    'NameScore',
    'TurkicName',
    'GreekName',
    'NordicName',

    # New Cultural Generators
    'JapaneseGenerator',
    'LatinGenerator',
    'CelticGenerator',
    'CelestialGenerator',

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
