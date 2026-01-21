#!/usr/bin/env python3
"""
Brand Name Generators
=====================
Provides multiple generation strategies:
- RuleBased: Phonetic rules and morpheme combinations
- Cultural: Greek, Nordic, Japanese, Latin, Celtic, Celestial, Turkic
- LLM: Claude-powered creative generation (optional)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

# Import generators from existing modules
from brand_generator import (
    BrandNameGenerator,
    NameScore,
    NameScorer,
    SoundQuality,
    VowelQuality,
    Phoneme,
    Syllable,
    SemanticMorpheme,
    SyllableGenerator,
)
from llm_generator import (
    LLMGenerator,
    LLMGeneratedName,
    LLMGenerationResult,
    get_llm_generator,
)
from .turkic_generator import (
    TurkicGenerator,
    TurkicName,
    generate_turkic,
)
from .greek_generator import (
    GreekGenerator,
    GreekName,
    generate_greek,
)
from .nordic_generator import (
    NordicGenerator,
    NordicName,
    generate_nordic,
)
from .cultural_generators import (
    JapaneseGenerator,
    LatinGenerator,
    CelticGenerator,
    CelestialGenerator,
    AnimalsGenerator,
    MythologyGenerator,
    LandmarksGenerator,
)
from .base_generator import (
    CulturalGenerator,
    GeneratedName as CulturalGeneratedName,
    HazardChecker,
    HazardResult,
    SyllableAnalyzer,
    SyllableInfo,
    PhonaestheticsEngine,
    IndustryManager,
    MemorabilityScorer,
    CultureBlender,
    CompetitiveDifferentiator,
)

# Clean aliases
RuleBasedGenerator = BrandNameGenerator
PhoneticGenerator = BrandNameGenerator
BrandGenerator = BrandNameGenerator
GeneratedName = NameScore

__all__ = [
    # Rule-based
    'RuleBasedGenerator',
    'PhoneticGenerator',
    'BrandGenerator',
    'BrandNameGenerator',
    'GeneratedName',
    'NameScore',
    'NameScorer',
    'SoundQuality',
    'VowelQuality',
    'Phoneme',
    'Syllable',
    'SemanticMorpheme',
    'SyllableGenerator',
    # LLM (optional)
    'LLMGenerator',
    'LLMGeneratedName',
    'LLMGenerationResult',
    'get_llm_generator',
    # Turkic
    'TurkicGenerator',
    'TurkicName',
    'generate_turkic',
    # Greek
    'GreekGenerator',
    'GreekName',
    'generate_greek',
    # Nordic
    'NordicGenerator',
    'NordicName',
    'generate_nordic',
    # Cultural Generators
    'JapaneseGenerator',
    'LatinGenerator',
    'CelticGenerator',
    'CelestialGenerator',
    'AnimalsGenerator',
    'MythologyGenerator',
    'LandmarksGenerator',
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
]
