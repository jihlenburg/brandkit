#!/usr/bin/env python3
"""
Brand Name Generators
=====================
Provides multiple generation strategies:
- RuleBased: Phonetic rules and morpheme combinations
- Markov: Statistical character-level generation
- LLM: Claude-powered creative generation
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
from markov_generator import (
    MarkovGenerator,
    MarkovModel,
    MarkovTrainer,
    HybridMarkovGenerator,
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
BrandGenerator = BrandNameGenerator  # Alias for convenience
GeneratedName = NameScore  # Alias for consistency

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
    # Markov
    'MarkovGenerator',
    'MarkovModel',
    'MarkovTrainer',
    'HybridMarkovGenerator',
    # LLM
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
]
