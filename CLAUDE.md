# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**Brandkit** - A comprehensive brand name generator and validator for German/English markets, with advanced linguistic features for culturally-aware, internationally-safe brand naming.

**Version:** 0.1.0 (see CHANGELOG.md for history)

### Core Capabilities

1. **Name Generation**
   - Rule-based phonetic generation (CV patterns, hardness control, semantic morphemes)
   - Markov chain generation (trained on 300+ successful brand names)
   - LLM-powered generation (Claude API for creative naming)
   - Cultural generators (Greek, Turkic, Nordic, Japanese, Latin, Celtic, Celestial)
   - Cross-culture blending for unique hybrid names
   - Industry-optimized generation (11 industry profiles)

2. **Linguistic Analysis**
   - Phonaesthetics (sound symbolism) - names that "sound like" their meaning
   - **Phonaesthetic scoring** - Research-backed sound beauty assessment (Crystal, Yorkston & Menon)
   - Brand archetypes (power, elegance, speed, nature, tech, trust, innovation)
   - Cross-linguistic hazard checking (avoid vulgar sound-alikes in other languages)
   - Syllable stress/rhythm analysis (trochaic, iambic, dactylic patterns)
   - **Pronounceability gate** - Hard filtering of unpronounceable names (Sonority Sequencing Principle)
   - Memorability scoring (pronounceability, distinctiveness, visual balance)
   - Competitive differentiation (ensure distinctiveness from competitors)

3. **Name Validation**
   - EU trademark search (EUIPO API with Nice class filtering)
   - US trademark search (RapidAPI/USPTO with post-filtering)
   - Domain availability (DNS-based checking for .com, .de, .eu)
   - Phonetic similarity (Soundex, Metaphone, Levenshtein against known brands)

4. **Database Management**
   - SQLite storage for generated names with automatic migrations
   - Status workflow (new -> candidate -> shortlist -> approved/rejected/blocked)
   - **Phonaesthetic scores stored** (overall, consonant, vowel, fluency, rhythm, naturalness)
   - **Quality tier tracking** (excellent/good/acceptable/poor)
   - **Validation results stored** (EU/US conflicts, domain availability, timestamps)
   - Class-specific trademark conflict tracking

## Project Structure

```
brands/
├── brandkit/                    # Main package (use this)
│   ├── __init__.py              # BrandKit class, public API
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Nice class profiles, API config
│   ├── generators/              # Name generation modules
│   │   ├── __init__.py          # Generator exports
│   │   ├── base_generator.py    # CulturalGenerator base class
│   │   │                        # HazardChecker, SyllableAnalyzer
│   │   │                        # PhonaestheticsEngine, IndustryManager
│   │   │                        # MemorabilityScorer, CultureBlender
│   │   │                        # CompetitiveDifferentiator
│   │   ├── cultural_generators.py  # Japanese, Latin, Celtic, Celestial
│   │   ├── greek_generator.py   # Greek mythology generator
│   │   ├── turkic_generator.py  # Turkic language generator
│   │   ├── nordic_generator.py  # Nordic/Scandinavian generator
│   │   ├── llm_generator.py     # Claude-powered generation
│   │   └── phonemes/            # YAML phoneme configurations
│   │       ├── __init__.py      # Config loaders
│   │       ├── greek.yaml       # Greek mythology roots
│   │       ├── turkic.yaml      # Turkic language roots
│   │       ├── nordic.yaml      # Nordic/Scandinavian roots
│   │       ├── japanese.yaml    # Japanese roots (89)
│   │       ├── latin.yaml       # Latin/Romance roots (135)
│   │       ├── celtic.yaml      # Celtic roots (124)
│   │       ├── celestial.yaml   # Celestial/Space roots (130+)
│   │       ├── phonaesthemes.yaml  # Sound symbolism (7 archetypes)
│   │       ├── hazards.yaml     # Cross-linguistic hazards
│   │       ├── industries.yaml  # Industry profiles (11)
│   │       └── strategies.yaml  # Generation strategies
│   ├── checkers/                # Validation modules
│   │   └── __init__.py          # TrademarkChecker unified interface
│   └── db/                      # Database modules
│       └── __init__.py          # Database exports
│
├── brand_generator.py           # Rule-based generator (standalone)
├── markov_generator.py          # Markov generator (standalone)
├── euipo_checker.py             # EUIPO API client (standalone)
├── rapidapi_checker.py          # USPTO/WIPO via RapidAPI (standalone)
├── domain_checker.py            # DNS domain checker (standalone)
├── similarity_checker.py        # Phonetic similarity (standalone)
├── namedb.py                    # SQLite database (standalone)
│
├── .env                         # API credentials (not in git)
└── brandnames.db                # SQLite database (auto-created)
```

## API Credentials

Create a `.env` file in the project root:

```bash
# EUIPO API (https://dev.euipo.europa.eu/)
EUIPO_CLIENT_ID=your_client_id
EUIPO_CLIENT_SECRET=your_client_secret

# RapidAPI for USPTO/WIPO (https://rapidapi.com/)
RAPIDAPI_KEY=your_rapidapi_key

# Anthropic Claude API (for LLM generation)
ANTHROPIC_API_KEY=your_anthropic_key
```

## Quick Start

### Python API

```python
from brandkit import BrandKit

kit = BrandKit()

# Generate names with cultural generators
names = kit.generate(count=10, method="japanese")
for name in names:
    print(f"{name.name}: {name.score:.2f}")

# Generate for specific industry
names = kit.generate_for_industry("tech", count=10)

# Cross-culture blending
names = kit.generate(count=10, method="blend",
                     cultures=["greek", "latin", "japanese"])

# Check for cross-linguistic hazards
result = kit.check_hazards("Voltix")
if result.is_safe:
    print("Name is safe internationally!")

# Score memorability with archetype
scores = kit.score_memorability("Voltara", archetype="power")
print(f"Overall: {scores['overall']:.2f}")

# Analyze phonaesthetics
analysis = kit.analyze_phonaesthetics("Glowix")
print(f"Onset: {analysis['onset']}, Feel: {analysis['feel']}")

# Phonaesthetic scoring (research-backed)
from brandkit.generators.phonemes import phonaesthetic_score, score_name
result = phonaesthetic_score("Lumina", category="luxury")
print(f"Score: {result['score']:.2f} ({result['quality']})")
# Breakdown: consonant, vowel, fluency, naturalness, rhythm scores

# Score for brand archetype
score = kit.score_for_archetype("Voltix", "power")

# Filter competitors
filtered = kit.filter_competitors(names, ["Tesla", "Volta", "Enphase"])

# Check availability with Nice class profile
result = kit.check("Voltix", nice_classes="camping_rv")
if result['available']:
    kit.save(name, status="candidate")
```

### CLI

```bash
# Generate names with various methods
python -m brandkit generate -n 10
python -m brandkit generate -n 10 --method japanese
python -m brandkit generate -n 10 --method nordic
python -m brandkit generate -n 10 --method blend

# Check a name (with Nice class profile)
python -m brandkit check "Voltix" --full --profile camping_rv

# List available profiles and industries
python -m brandkit profiles
```

## Generation Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `rule_based` | Phonetic rules, CV patterns | General purpose |
| `markov` | Statistical character generation | Realistic names |
| `llm` | Claude-powered creative generation | Creative/conceptual |
| `hybrid` | Combination of rule-based + Markov | Balanced |
| `greek` | Greek mythology roots | Power, tech |
| `turkic` | Turkic language vowel harmony | Automotive (VW style) |
| `nordic` | Norse mythology, Scandinavian | Outdoor, strength |
| `japanese` | CV patterns, minimalist | Tech, gaming |
| `latin` | Classical/Romance elegance | Luxury, pharma |
| `celtic` | Nature, mythology | Craft, heritage |
| `celestial` | Space/astronomy roots | Tech, aerospace, energy |
| `blend` | Cross-culture fusion | Unique names |
| `all` | Mix all methods | Variety |

## Industry Profiles

Available industries for `generate_for_industry()`:

| Industry | Cultural Sources | Archetypes |
|----------|-----------------|------------|
| `tech` | Greek, Japanese, Latin, Celestial | Innovation, speed |
| `automotive` | Turkic, Nordic, Latin | Power, speed |
| `pharma` | Greek, Latin | Trust, precision |
| `luxury` | Latin, French | Elegance |
| `food_beverage` | Latin, Celtic, Nordic | Warmth, nature |
| `finance` | Latin, Greek | Trust, stability |
| `energy` | Greek, Nordic, Celestial | Power, nature |
| `outdoor` | Nordic, Celtic | Nature, adventure |
| `wellness` | Japanese, Latin, Celtic | Nature, trust |
| `gaming` | Japanese, Greek, Nordic | Innovation, speed |
| `ecommerce` | Greek, Japanese | Innovation, speed |

## Brand Archetypes

For `score_for_archetype()` and phonaesthetics:

- **power** - Strong consonants (k, t, v, x), short vowels
- **elegance** - Soft consonants (l, s, f), flowing sounds
- **speed** - Sharp sounds (z, x, v), quick rhythm
- **nature** - Soft consonants, earthy vowels
- **tech** - Crisp consonants (x, k, z), modern feel
- **trust** - Solid consonants (b, d, m), stable rhythm
- **innovation** - Unusual combinations, distinctive sounds

## Cross-Linguistic Hazards

The `check_hazards()` method detects:

- **Exact word hazards**: "gift" (poison in German), "mist" (manure)
- **Sound-alike hazards**: Names that sound vulgar when pronounced
- **Phonetic patterns**: Regex patterns for risky sound combinations
- **Cultural sensitivities**: Religious terms, sacred concepts

```python
result = kit.check_hazards("Gift")
# HazardResult(is_safe=False, severity='critical', issues=[...])
```

## Nice Classification System

### Predefined Profiles

| Profile | Classes | Description |
|---------|---------|-------------|
| `camping_rv` | 7, 9, 11, 12 | Camping equipment, RV/caravan accessories |
| `power_electronics` | 7, 9 | DC/DC converters, inverters, power supplies |
| `electronics` | 9 | Electrical/electronic apparatus, computers |
| `software` | 9, 42 | Computer software and services |
| `energy` | 7, 9, 11 | Power generation, electrical systems |
| `automotive` | 7, 9, 12 | Vehicles, automotive parts |
| `clothing` | 25, 35 | Clothing, footwear, headgear |
| `household` | 11, 21 | Household appliances and utensils |
| `food_beverage` | 29, 30, 32, 33 | Food products and beverages |
| `retail` | 35 | Retail services, advertising |
| `consulting` | 35, 42 | Business and technical consulting |

## Key Classes

### BrandKit Methods

```python
kit = BrandKit()

# Generation
kit.generate(count, method, **kwargs)
kit.generate_for_industry(industry, count, **kwargs)

# Linguistic Analysis
kit.check_hazards(name, markets=None) -> HazardResult
kit.score_memorability(name, archetype=None, industry=None) -> dict
kit.analyze_phonaesthetics(name) -> dict
kit.score_for_archetype(name, archetype) -> float
kit.filter_competitors(names, competitors) -> list

# Industry Info
kit.list_industries() -> list
kit.get_industry_profile(industry) -> dict

# Validation
kit.check(name, check_all=True, nice_classes=None) -> dict

# Persistence (auto-calculates phonaesthetic scores)
kit.save(name, status="candidate", method="greek") -> NameRecord
```

### Cultural Generators

```python
from brandkit.generators import (
    JapaneseGenerator,
    LatinGenerator,
    CelticGenerator,
    CelestialGenerator,
)

gen = JapaneseGenerator(seed=42)
names = gen.generate(
    count=20,
    categories=['nature', 'light'],  # Filter root categories
    archetype='tech',                  # Brand archetype
    industry='gaming',                 # Industry profile
    min_length=4,
    max_length=9,
    check_hazards=True,
    markets=['german', 'spanish'],     # Markets to check
)
```

### Base Generator Components

```python
from brandkit.generators import (
    HazardChecker,
    SyllableAnalyzer,
    PhonaestheticsEngine,
    IndustryManager,
    MemorabilityScorer,
    CultureBlender,
    CompetitiveDifferentiator,
)

# Check hazards
checker = HazardChecker()
result = checker.check("Voltix", markets=['german', 'french'])

# Analyze syllables
analyzer = SyllableAnalyzer()
info = analyzer.analyze("Voltara")  # SyllableInfo(count=3, pattern='STRONG-weak-weak', ...)

# Phonaesthetics
engine = PhonaestheticsEngine()
score = engine.score_for_archetype("Glowix", "tech")
analysis = engine.analyze("Glowix")

# Industry profiles
manager = IndustryManager()
profile = manager.get_profile("tech")
industries = manager.list_industries()

# Memorability
scorer = MemorabilityScorer()
scores = scorer.score("Voltara", archetype="power")

# Cross-culture blending
blender = CultureBlender(['greek', 'latin', 'japanese'])
names = blender.blend(count=20, archetype="innovation")

# Competitive differentiation
diff = CompetitiveDifferentiator(['Tesla', 'Volta', 'Enphase'])
filtered = diff.filter_names(names)
is_distinct = diff.is_distinct("Voltix", threshold=0.5)
```

## YAML Configuration Files

All in `brandkit/generators/phonemes/`:

| File | Contents |
|------|----------|
| `greek.yaml` | Greek mythology roots (gods, beasts, concepts) |
| `turkic.yaml` | Turkic language roots with vowel harmony rules |
| `nordic.yaml` | Norse mythology, Scandinavian nature words |
| `japanese.yaml` | 89 Japanese roots (CV patterns, avoids shi/ku) |
| `latin.yaml` | 135 Latin/Romance roots (classical, Italian, French) |
| `celtic.yaml` | 124 Celtic roots (Irish, Welsh, Scottish) |
| `celestial.yaml` | 130+ Celestial/Space roots (stars, planets, cosmos) |
| `phonaesthemes.yaml` | Sound symbolism, 7 brand archetypes |
| `hazards.yaml` | Cross-linguistic hazards, sound-alikes, patterns |
| `industries.yaml` | 11 industry profiles with cultural sources |
| `strategies.yaml` | Phonotactics, scoring rules, rhythm, phonaesthetic quality, pronounceability gates |

### Loading Configs Directly

```python
from brandkit.generators.phonemes import (
    load_japanese, load_latin, load_celtic, load_celestial,
    load_phonaesthemes, load_hazards, load_industries,
    get_all_cultures, get_culture,
)

# Load a culture
jp = load_japanese()
roots = jp.get_all_roots()  # List of (phoneme, meaning, category)

# Load all cultures
cultures = get_all_cultures()  # Dict[str, PhonemeConfig]

# Load meta configs
hazards = load_hazards()  # HazardsConfig
industries = load_industries()  # IndustriesConfig
```

### Phonaesthetic Scoring (Research-Backed)

```python
from brandkit.generators.phonemes import (
    phonaesthetic_score,
    score_name,
    is_pronounceable,
    analyze_rhythm,
    syllabify,
)

# Check pronounceability (hard gate)
is_ok, reason = is_pronounceable("Resafafn")
# Returns: (False, "unpronounceable_ending:fn")

# Get detailed phonaesthetic breakdown
result = phonaesthetic_score("Lumina", category="luxury")
# Returns:
#   score: 0.61 (overall 0.0-1.0)
#   quality: "acceptable" | "good" | "excellent" | "poor"
#   consonant_score: 0.56 (pleasant consonants: l,m,n,s,r)
#   vowel_score: 0.53 (front vowels preferred)
#   fluency_score: 0.55 (CV balance, repetition)
#   naturalness_score: 0.51 (phonotactic patterns)
#   rhythm_score: 1.00 (trochaic = ideal)
#   category_fit_score: 0.67 (matches luxury sounds)

# Analyze rhythm (metrical phonology)
rhythm = analyze_rhythm("Voltix")
# Returns:
#   syllables: ['vol', 'tix']
#   syllable_count: 2
#   weights: ['H', 'H']  # Heavy syllables
#   stress_pattern: 'SU'  # STRONG-weak
#   rhythm_type: 'trochaic'
#   rhythm_score: 1.0

# Combined score (pronounceability gate + phonaesthetics)
score = score_name("Voltix", category="tech")
# Returns: 0.66 (0.0 if unpronounceable)
```

**Research References:**
- Matzinger & Kosic (2025): Pleasant words are easier to remember
- Nemestothy et al. (2024): Front vowels rated more pleasant
- David Crystal: Beautiful words have l, m, s, n; 2-3 syllables; first-syllable stress
- Yorkston & Menon (2004): Sound symbolism in brand names

### Database API (v0.4.0)

```python
from brandkit.db import get_db, NameStatus, QualityTier

db = get_db()

# Status management
db.update_status("Voltix", NameStatus.CANDIDATE)
db.update_status("Voltix", "candidate")  # String also works
count = db.count_by_status(NameStatus.NEW)

# Phonaesthetic scores (auto-set by kit.save(), or manual)
db.update_phonaesthetic_scores(
    "Lumina",
    overall=0.61,
    consonant=0.56,
    vowel=0.53,
    fluency=0.55,
    rhythm=1.0,
    naturalness=0.51,
    quality_tier="excellent"
)

# Validation results (summary flags for quick filtering)
db.update_validation_results(
    "Lumina",
    eu_conflict=False,   # True if ANY EU class has conflict
    us_conflict=True,    # True if ANY US class has conflict
    domains={'.com': False, '.de': True, '.eu': True}
)

# Class-specific trademark results (detailed)
db.save_trademark_check(
    "Lumina",
    nice_class=9,
    region="EU",
    available=True,
    conflicts_count=0
)
db.save_trademark_check(
    "Lumina",
    nice_class=12,
    region="EU",
    available=False,
    conflicts_count=3,
    conflict_details='["LUMINA AUTO", "LUMINEX"]'
)

# Query class-specific results
checks = db.get_trademark_checks("Lumina")
# Returns: {'EU': {9: {'available': True, ...}, 12: {'available': False, ...}}}

available_classes = db.get_available_classes("Lumina", region="EU")
# Returns: [9]  (class 9 is available, class 12 has conflicts)

conflicting_classes = db.get_conflicting_classes("Lumina", region="EU")
# Returns: [12]

# Query by quality
excellent = db.get_excellent(limit=50)
good = db.get_by_quality_tier(QualityTier.GOOD)

# Query by validation status (uses summary flags)
available = db.get_available()   # No EU/US conflicts in ANY class
conflicts = db.get_conflicts()   # Has EU or US conflicts in ANY class

# Access stored data
brand = db.get("Lumina")
print(brand.score_phonaesthetic)  # 0.61
print(brand.quality_tier.value)   # "excellent"
print(brand.eu_conflict)          # False
print(brand.us_conflict)          # True
print(brand.validated_at)         # ISO timestamp
```

**Database Schema (v0.4.0 additions to `names` table):**

| Column | Type | Description |
|--------|------|-------------|
| `score_phonaesthetic` | REAL | Overall phonaesthetic score (0.0-1.0) |
| `score_consonant` | REAL | Consonant quality score |
| `score_vowel` | REAL | Vowel quality score |
| `score_fluency` | REAL | Processing fluency score |
| `score_rhythm` | REAL | Rhythm score |
| `score_naturalness` | REAL | Phonotactic naturalness score |
| `quality_tier` | TEXT | excellent/good/acceptable/poor |
| `validated_at` | TEXT | Validation timestamp (ISO) |
| `eu_conflict` | INTEGER | Summary: any EU conflict? (0/1) |
| `us_conflict` | INTEGER | Summary: any US conflict? (0/1) |
| `domains_available` | TEXT | JSON dict of domain availability |

**Class-Specific Trademark Results (`trademark_checks` table):**

| Column | Type | Description |
|--------|------|-------------|
| `name_id` | INTEGER | FK to names table |
| `nice_class` | INTEGER | Nice class (1-45) |
| `region` | TEXT | EU, US, or WIPO |
| `available` | INTEGER | Available in this class? (0/1) |
| `conflicts_count` | INTEGER | Number of conflicts |
| `conflict_details` | TEXT | JSON with conflict names |
| `checked_at` | TEXT | Timestamp |

**Enums:**

```python
class NameStatus(Enum):
    NEW = "new"
    CANDIDATE = "candidate"
    SHORTLIST = "shortlist"
    APPROVED = "approved"
    REJECTED = "rejected"
    BLOCKED = "blocked"

class QualityTier(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"

class BlockReason(Enum):
    EUIPO_CONFLICT = "euipo_conflict"
    USPTO_CONFLICT = "uspto_conflict"
    UNPRONOUNCEABLE = "unpronounceable"
    # ... and others
```

## Versioning & Release Workflow

**IMPORTANT: Always follow this workflow for any code changes.**

### Version Files
- `brandkit/__init__.py` - `__version__ = "X.Y.Z"`
- `pyproject.toml` - `version = "X.Y.Z"`
- `CHANGELOG.md` - Release notes

### Semantic Versioning (SemVer)
- **MAJOR (X)**: Breaking API changes
- **MINOR (Y)**: New features, backward compatible
- **PATCH (Z)**: Bug fixes, backward compatible

### Workflow for Every Change

1. **Before committing**: Update `CHANGELOG.md`
   - Add changes under `[Unreleased]` section
   - Use categories: Added, Changed, Fixed, Removed, Security

2. **When releasing a new version**:
   ```bash
   # 1. Move unreleased items to new version section in CHANGELOG.md
   # 2. Update version in BOTH files:
   #    - brandkit/__init__.py: __version__ = "X.Y.Z"
   #    - pyproject.toml: version = "X.Y.Z"
   # 3. Commit
   git commit -m "Release vX.Y.Z"
   # 4. Tag
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   # 5. Push
   git push && git push --tags
   ```

### CHANGELOG Format (Keep a Changelog)
```markdown
## [Unreleased]

### Added
- New feature description

### Changed
- Modified behavior

### Fixed
- Bug fix description

## [X.Y.Z] - YYYY-MM-DD

### Added
...
```

### Commit Message Convention
- `feat:` New feature (bumps MINOR)
- `fix:` Bug fix (bumps PATCH)
- `docs:` Documentation only
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

## Development Notes

### YAML Gotchas

YAML parses certain values as booleans:
- `on`, `off`, `yes`, `no`, `true`, `false`

Always quote these as strings: `"on"`, `"no"`

### Key Design Decisions

1. **Cultural generators use abstract base class**: `CulturalGenerator` provides hazard checking, syllable analysis, memorability scoring to all generators.

2. **Lazy loading**: Expensive components (Markov, LLM, new cultural generators) are loaded on first use.

3. **YAML-driven configuration**: All phoneme data, archetypes, hazards, and industry profiles are in YAML for easy editing.

4. **Post-filtering for RapidAPI**: USPTO API doesn't support class filtering, so results are filtered after retrieval.

### Testing

```bash
# Test phoneme loading
python3 -c "from brandkit.generators.phonemes import get_all_cultures; print(get_all_cultures().keys())"

# Test generators
python3 -c "from brandkit import BrandKit; kit = BrandKit(); print(kit.generate(count=5, method='japanese'))"

# Test hazard checking
python3 -c "from brandkit import BrandKit; kit = BrandKit(); print(kit.check_hazards('Gift'))"
```

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

Current version: **0.1.0** (2026-01-20)
