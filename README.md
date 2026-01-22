# BrandKit

A comprehensive brand name generator and validator for German and English markets.

Generate phonetically pleasing brand names using linguistic principles from multiple cultures, validate them against EU/US trademark databases, check domain availability, and ensure cross-linguistic safety.

## Features

### Name Generation

- **Rule-based phonetic generation** with customizable hardness
- **Markov chain generation** trained on successful brand names
- **LLM-powered generation** (Claude API)
- **Cultural generators** drawing from world linguistics:
  - Greek mythology (beasts, gods, concepts)
  - Turkic languages (vowel harmony, steppe imagery)
  - Nordic/Scandinavian (Norse mythology, nature)
  - Japanese (minimalist, tech-forward CV patterns)
  - Latin/Romance (classical, Italian, French elegance)
  - Celtic (Irish, Welsh, Scottish nature/mythology)
  - Celestial (space, astronomy, futuristic tech)
- **Cross-culture blending** for unique hybrid names
- **Industry-optimized generation** for 11 industry profiles

### Advanced Linguistic Features

- **Phonaesthetics (sound symbolism)** - Generate names that "sound like" their meaning
- **Research-backed phonaesthetic scoring** - Evaluate name beauty based on latest linguistics research
- **Brand archetypes** - Power, elegance, speed, nature, tech, trust, innovation
- **Cross-linguistic hazard checking** - Avoid names that sound vulgar in other languages
- **Pronounceability filtering** - Hard gate based on Sonority Sequencing Principle
- **Memorability scoring** - Comprehensive analysis of name quality
- **Syllable/stress analysis** - Trochaic, iambic, dactylic patterns
- **Rhythm analysis** - Metrical phonology for optimal brand name cadence
- **Competitive differentiation** - Ensure distinctiveness from competitors

### Validation

- **EU trademark search** via EUIPO API
- **US trademark search** via RapidAPI/USPTO
- **Nice classification filtering** (predefined profiles or custom classes)
- **Domain availability** (.com, .de, .eu, .io, .co)
- **Phonetic similarity detection** (Soundex, Metaphone, Levenshtein)
- **Known brand conflict checking**

### Database Management

- SQLite storage for generated names
- Status workflow tracking
- Class-specific trademark conflict storage

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd brands

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with your API credentials:

```bash
# EUIPO API (https://dev.euipo.europa.eu/)
EUIPO_CLIENT_ID=your_client_id
EUIPO_CLIENT_SECRET=your_client_secret

# RapidAPI for USPTO (https://rapidapi.com/)
RAPIDAPI_KEY=your_rapidapi_key

# Anthropic Claude API (optional, for LLM generation)
ANTHROPIC_API_KEY=your_anthropic_key
```

## Quick Start

### Command Line

```bash
# Generate 10 names (rule-based)
python -m brandkit generate -n 10

# Generate with cultural methods
python -m brandkit generate -n 10 --method japanese
python -m brandkit generate -n 10 --method nordic
python -m brandkit generate -n 10 --method latin

# Check a name with full validation
python -m brandkit check "Voltix" --full --profile camping_rv

# List available Nice class profiles
python -m brandkit profiles

# List saved names
python -m brandkit list --status new
```

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
print(f"Feel: {analysis['feel']}")

# Check trademark availability
result = kit.check("Voltix", nice_classes="camping_rv")
if result['available']:
    kit.save(name, status="candidate")

# Filter out names similar to competitors
filtered = kit.filter_competitors(names, ["Tesla", "Volta", "Enphase"])
```

## Generation Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `rule_based` | Phonetic rules, CV patterns | General purpose |
| `markov` | Statistical character generation | Realistic names |
| `llm` | Claude-powered creative generation | Creative/conceptual |
| `greek` | Greek mythology roots | Power, tech |
| `turkic` | Turkic language harmony | Automotive (VW style) |
| `nordic` | Norse mythology, Scandinavian | Outdoor, strength |
| `japanese` | CV patterns, minimalist | Tech, gaming |
| `latin` | Classical/Romance elegance | Luxury, pharma |
| `celtic` | Nature, mythology | Craft, heritage |
| `celestial` | Space/astronomy roots | Tech, aerospace, energy |
| `blend` | Cross-culture fusion | Unique names |

## Industry Profiles

Generate names optimized for specific industries:

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

```python
# Generate industry-specific names
names = kit.generate_for_industry("wellness", count=20)
```

## Brand Archetypes

Score and generate names for brand personalities:

- **power** - Strong consonants (k, t, v, x), short vowels
- **elegance** - Soft consonants (l, s, f), flowing sounds
- **speed** - Sharp sounds (z, x, v), quick rhythm
- **nature** - Soft consonants, earthy vowels
- **tech** - Crisp consonants (x, k, z), modern feel
- **trust** - Solid consonants (b, d, m), stable rhythm
- **innovation** - Unusual combinations, distinctive sounds

```python
# Score name for archetype fit
score = kit.score_for_archetype("Voltix", "power")
print(f"Power fit: {score:.2f}")
```

## Cross-Linguistic Hazard Checking

Avoid embarrassing international mistakes:

```python
result = kit.check_hazards("Gift")
# Returns: HAZARD (means "poison" in German)

result = kit.check_hazards("Mist")
# Returns: HAZARD (means "manure" in German)
```

The hazard checker detects:
- Exact word hazards in multiple languages
- Sound-alike problems (e.g., names that sound vulgar)
- Phonetic patterns to avoid
- Cultural/religious sensitivities

## Phonaesthetic Scoring

Research-backed evaluation of how pleasant a name sounds, based on:

- **David Crystal's research**: Beautiful words have consonants like l, m, s, n
- **Nemestothy et al. (2024)**: Front vowels (e, i) rated more pleasant
- **Yorkston & Menon (2004)**: Sound symbolism affects brand perception
- **Metrical phonology**: Trochaic rhythm (STRONG-weak) preferred in English/German

```python
from brandkit.generators.phonemes import phonaesthetic_score, is_pronounceable

# Check if name passes pronounceability gate
is_ok, reason = is_pronounceable("Resafafn")
# Returns: (False, "unpronounceable_ending:fn")

# Get detailed phonaesthetic breakdown
result = phonaesthetic_score("Lumina", category="luxury")
print(f"Score: {result['score']:.2f}")  # 0.61
print(f"Quality: {result['quality']}")  # "acceptable"

# Component scores:
# - consonant_score: Pleasant sounds (l,m,n,s,r) vs harsh
# - vowel_score: Front vowels preferred for aesthetics
# - fluency_score: CV balance, mild repetition
# - rhythm_score: Trochaic 2-syllable = ideal
# - category_fit_score: Match to industry (tech, luxury, power, nature, speed)
```

Score thresholds:
- **excellent** (≥0.85): Exceptionally pleasant sounding
- **good** (≥0.70): Pleasant, memorable
- **acceptable** (≥0.50): Neutral, workable
- **poor** (<0.50): Needs improvement

## Nice Class Profiles

Predefined product category profiles for trademark searches:

| Profile | Classes | Description |
|---------|---------|-------------|
| `camping_rv` | 7, 9, 11, 12 | Camping equipment, RV/caravan accessories |
| `power_electronics` | 7, 9 | DC/DC converters, inverters, power supplies |
| `electronics` | 9 | Electrical/electronic apparatus |
| `software` | 9, 42 | Computer software and services |
| `energy` | 7, 9, 11 | Power generation, electrical systems |
| `automotive` | 7, 9, 12 | Vehicles, automotive parts |
| `clothing` | 25, 35 | Clothing, footwear, headgear |
| `household` | 11, 21 | Household appliances |
| `food_beverage` | 29, 30, 32, 33 | Food products and beverages |
| `retail` | 35 | Retail services, advertising |
| `consulting` | 35, 42 | Business and technical consulting |

## Project Structure

```
brands/
├── brandkit/                    # Main package
│   ├── __init__.py              # BrandKit class, public API
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Nice class profiles, configuration
│   ├── generators/              # Name generation modules
│   │   ├── __init__.py          # Generator exports
│   │   ├── base_generator.py    # CulturalGenerator base class
│   │   ├── cultural_generators.py # Japanese, Latin, Celtic, Celestial
│   │   ├── greek_generator.py   # Greek mythology generator
│   │   ├── turkic_generator.py  # Turkic language generator
│   │   ├── nordic_generator.py  # Nordic/Scandinavian generator
│   │   └── phonemes/            # YAML phoneme configurations
│   │       ├── greek.yaml
│   │       ├── turkic.yaml
│   │       ├── nordic.yaml
│   │       ├── japanese.yaml
│   │       ├── latin.yaml
│   │       ├── celtic.yaml
│   │       ├── celestial.yaml
│   │       ├── phonaesthemes.yaml  # Sound symbolism
│   │       ├── hazards.yaml        # Cross-linguistic hazards
│   │       ├── industries.yaml     # Industry profiles
│   │       ├── base_generator.yaml # Cultural generator tuning
│   │       └── strategies.yaml     # Phonotactics, rhythm, phonaesthetic scoring
│   ├── checkers/                # Validation modules
│   └── db/                      # Database modules
├── .env                         # API credentials (create this)
├── brandnames.db                # SQLite database (auto-created)
└── README.md
```

## CLI Reference

```bash
brandkit --version                    # Show version
brandkit --help                       # Show all commands
```

### Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `generate` | `gen`, `g` | Generate brand names |
| `check` | `c` | Check name availability |
| `hazards` | `haz`, `h` | Check cross-linguistic hazards |
| `score` | `s` | Get phonaesthetic score |
| `stats` | | Show database statistics |
| `list` | `ls`, `l` | List names from database |
| `export` | | Export database to JSON |
| `save` | | Save a name to database |
| `block` | | Block a name |
| `profiles` | | List Nice class profiles |
| `industries` | | List available industries |
| `discover` | `disc`, `d` | Automated discovery pipeline |
| `tm-conflicts` | | Show stored trademark matches |
| `tm-risk` | | Show high/critical trademark risk matches |
| `reset` | | Reset database |

### Examples

```bash
# Generate names
brandkit generate -n 10 --method japanese
brandkit gen -n 20 -m blend --save
brandkit generate --industry tech

# Check availability
brandkit check "Voltix" --full --profile camping_rv
brandkit c "Voltix" --classes 9,12

# Phonaesthetic scoring
brandkit score "Lumina" --category luxury -v

# Hazard checking
brandkit hazards "Gift" --markets german,french

# Database operations
brandkit list --status candidate --limit 20
brandkit list --quality excellent --json
brandkit stats
brandkit save "NewName" --status candidate
brandkit block "BadName" --reason "negative_meaning"

# Discovery pipeline (generate -> filter -> validate -> save)
brandkit discover -n 100 --method blend            # Single batch of 100
brandkit discover --target 50 --min-quality good   # Loop until 50 good names found
brandkit discover --target 100 -q excellent -m japanese  # 100 excellent Japanese names
brandkit discover --target 10 --parallel           # Adaptive batching (starts with 50, adjusts)
brandkit discover -n 50 --parallel --max-concurrent 20  # Fixed batch with more concurrency
brandkit discover -n 50 --profiling                # Profile to identify bottlenecks
brandkit discover -n 50 --profiling --profile-output profile.json  # Save profiling data

# Parallel mode shows a live terminal UI with:
# - Progress bar and metrics (target/found/round/time)
# - Per-stage progress with checkmarks
# - Per-worker status with animated spinners
# - TLD availability: .com ✓ .de ✓ .eu ⋯ .io ○
# - Live results feed
```

## License

GPL-3.0
