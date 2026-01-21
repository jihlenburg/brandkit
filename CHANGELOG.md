# Changelog

All notable changes to Brandkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Versioning workflow documentation in CLAUDE.md
- CLAUDE.md now tracked in git for consistent development guidelines
- CLI: All generation methods now available (nordic, japanese, latin, celtic, celestial, blend)
- CLI: New commands: `hazards`, `score`, `stats`, `export`, `save`, `block`, `industries`
- CLI: Command aliases for convenience (`g`, `gen`, `c`, `s`, `ls`, `l`, `d`, `disc`, `haz`, `h`)
- CLI: `--version` / `-V` flag
- CLI: `--quiet` / `-q` flag for scripting
- CLI: `--industry` flag for industry-specific generation
- CLI: `--archetype` flag for brand archetype targeting
- CLI: `--save` flag to save generated names directly
- CLI: Input validation (empty names, special characters, length limits)
- CLI: JSON output option for `list` command (`--json`)
- CLI: Quality tier filtering (`--quality excellent/good/acceptable/poor`)
- CLI: Available/conflict filtering (`--available`, `--conflicts`)
- CLI: `discover --target N` loops until N valid names found
- CLI: `discover --min-quality` filters by quality threshold (excellent/good/acceptable)
- CLI: `discover --max-rounds` safety limit for target mode
- CLI: `discover` now includes pronounceability + quality filtering as step 2/5
- CLI: `discover --profiling` enables performance profiling with detailed bottleneck analysis
- CLI: `discover --profile-output FILE` saves profiling data to JSON for further analysis
- CLI: `discover --parallel` enables parallel domain/trademark checking (~4x speedup)
- CLI: `discover --max-concurrent N` controls max concurrent checks in parallel mode
- CLI: `discover --target N` now uses adaptive batch sizing (starts with 5×N, adjusts based on yield rate)
- CLI: `discover --target N --min-quality excellent/good` uses accumulate-then-validate mode:
  - Accumulates quality candidates before running expensive validation
  - More efficient for strict quality thresholds (excellent ~0.01%, good ~30-60% pass rate)
  - Shows progress: accumulated candidates, pass rate, total generated
- Rich-based live terminal UI for parallel discovery:
  - Progress bar with target/found/round metrics
  - Per-stage progress with checkmarks
  - Per-worker status lines with animated spinners
  - TLD availability visualization (✓/✗/⋯/○)
  - Live results feed
  - Automatic fallback to simple output for non-TTY
- Profiler module (`brandkit/profiler.py`) with context manager-based stage timing
- Parallel checking infrastructure (`brandkit/parallel.py`):
  - Parallel TLD lookups within DomainChecker (5x speedup per name)
  - Batch processing for multiple names concurrently
  - Interleaved pipeline (trademark checks start as domain results arrive)
  - Rate limiting for API protection
  - Retry logic with exponential backoff
- UI module (`brandkit/ui.py`) with Rich-based live display
- Entropy module (`brandkit/generators/entropy.py`) for enhanced randomness:
  - True random number generation using hardware entropy (os.urandom, secrets)
  - Phoneme mutation engine (voicing shifts, vowel shifts, lenition)
  - Expanded syllable structure templates (simple, complex, germanic, japanese)
  - Morphological operations (blending, infixation, reduplication, metathesis)
  - Cross-linguistic phoneme injection (Slavic, Germanic, Romance, Celtic)
- Calibration module (`brandkit/calibration.py`) for threshold validation:
  - Corpus of 248 validated brand names (excellent, problematic, neutral, pseudowords)
  - Statistical analysis with Cohen's d separation metrics
  - Cross-validation of threshold accuracy

### Changed
- Quality thresholds scientifically recalibrated using empirical data:
  - Corpus: 134 excellent brands (Interbrand Top 100), 21 problematic, 57 neutral
  - Finding: Cohen's d = 0.41 (moderate separation), 77% overlap between categories
  - New thresholds: excellent=0.585, good=0.570, acceptable=0.550, poor=0.520
  - Note: Score measures phonetic pleasantness, not brand success potential
- CLI now loads quality thresholds from strategies.yaml instead of hardcoding
- Generators updated to use true random entropy instead of Python's random module
- BlockReason now accepts dynamic strings (e.g., "pronounceability:awkward_start:sv")
- CLI completely refactored for cleaner code and better UX
- Database path fixed to use `data/brandnames.db`
- Output formatting improved with table display

### Fixed
- `list` command crash when block_reason contains custom pronounceability strings
- `reset` command now uses correct database path (`data/brandnames.db`)
- YAML boolean parsing bug: unquoted `on` in suffix lists was parsed as `True`
  - Fixed by quoting `"on"` in celestial.yaml and adding defensive filtering
- Quality threshold miscalibration: old threshold (0.605) was unreachable
  - 0% of generated names passed "excellent" threshold
  - Recalibrated using empirical brand name corpus

## [0.1.0] - 2026-01-20

### Added
- Multi-cultural phoneme-based name generation
  - Greek mythology roots and suffixes
  - Nordic/Scandinavian patterns
  - Turkic vowel harmony (VW/Nissan style)
  - Japanese-inspired phonetics
  - Latin/Romance morphemes
  - Celtic patterns
  - Celestial/space themes
- Phonaesthetic scoring system based on linguistic research
  - Consonant quality (Crystal's research)
  - Vowel quality (front/back vowels)
  - Processing fluency (CV balance)
  - Rhythm analysis (syllable weight, stress patterns)
  - Cluster quality penalties
  - Ending quality bonuses
- Pronounceability validation
  - Phonotactic constraints for English/German
  - Awkward initial cluster detection (sv-, stj-, gn-, etc.)
  - Chemical/pharmaceutical ending rejection (-ode, -ium, -ase)
  - Impossible internal cluster detection
  - Repetitive pattern filtering
- Trademark checking
  - USPTO via RapidAPI
  - EUIPO integration (pending approval)
  - Nice class filtering with industry profiles
- Domain availability checking (.com, .de, .eu, .io, .co)
- SQLite database for name management
  - Status tracking (new, candidate, blocked, etc.)
  - Quality tier classification
  - Trademark check history
  - Comments and tags
- Generation methods
  - Rule-based phonetic generation
  - Markov chain with semantic seeding
  - LLM-powered generation (Claude)
  - Culture blending
- Cross-linguistic hazard checking
- Industry-specific generation profiles
- CLI interface (`python -m brandkit`)

### Technical
- YAML-based phoneme configuration
- Lazy-loaded generators for performance
- Comprehensive scoring with configurable weights

[Unreleased]: https://github.com/jihlenburg/brandkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jihlenburg/brandkit/releases/tag/v0.1.0
