# Changelog

All notable changes to Brandkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- [Codex] Base generator tuning in `brandkit/generators/phonemes/base_generator.yaml` (memorability, hazards, blending, competitive differentiation).
- [Codex] Phonetic similarity tuning in `strategies.yaml` (cache size and Soundex/Metaphone partial match scores).
- [Codex] Configurable defaults for generation mix/blend cultures, similarity top-matches, and RapidAPI max results in `app.yaml`.
- [Claude] Comprehensive test suite (127 tests):
  - `test_discovery_pipeline.py` - RateLimiter, RetryHandler, parallel checkers, E2E pipeline
  - `test_hazard_checker.py` - German/English hazards, syllable-aware detection, severity levels
  - `test_similarity_checker.py` - Soundex, Metaphone, Cologne Phonetics, SimilarityMatch
  - `test_phonetics_en_de.py` - German normalization, G2P, market-specific pronounceability
  - `test_network_timeout.py` - DNS/socket timeouts, retry behavior, connection errors
- [Claude] Multi-agent collaboration rules in CLAUDE.md for Claude/Codex cooperation
- [Claude] AGENTS.md symlink to CLAUDE.md for cross-agent instruction sharing
- Test suite for DB schema/migrations and cache behavior:
  - Verifies trademark match risk fields in schema and reads/writes
  - Ensures EUIPO cache separation by Nice classes
  - Preserves RapidAPI Nice class data through cache for filtering
  - Validates domain cache keys by TLD set
- Post-generation quality filter with EN/DE pronounceability gating, hazard screening,
  known-brand similarity checks, and diversity constraints (`brandkit/quality.py`)
- Variant-query mode for trademark checks (EUIPO/RapidAPI) to broaden phonetic coverage
- Cologne Phonetics (German) in phonetic similarity scoring
- CLI command `tm-risk` to report blocking (HIGH/CRITICAL) trademark matches
- Phonetic similarity scoring module (`brandkit/phonetic_similarity.py`):
  - Soundex algorithm (USPTO standard)
  - Double Metaphone for pronunciation-based matching
  - Normalized Levenshtein for visual similarity
  - Combined weighted scoring (Soundex 35%, Metaphone 40%, Levenshtein 25%)
  - Risk level calculation based on trademark status (CRITICAL/HIGH/MEDIUM/LOW)
- EU/German trademark law compliance module (`brandkit/eu_trademark_analysis.py`):
  - German phonetic equivalents (ei/ai, v/f, sch/sh, umlauts)
  - Related Nice classes (Dienstleistungsähnlichkeit) mapping
  - Well-known marks (Bekannte Marken) database for Art. 8(5) EUTMR
  - Conceptual similarity (begriffliche Ähnlichkeit) checking
  - Comprehensive EU trademark conflict analysis
- Database methods for risk assessment:
  - `recalculate_trademark_risks()` - batch update existing matches
  - `update_trademark_match_risk()` - update individual match
  - `get_high_risk_matches()` - query CRITICAL/HIGH risk conflicts
- `risk_level` and `phonetic_similarity` columns in trademark_matches table
- Trademark match storage for conflict review:
  - New `trademark_matches` table stores individual conflicting marks
  - Stores: match name, serial number, Nice classes, status, similarity score
  - Methods: `save_trademark_match()`, `save_trademark_matches_batch()`, `get_trademark_matches()`, `clear_trademark_matches()`
- Memorability scoring based on cognitive psychology research:
  - Optimal length (4-7 characters) bonus
  - Syllable count factor (2-3 syllables optimal)
  - Strong initial consonant detection
  - Alliteration and assonance bonuses
- CLI: Excel export now includes sub-scores and pronounceability:
  - Consonant, Vowel, Fluency, Rhythm, Natural, Memorability columns
  - Pronounceability check column with detailed failure reasons
  - Color coding: pronounceability issues highlighted in orange
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
- [Codex] Cultural generation, memorability, hazard gating, blending, and competitive differentiation now read from YAML (no hard-coded thresholds).
- [Codex] Phonetic similarity cache sizing and partial-match scoring now YAML-driven.
- [Codex] “all” method list, blend default cultures, similarity top-match limit, and RapidAPI max-results are config-driven.
- [Codex] Domain/EUIPO/RapidAPI checker cache hash length and timeouts are now YAML-required.
- [Codex] Discovery UI result feed length now configurable via app.yaml.
- [Codex] Tightened diversity constraints and penalties to reduce repetitive prefixes/suffixes.
- [Codex] Moved cache directories to repo-local `data/cache` paths to avoid permission errors.
- [Codex] Tightened quality gate thresholds and expanded rule-based syllable/ending variety to reduce homogeneity.
- Database schema cleaned up - removed 7 legacy columns:
  - Removed: `score`, `score_de`, `score_en`, `score_euphony`, `euipo_checked`, `euipo_matches`, `euipo_url`
  - Added: `score_cluster_quality`, `score_ending_quality` (previously calculated but not stored)
  - All phonaesthetic sub-scores now stored and exported
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
- `discover` now defaults to `--method all` (instead of `blend`)
- CLI now saves the actual per-name method when available (e.g., greek/nordic),
  not just the CLI `--method` value
- Rule-based/LLM generation now explicitly stamp a `method` value for DB accuracy
- Cultural generators now hard-gate EN/DE pronounceability during generation
- Trademark checks now treat high phonetic similarity as blocking (US/EU risk policy)
- Trademark risk assessment now considers class overlap when assigning risk levels
- Trademark collision thresholds and status risk mapping now load from `strategies.yaml`
- Quality filter similarity threshold now loads from `strategies.yaml`

### Fixed
- [Codex] Corrected invalid f-string escape in `brandkit/parallel.py`.
- [Codex] Fixed indentation error in `brandkit/cli.py` that broke CLI execution.
- [Codex] Expanded English sexual/profanity hazards and phonetic patterns to block names like “Orgesex” and “Pinis”.
- [Codex] Blocked existing DB entries flagged for sexual/profanity hazards.
- Trademark match storage schema now includes `is_exact`, `risk_level`, and `phonetic_similarity`
  - Migrations add missing columns for existing databases
- EUIPO cache now keys by Nice classes and environment to avoid stale class-filtered results
- RapidAPI cache now persists Nice classes so class filtering remains correct on cache hits
- Domain cache now keys by TLD set to prevent cross-TLD reuse
- Added explicit request timeouts for EUIPO/RapidAPI connections
- Domain checker restores global socket timeout after DNS checks
- `list` command crash when block_reason contains custom pronounceability strings
- `reset` command now uses correct database path (`data/brandnames.db`)
- YAML boolean parsing bug: unquoted `on` in suffix lists was parsed as `True`
  - Fixed by quoting `"on"` in celestial.yaml and adding defensive filtering
- Quality threshold miscalibration: old threshold (0.605) was unreachable
  - 0% of generated names passed "excellent" threshold
  - Recalibrated using empirical brand name corpus

### TODO
- Silence pytest-asyncio deprecation warning by setting `asyncio_default_fixture_loop_scope` in `pyproject.toml`
- Add tests for network timeout handling and retry behavior in trademark/domain checkers
- Add end-to-end tests for the discovery pipeline with mocked external APIs

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
