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
- Profiler module (`brandkit/profiler.py`) with context manager-based stage timing
- Parallel checking infrastructure (`brandkit/parallel.py`):
  - Parallel TLD lookups within DomainChecker (5x speedup per name)
  - Batch processing for multiple names concurrently
  - Interleaved pipeline (trademark checks start as domain results arrive)
  - Rate limiting for API protection
  - Retry logic with exponential backoff

### Changed
- BlockReason now accepts dynamic strings (e.g., "pronounceability:awkward_start:sv")
- CLI completely refactored for cleaner code and better UX
- Database path fixed to use `data/brandnames.db`
- Output formatting improved with table display

### Fixed
- `list` command crash when block_reason contains custom pronounceability strings
- `reset` command now uses correct database path (`data/brandnames.db`)

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
