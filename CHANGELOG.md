# Changelog

All notable changes to Brandkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Versioning workflow documentation in CLAUDE.md
- CLAUDE.md now tracked in git for consistent development guidelines

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
