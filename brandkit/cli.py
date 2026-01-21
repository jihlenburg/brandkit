#!/usr/bin/env python3
"""
Brandkit CLI
============
Command-line interface for brand name generation and validation.

Usage:
    brandkit generate -n 10 --method japanese
    brandkit check "Voltix" --profile camping_rv
    brandkit score "Lumina"
    brandkit stats
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Add parent to path
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from brandkit import __version__

# =============================================================================
# Constants
# =============================================================================

GENERATION_METHODS = [
    'rule_based', 'markov', 'llm', 'hybrid',
    'greek', 'turkic', 'nordic', 'japanese', 'latin', 'celtic', 'celestial',
    'blend', 'all'
]

INDUSTRIES = [
    'tech', 'automotive', 'pharma', 'luxury', 'food_beverage',
    'finance', 'energy', 'outdoor', 'wellness', 'gaming', 'ecommerce'
]

# =============================================================================
# Utilities
# =============================================================================

class Output:
    """Handles CLI output with quiet mode support."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet

    def print(self, *args, **kwargs):
        if not self.quiet:
            print(*args, **kwargs)

    def error(self, msg: str):
        print(f"Error: {msg}", file=sys.stderr)

    def success(self, msg: str):
        if not self.quiet:
            print(f"OK: {msg}")

    def table(self, headers: list, rows: list, col_widths: list = None):
        """Print a formatted table."""
        if self.quiet:
            return

        if not col_widths:
            col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
                         for i, h in enumerate(headers)]

        # Header
        header_line = ''.join(str(h).ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print('-' * len(header_line))

        # Rows
        for row in rows:
            print(''.join(str(c).ljust(w) for c, w in zip(row, col_widths)))


def validate_name(name: str) -> tuple[bool, str]:
    """Validate a brand name input."""
    if not name or not name.strip():
        return False, "Name cannot be empty"

    name = name.strip()

    if len(name) < 2:
        return False, "Name must be at least 2 characters"

    if len(name) > 30:
        return False, "Name must be at most 30 characters"

    if not re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', name):
        return False, "Name must start with a letter and contain only letters/numbers"

    return True, name


def get_name_str(name) -> str:
    """Extract name string from various name types."""
    if hasattr(name, 'name'):
        return name.name
    return str(name)


def get_name_score(name) -> float:
    """Extract score from various name types."""
    for attr in ('total_score', 'score', 'score_estimate'):
        if hasattr(name, attr):
            return getattr(name, attr)
    return 0.5


# =============================================================================
# Commands
# =============================================================================

def cmd_generate(args, out: Output):
    """Generate brand names."""
    from brandkit import BrandKit

    kit = BrandKit()

    out.print(f"Generating {args.count} names using '{args.method}'...")

    kwargs = {}
    if args.industry:
        kwargs['industry'] = args.industry
    if args.archetype:
        kwargs['archetype'] = args.archetype

    try:
        if args.industry and args.method not in ('blend', 'all'):
            # Use industry-specific generation
            names = kit.generate_for_industry(args.industry, count=args.count)
        else:
            names = kit.generate(count=args.count, method=args.method, **kwargs)
    except ValueError as e:
        out.error(str(e))
        return 1

    if not names:
        out.print("No names generated.")
        return 0

    out.print()

    # Format output based on verbosity
    rows = []
    for i, name in enumerate(names, 1):
        name_str = get_name_str(name)
        score = get_name_score(name)

        if args.verbose:
            method = getattr(name, 'method', args.method)
            roots = getattr(name, 'roots_used', [])
            roots_str = ', '.join(roots[:3]) if roots else '-'
            rows.append([i, name_str, f"{score:.2f}", method, roots_str])
        else:
            rows.append([i, name_str, f"{score:.2f}"])

    if args.verbose:
        out.table(['#', 'Name', 'Score', 'Method', 'Roots'], rows, [4, 18, 8, 12, 30])
    else:
        out.table(['#', 'Name', 'Score'], rows, [4, 18, 8])

    # Save if requested
    if args.save:
        saved = 0
        for name in names:
            try:
                kit.save(name, status='candidate', method=args.method)
                saved += 1
            except Exception:
                pass
        out.print(f"\nSaved {saved} names to database.")

    return 0


def cmd_check(args, out: Output):
    """Check a brand name for availability."""
    from brandkit import BrandKit, NICE_PROFILES

    # Validate name
    valid, result = validate_name(args.name)
    if not valid:
        out.error(result)
        return 1
    name = result

    kit = BrandKit()

    # Resolve Nice classes
    nice_classes = None
    if args.profile:
        nice_classes = args.profile
    elif args.classes:
        nice_classes = [int(c.strip()) for c in args.classes.split(',')]

    # Show what we're checking
    out.print(f"Checking: {name}")
    if nice_classes:
        if isinstance(nice_classes, str):
            info = NICE_PROFILES.get(nice_classes, {})
            classes_str = ', '.join(str(c) for c in info.get('classes', []))
            out.print(f"Profile: {nice_classes} (Classes: {classes_str})")
        else:
            out.print(f"Classes: {', '.join(str(c) for c in nice_classes)}")
    out.print("=" * 50)

    # Run check
    result = kit.check(name, check_all=args.full, nice_classes=nice_classes)

    # Similarity
    sim = result['similarity']
    status = "SAFE" if sim.is_safe else "WARNING"
    out.print(f"\nSimilarity: {status}")
    if sim.similar_brands:
        for m in sim.similar_brands[:3]:
            flags = []
            if m.soundex_match:
                flags.append("Soundex")
            if m.metaphone_match:
                flags.append("Metaphone")
            out.print(f"  - {m.known_brand} ({', '.join(flags)}, sim={m.text_similarity:.2f})")

    # Domain
    if 'domain' in result:
        out.print(f"\nDomains:")
        for tld, d in result['domain'].domains.items():
            status = "AVAILABLE" if d.available else "taken"
            out.print(f"  {d.domain}: {status}")

    # Trademark
    if 'trademark' in result:
        tm = result['trademark']
        out.print(f"\nTrademarks:")
        if tm.get('euipo'):
            eu = tm['euipo']
            total = eu.exact_matches + eu.similar_matches
            status = "CLEAR" if total == 0 else f"{total} conflicts"
            out.print(f"  EU: {status}")
        if tm.get('uspto'):
            us = tm['uspto']
            total = us.exact_matches + us.similar_matches
            status = "CLEAR" if total == 0 else f"{total} conflicts"
            out.print(f"  US: {status}")
        if tm.get('euipo_error'):
            out.print(f"  EU Error: {tm['euipo_error']}")
        if tm.get('uspto_error'):
            out.print(f"  US Error: {tm['uspto_error']}")

    # Summary
    out.print(f"\n{'=' * 50}")
    if result['available']:
        out.print("AVAILABLE - No conflicts found")
    else:
        out.print("CONFLICTS FOUND:")
        for w in result['warnings']:
            out.print(f"  - {w}")

    return 0 if result['available'] else 1


def cmd_hazards(args, out: Output):
    """Check a name for cross-linguistic hazards."""
    from brandkit import BrandKit

    valid, result = validate_name(args.name)
    if not valid:
        out.error(result)
        return 1
    name = result

    kit = BrandKit()

    markets = args.markets.split(',') if args.markets else None
    result = kit.check_hazards(name, markets=markets)

    out.print(f"Hazard Check: {name}")
    out.print("=" * 50)

    if result.is_safe:
        out.print("SAFE - No hazards found")
    else:
        out.print(f"WARNING - Severity: {result.severity}")
        out.print("\nIssues found:")
        for issue in result.issues:
            out.print(f"  - [{issue.market}] {issue.type}: {issue.description}")

    return 0 if result.is_safe else 1


def cmd_score(args, out: Output):
    """Get phonaesthetic score for a name."""
    from brandkit.generators.phonemes import phonaesthetic_score, is_pronounceable, analyze_rhythm

    valid, result = validate_name(args.name)
    if not valid:
        out.error(result)
        return 1
    name = result

    out.print(f"Scoring: {name}")
    out.print("=" * 50)

    # Pronounceability check
    is_ok, reason = is_pronounceable(name)
    if not is_ok:
        out.print(f"\nPronounceability: FAIL ({reason})")
        out.print("Score: 0.00 (unpronounceable)")
        return 1

    out.print(f"\nPronounceability: PASS")

    # Get detailed score
    result = phonaesthetic_score(name, category=args.category)

    out.print(f"\nOverall Score: {result['score']:.3f} ({result['quality']})")
    out.print(f"\nComponent Scores:")
    out.print(f"  Consonant quality: {result['consonant_score']:.3f}")
    out.print(f"  Vowel quality:     {result['vowel_score']:.3f}")
    out.print(f"  Fluency:           {result['fluency_score']:.3f}")
    out.print(f"  Naturalness:       {result['naturalness_score']:.3f}")
    out.print(f"  Rhythm:            {result['rhythm_score']:.3f}")
    out.print(f"  Cluster quality:   {result['cluster_quality_score']:.3f}")
    out.print(f"  Ending quality:    {result['ending_quality_score']:.3f}")

    if args.category and result.get('category_fit_score'):
        out.print(f"  Category fit ({args.category}): {result['category_fit_score']:.3f}")

    # Rhythm analysis
    if args.verbose:
        rhythm = analyze_rhythm(name)
        out.print(f"\nRhythm Analysis:")
        out.print(f"  Syllables: {' - '.join(rhythm['syllables'])}")
        out.print(f"  Pattern: {rhythm['weight_pattern']} ({rhythm['rhythm_type']})")
        out.print(f"  Stress: {rhythm['stress_pattern']}")

    return 0


def cmd_stats(args, out: Output):
    """Show database statistics."""
    from brandkit import get_db

    db = get_db()
    stats = db.stats()

    out.print("Database Statistics")
    out.print("=" * 50)
    out.print(f"\nTotal names: {stats['total']}")

    if stats['by_status']:
        out.print(f"\nBy Status:")
        for status, count in sorted(stats['by_status'].items()):
            out.print(f"  {status:<12}: {count:>5}")

    if stats['by_block_reason']:
        out.print(f"\nBy Block Reason:")
        for reason, count in sorted(stats['by_block_reason'].items(), key=lambda x: -x[1])[:10]:
            out.print(f"  {reason:<35}: {count:>5}")

    if stats.get('avg_score'):
        out.print(f"\nAverage Score: {stats['avg_score']:.3f}")
    if stats.get('top_score'):
        out.print(f"Top Score: {stats['top_score']:.3f}")

    return 0


def cmd_list(args, out: Output):
    """List names from database."""
    from brandkit import get_db
    from brandkit.db import NameStatus, QualityTier

    db = get_db()

    # Build query based on filters
    if args.status:
        try:
            status_enum = NameStatus(args.status)
            names = db.get_by_status(status_enum, limit=args.limit)
        except ValueError:
            out.error(f"Invalid status: {args.status}")
            return 1
    elif args.quality:
        try:
            tier = QualityTier(args.quality)
            names = db.get_by_quality_tier(tier, limit=args.limit)
        except ValueError:
            out.error(f"Invalid quality tier: {args.quality}")
            return 1
    elif args.available:
        names = db.get_available(limit=args.limit)
    elif args.conflicts:
        names = db.get_conflicts(limit=args.limit)
    else:
        names = db.get_candidates(limit=args.limit)

    if not names:
        out.print("No names found.")
        return 0

    # Format output
    if args.json:
        data = [n.to_dict() for n in names]
        print(json.dumps(data, indent=2))
    else:
        rows = []
        for n in names:
            score = n.score_phonaesthetic or n.score or 0
            quality = n.quality_tier.value if n.quality_tier else '-'
            rows.append([n.name, n.status.value, f"{score:.2f}", quality])

        out.table(['Name', 'Status', 'Score', 'Quality'], rows, [20, 12, 8, 12])
        out.print(f"\nTotal: {len(names)}")

    return 0


def cmd_export(args, out: Output):
    """Export database to JSON."""
    from brandkit import get_db

    db = get_db()

    output_path = args.output or 'brandkit_export.json'

    json_str = db.export_json(output_path)

    if args.output:
        out.success(f"Exported to {output_path}")
    else:
        print(json_str)

    return 0


def cmd_save(args, out: Output):
    """Save a name to the database."""
    from brandkit import BrandKit
    from brandkit.db import NameStatus

    valid, result = validate_name(args.name)
    if not valid:
        out.error(result)
        return 1
    name = result

    kit = BrandKit()

    try:
        status = NameStatus(args.status) if args.status else NameStatus.CANDIDATE
    except ValueError:
        out.error(f"Invalid status: {args.status}")
        return 1

    record = kit.save(name, status=status.value, method=args.method or 'manual')

    if record:
        out.success(f"Saved '{name}' as {status.value}")
        if record.score_phonaesthetic:
            out.print(f"  Score: {record.score_phonaesthetic:.3f} ({record.quality_tier.value if record.quality_tier else 'unrated'})")
    else:
        out.error(f"Failed to save '{name}'")
        return 1

    return 0


def cmd_block(args, out: Output):
    """Block a name."""
    from brandkit import get_db

    valid, result = validate_name(args.name)
    if not valid:
        out.error(result)
        return 1
    name = result

    db = get_db()

    reason = args.reason or 'manual_block'
    success = db.block(name, reason=reason, notes=args.notes)

    if success:
        out.success(f"Blocked '{name}' (reason: {reason})")
    else:
        out.error(f"Failed to block '{name}'")
        return 1

    return 0


def cmd_profiles(args, out: Output):
    """List available Nice class profiles."""
    from brandkit import NICE_PROFILES

    out.print("Nice Class Profiles")
    out.print("=" * 60)
    out.print()

    for name, info in sorted(NICE_PROFILES.items()):
        classes_str = ', '.join(str(c) for c in info['classes'])
        out.print(f"  {name:<18} [{classes_str}]")
        out.print(f"  {'':<18} {info['description']}")
        out.print()

    return 0


def cmd_industries(args, out: Output):
    """List available industries for generation."""
    from brandkit import BrandKit

    kit = BrandKit()
    industries = kit.list_industries()

    out.print("Available Industries")
    out.print("=" * 60)
    out.print()

    for industry in sorted(industries):
        profile = kit.get_industry_profile(industry)
        cultures = ', '.join(profile.get('cultural_sources', []))
        out.print(f"  {industry:<15} Cultures: {cultures}")

    return 0


def _discover_accumulate(args, out: Output, kit, sim_checker, domain_checker, profiler,
                         min_score: float, min_quality: str, nice_profile: str,
                         target_count: int, use_parallel: bool, max_concurrent: int):
    """
    Accumulate-then-validate discovery mode.

    For strict quality thresholds (excellent/good), this mode:
    1. Accumulates candidates until validation_batch_size is reached
    2. Validates the batch (similarity, domain, trademark)
    3. Repeats until target is met

    This is more efficient than round-based mode for rare quality levels.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor
    from brandkit.generators.phonemes import phonaesthetic_score, is_pronounceable
    from brandkit.ui import get_ui, RICH_AVAILABLE

    # Configuration
    GENERATION_CHUNK = 100  # Names per generation call
    VALIDATION_BATCH = 50   # Candidates to accumulate before validating
    MAX_GENERATION = 20000  # Safety limit for accumulation phase

    # Get UI with accumulate mode
    ui = get_ui(
        target=target_count,
        batch_size=VALIDATION_BATCH,
        method=args.method,
        profile=nice_profile,
        parallel=use_parallel,
        max_workers=max_concurrent,
        quiet=out.quiet or args.profiling,
        accumulate_mode=True
    )

    total_generated = 0
    total_saved = 0
    all_viable = []
    validation_round = 0
    validation_yield_history = []  # Track yield at validation stage

    with ui:
        while total_saved < target_count:
            validation_round += 1
            candidates = []  # (name, score, quality) tuples

            # === PHASE 1: ACCUMULATE CANDIDATES ===
            ui.start_accumulation(validation_round, VALIDATION_BATCH, min_quality)
            accumulation_generated = 0

            while len(candidates) < VALIDATION_BATCH:
                if accumulation_generated >= MAX_GENERATION:
                    ui.log(f"Generated {MAX_GENERATION} names without enough candidates. Stopping.", style="red")
                    break

                # Generate a chunk
                with profiler.stage("generation", items=GENERATION_CHUNK):
                    try:
                        names = kit.generate(count=GENERATION_CHUNK, method=args.method)
                    except Exception as e:
                        ui.log(f"Generation failed: {e}", style="red")
                        break

                accumulation_generated += len(names)
                total_generated += len(names)

                # Filter by pronounceability + quality
                with profiler.stage("quality_filter", items=len(names)):
                    for name in names:
                        name_str = get_name_str(name)

                        is_ok, reason = is_pronounceable(name_str)
                        if not is_ok:
                            continue

                        result = phonaesthetic_score(name_str)
                        if result['score'] >= min_score:
                            candidates.append((name, result['score'], result['quality']))

                # Update accumulation progress
                pass_rate = len(candidates) / accumulation_generated * 100 if accumulation_generated > 0 else 0
                ui.update_accumulation(
                    accumulated=len(candidates),
                    target=VALIDATION_BATCH,
                    generated=accumulation_generated,
                    pass_rate=pass_rate
                )

            if not candidates:
                ui.log("No candidates found. Try a different method or lower quality threshold.", style="red")
                break

            # === PHASE 2: VALIDATE CANDIDATES ===
            ui.start_validation(len(candidates))

            # Stage: Similarity filter
            sim_passed = []
            with profiler.stage("similarity_check", items=len(candidates)):
                for name, score, quality in candidates:
                    name_str = get_name_str(name)
                    result = sim_checker.check(name_str)
                    if result.is_safe:
                        sim_passed.append((name, score, quality))

            ui.update_validation_stage("similarity", passed=len(sim_passed), total=len(candidates))

            if not sim_passed:
                ui.log("All candidates failed similarity check.", style="yellow")
                continue

            # Stage: Domain filter
            domain_passed = []
            if use_parallel:
                with profiler.stage("domain_check", items=len(sim_passed)):
                    name_to_info = {get_name_str(n): (n, s, q) for n, s, q in sim_passed}
                    name_list = list(name_to_info.keys())

                    def check_domain(name_str):
                        result = domain_checker.check(name_str)
                        available = [tld for tld, d in result.domains.items() if d.available]
                        return (name_str, available)

                    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                        results = list(executor.map(check_domain, name_list))

                    for name_str, available in results:
                        if available:
                            name, score, quality = name_to_info[name_str]
                            domain_passed.append((name, score, quality, available))
            else:
                with profiler.stage("domain_check", items=len(sim_passed)):
                    for name, score, quality in sim_passed:
                        name_str = get_name_str(name)
                        result = domain_checker.check(name_str)
                        available = [tld for tld, d in result.domains.items() if d.available]
                        if available:
                            domain_passed.append((name, score, quality, available))
                        time.sleep(0.05)

            ui.update_validation_stage("domain", passed=len(domain_passed), total=len(sim_passed))

            if not domain_passed:
                ui.log("All candidates failed domain check.", style="yellow")
                continue

            # Stage: Trademark check
            domain_passed.sort(key=lambda x: x[1], reverse=True)
            viable = []

            if use_parallel:
                with profiler.stage("trademark_check", items=len(domain_passed)):
                    tm_candidates = []
                    for name, score, quality, domains in domain_passed:
                        name_str = get_name_str(name)
                        existing = kit.db.get(name_str)
                        if not existing:
                            tm_candidates.append((name, score, quality, domains, name_str))

                    tm_workers = min(5, max_concurrent // 2) or 1

                    def check_trademark(item):
                        name, score, quality, domains, name_str = item
                        try:
                            result = kit.check(name_str, check_all=True, nice_classes=nice_profile)
                            return (name, score, quality, domains, name_str, result.get('available', False))
                        except Exception:
                            return (name, score, quality, domains, name_str, False)

                    with ThreadPoolExecutor(max_workers=tm_workers) as executor:
                        results = list(executor.map(check_trademark, tm_candidates))

                    for name, score, quality, domains, name_str, is_available in results:
                        if total_saved + len(viable) >= target_count:
                            break
                        if is_available:
                            viable.append((name, score, quality, domains))
                        ui.add_result(name_str, success=is_available,
                                     detail=".com" if "com" in domains else "")
            else:
                with profiler.stage("trademark_check", items=len(domain_passed)):
                    for name, score, quality, domains in domain_passed:
                        if total_saved + len(viable) >= target_count:
                            break

                        name_str = get_name_str(name)
                        existing = kit.db.get(name_str)
                        if existing:
                            continue

                        result = kit.check(name_str, check_all=True, nice_classes=nice_profile)
                        is_available = result.get('available', False)
                        if is_available:
                            viable.append((name, score, quality, domains))

                        ui.add_result(name_str, success=is_available,
                                     detail=".com" if "com" in domains else "")
                        time.sleep(0.2)

            ui.update_validation_stage("trademark", passed=len(viable), total=len(domain_passed))

            # Save viable names
            if viable:
                with profiler.stage("database_save", items=len(viable)):
                    for name, score, quality, domains in viable:
                        name_str = get_name_str(name)
                        try:
                            kit.save(name, status='candidate', method=args.method)
                            total_saved += 1
                            all_viable.append((name_str, score, quality, domains))
                        except Exception:
                            pass

            # Track validation yield for adaptive batch sizing
            validation_yield_history.append((len(candidates), len(viable)))

            ui.update_progress(found=total_saved, generated=total_generated)

            # Adaptive validation batch size based on yield
            if validation_yield_history:
                total_validated = sum(c for c, v in validation_yield_history)
                total_viable = sum(v for c, v in validation_yield_history)
                if total_validated > 0 and total_viable > 0:
                    validation_yield = total_viable / total_validated
                    remaining = target_count - total_saved
                    # Adjust batch to likely yield remaining names
                    VALIDATION_BATCH = max(20, min(100, int(remaining / validation_yield * 1.5)))

        # Print summary
        ui.print_summary(all_viable)

    return total_saved, total_generated, all_viable


def cmd_discover(args, out: Output):
    """Automated pipeline: generate, check, and save viable names."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from brandkit import BrandKit, NICE_PROFILES, SimilarityChecker
    from domain_checker import DomainChecker
    from brandkit.db import NameStatus, QualityTier
    from brandkit.generators.phonemes import phonaesthetic_score, is_pronounceable
    from brandkit.profiler import DiscoveryProfiler
    from brandkit.ui import get_ui, RICH_AVAILABLE

    # Initialize profiler
    profiler = DiscoveryProfiler(enabled=getattr(args, 'profiling', False))
    profiler.start()

    kit = BrandKit()
    sim_checker = SimilarityChecker()

    # Initialize domain checker with parallel TLD checking if --parallel
    use_parallel = getattr(args, 'parallel', False)
    max_concurrent = getattr(args, 'max_concurrent', 10)
    domain_checker = DomainChecker(parallel=use_parallel)

    nice_profile = args.profile
    profile_info = NICE_PROFILES.get(nice_profile, {})
    classes_str = ', '.join(str(c) for c in profile_info.get('classes', []))

    # Quality threshold mapping - load from strategies.yaml
    from brandkit.generators.phonemes import load_strategies
    strategies = load_strategies()
    phonaesthetic_config = strategies.get_phonaesthetic_config()
    quality_thresholds = phonaesthetic_config.get('thresholds', {
        'excellent': 0.59,
        'good': 0.58,
        'acceptable': 0.55,
        'poor': 0.52,
    })
    min_quality = args.min_quality or 'acceptable'
    min_score = quality_thresholds.get(min_quality, 0.55)

    # Target mode vs batch mode
    target_mode = args.target is not None
    target_count = args.target or 0

    # Use accumulate mode for strict quality thresholds with target mode
    use_accumulate_mode = target_mode and min_quality in ('excellent', 'good')

    if use_accumulate_mode:
        out.print(f"Using accumulate-then-validate mode for '{min_quality}' quality...")
        total_saved, total_generated, all_viable = _discover_accumulate(
            args, out, kit, sim_checker, domain_checker, profiler,
            min_score, min_quality, nice_profile, target_count,
            use_parallel, max_concurrent
        )
        profiler.stop()
        if args.profiling:
            report = profiler.report()
            if report:
                out.print(report)
            profile_output = getattr(args, 'profile_output', None)
            if profile_output:
                profiler.save_json(profile_output)
                out.print(f"\nProfiling data saved to: {profile_output}")
        if total_saved < target_count:
            out.print(f"\nNote: Only found {total_saved}/{target_count} names.")
        return 0

    # Target mode vs batch mode
    target_mode = args.target is not None
    target_count = args.target or 0

    # Adaptive batch sizing
    explicit_count = args.count is not None
    if explicit_count:
        batch_size = args.count
    elif target_mode:
        # Start with 5x target, minimum 20, maximum 100
        batch_size = max(20, min(100, target_count * 5))
    else:
        # Default for single-batch mode
        batch_size = 100

    # Yield rate tracking for adaptive sizing
    yield_history = []  # List of (generated, viable) tuples
    MIN_BATCH = 10
    MAX_BATCH = 200

    # Check if we should use Rich UI (parallel mode + TTY + Rich available)
    use_ui = use_parallel and RICH_AVAILABLE and not args.profiling

    total_generated = 0
    total_saved = 0
    all_viable = []
    round_num = 0
    max_rounds = args.max_rounds or 50  # Safety limit

    # Get UI
    ui = get_ui(
        target=target_count,
        batch_size=batch_size,
        method=args.method,
        profile=nice_profile,
        parallel=use_parallel,
        max_workers=max_concurrent,
        quiet=out.quiet or args.profiling  # Disable UI when profiling
    )

    with ui:
        while True:
            round_num += 1
            ui.start_round(round_num)

            if target_mode and total_saved >= target_count:
                break

            if round_num > max_rounds:
                ui.log(f"Reached max rounds ({max_rounds}). Stopping.")
                break

            # Adaptive batch sizing for subsequent rounds (when not explicitly set)
            if round_num > 1 and not explicit_count and target_mode and yield_history:
                remaining = target_count - total_saved
                # Calculate yield rate from history (viable / generated)
                total_gen = sum(g for g, v in yield_history)
                total_viable = sum(v for g, v in yield_history)
                if total_gen > 0 and total_viable > 0:
                    yield_rate = total_viable / total_gen
                    # Generate enough to likely get remaining, with 50% buffer
                    needed = int(remaining / yield_rate * 1.5)
                    batch_size = max(MIN_BATCH, min(MAX_BATCH, needed))
                    ui.update_batch_size(batch_size)
                elif total_gen > 0 and total_viable == 0:
                    # Zero yield - increase batch size progressively
                    batch_size = min(MAX_BATCH, batch_size * 2)
                    ui.update_batch_size(batch_size)
                    if round_num == 3:
                        ui.log("Warning: 0% yield rate. Consider lowering --min-quality.", style="yellow")
                    if round_num >= 5 and batch_size >= MAX_BATCH:
                        ui.log("Stopping: 0% yield after 5 rounds at max batch size.", style="red")
                        break

            # Stage 1: Generate
            ui.update_stage("generation", total=batch_size)
            with profiler.stage("generation", items=batch_size):
                try:
                    names = kit.generate(count=batch_size, method=args.method)
                except Exception as e:
                    ui.log(f"Generation failed: {e}", style="red")
                    if not target_mode:
                        return 1
                    continue

            total_generated += len(names)
            ui.update_progress(generated=total_generated)
            ui.complete_stage("generation", passed=len(names), total=len(names))

            # Stage 2: Pronounceability + Quality filter
            ui.update_stage("quality", total=len(names))
            quality_passed = []
            with profiler.stage("quality_filter", items=len(names)):
                for name in names:
                    name_str = get_name_str(name)

                    with profiler.stage("pronounceability"):
                        is_ok, reason = is_pronounceable(name_str)
                    if not is_ok:
                        continue

                    with profiler.stage("phonaesthetic_score"):
                        result = phonaesthetic_score(name_str)
                    if result['score'] >= min_score:
                        quality_passed.append((name, result['score'], result['quality']))

            ui.complete_stage("quality", passed=len(quality_passed), total=len(names))

            if not quality_passed:
                # Track zero yield for adaptive sizing
                yield_history.append((len(names), 0))
                if not target_mode:
                    break
                continue

            # Stage 3: Similarity filter
            ui.update_stage("similarity", total=len(quality_passed))
            sim_passed = []
            with profiler.stage("similarity_check", items=len(quality_passed)):
                for name, score, quality in quality_passed:
                    name_str = get_name_str(name)
                    result = sim_checker.check(name_str)
                    if result.is_safe:
                        sim_passed.append((name, score, quality))

            ui.complete_stage("similarity", passed=len(sim_passed), total=len(quality_passed))

            if not sim_passed:
                # Track zero yield for adaptive sizing
                yield_history.append((len(names), 0))
                if not target_mode:
                    break
                continue

            # Stage 4: Domain filter (parallel or sequential)
            ui.update_stage("domain", total=len(sim_passed))
            domain_passed = []

            if use_parallel:
                # Parallel domain checking with UI updates
                with profiler.stage("domain_check", items=len(sim_passed)):
                    name_to_info = {get_name_str(n): (n, s, q) for n, s, q in sim_passed}
                    name_list = list(name_to_info.keys())

                    def check_domain_with_ui(args_tuple):
                        idx, name_str = args_tuple
                        ui.add_worker("domain", idx, name_str)
                        try:
                            result = domain_checker.check(name_str)
                            available = []
                            for tld in ['com', 'de', 'eu', 'io', 'co']:
                                if tld in result.domains:
                                    status = "ok" if result.domains[tld].available else "fail"
                                    ui.update_worker("domain", idx, tld=tld, tld_status=status)
                                    if result.domains[tld].available:
                                        available.append(tld)
                            success = len(available) > 0
                            ui.complete_worker("domain", idx, success=success,
                                             details=f"{len(available)} available")
                            return (name_str, result, available)
                        except Exception as e:
                            ui.complete_worker("domain", idx, success=False, details=str(e))
                            return (name_str, None, [])

                    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                        futures = list(executor.map(
                            check_domain_with_ui,
                            enumerate(name_list)
                        ))

                    for name_str, result, available in futures:
                        if available:
                            name, score, quality = name_to_info[name_str]
                            domain_passed.append((name, score, quality, available))
            else:
                # Sequential domain checking
                with profiler.stage("domain_check", items=len(sim_passed)):
                    for i, (name, score, quality) in enumerate(sim_passed):
                        name_str = get_name_str(name)
                        with profiler.stage("dns_lookup"):
                            result = domain_checker.check(name_str)
                        available = [tld for tld, d in result.domains.items() if d.available]
                        if available:
                            domain_passed.append((name, score, quality, available))
                        time.sleep(0.05)

            ui.complete_stage("domain", passed=len(domain_passed), total=len(sim_passed))

            if not domain_passed:
                # Track zero yield for adaptive sizing
                yield_history.append((len(names), 0))
                if not target_mode:
                    break
                continue

            # Stage 5: Trademark check (parallel or sequential)
            domain_passed.sort(key=lambda x: x[1], reverse=True)
            to_check = domain_passed[:args.top] if not target_mode else domain_passed

            ui.update_stage("trademark", total=len(to_check))
            viable = []

            if use_parallel:
                with profiler.stage("trademark_check", items=len(to_check)):
                    # Prepare candidates
                    candidates = []
                    for name, score, quality, domains in to_check:
                        name_str = get_name_str(name)
                        existing = kit.db.get(name_str)
                        if not existing:
                            candidates.append((name, score, quality, domains, name_str))

                    tm_workers = min(5, max_concurrent // 2) or 1

                    def check_tm_with_ui(args_tuple):
                        idx, (name, score, quality, domains, name_str) = args_tuple
                        ui.add_worker("trademark", idx, name_str)
                        ui.update_worker("trademark", idx, details="USPTO...")
                        try:
                            result = kit.check(name_str, check_all=True, nice_classes=nice_profile)
                            is_available = result.get('available', False)
                            detail = "CLEAR" if is_available else "conflict"
                            ui.complete_worker("trademark", idx, success=is_available, details=detail)
                            return (name, score, quality, domains, name_str, result)
                        except Exception as e:
                            ui.complete_worker("trademark", idx, success=False, details=str(e))
                            return (name, score, quality, domains, name_str, {'available': False})

                    with ThreadPoolExecutor(max_workers=tm_workers) as executor:
                        results = list(executor.map(
                            check_tm_with_ui,
                            enumerate(candidates)
                        ))

                    for name, score, quality, domains, name_str, result in results:
                        if target_mode and (total_saved + len(viable)) >= target_count:
                            break

                        is_available = result.get('available', False)
                        if is_available:
                            viable.append((name, score, quality, domains))

                        com_str = ".com" if "com" in domains else ""
                        ui.add_result(name_str, success=is_available, detail=com_str)
            else:
                with profiler.stage("trademark_check", items=len(to_check)):
                    for i, (name, score, quality, domains) in enumerate(to_check):
                        if target_mode and (total_saved + len(viable)) >= target_count:
                            break

                        name_str = get_name_str(name)
                        existing = kit.db.get(name_str)
                        if existing:
                            continue

                        with profiler.stage("api_call"):
                            result = kit.check(name_str, check_all=True, nice_classes=nice_profile)

                        is_available = result.get('available', False)
                        if is_available:
                            viable.append((name, score, quality, domains))

                        com_str = ".com" if "com" in domains else ""
                        ui.add_result(name_str, success=is_available, detail=com_str)
                        time.sleep(0.2)

            ui.complete_stage("trademark", passed=len(viable), total=len(to_check))

            # Save viable names
            round_viable = 0
            if viable:
                with profiler.stage("database_save", items=len(viable)):
                    for name, score, quality, domains in viable:
                        name_str = get_name_str(name)
                        try:
                            kit.save(name, status='candidate', method=args.method)
                            total_saved += 1
                            round_viable += 1
                            all_viable.append((name_str, score, quality, domains))
                        except Exception:
                            pass

                ui.update_progress(found=total_saved)

            # Track yield rate for adaptive batch sizing
            yield_history.append((len(names), round_viable))

            # Exit if not in target mode (single batch)
            if not target_mode:
                break

        # Print summary
        ui.print_summary(all_viable)

    # Stop profiler
    profiler.stop()

    # Print profiling report if enabled
    if args.profiling:
        report = profiler.report()
        if report:
            out.print(report)

        profile_output = getattr(args, 'profile_output', None)
        if profile_output:
            profiler.save_json(profile_output)
            out.print(f"\nProfiling data saved to: {profile_output}")

    if target_mode and total_saved < target_count:
        out.print(f"\nNote: Only found {total_saved}/{target_count} names. Try different method or lower quality threshold.")

    return 0


def cmd_reset(args, out: Output):
    """Reset database."""
    db_path = Path(__file__).parent.parent / 'data' / 'brandnames.db'

    if not db_path.exists():
        out.print("Database does not exist.")
        return 0

    if not args.force:
        response = input(f"Delete {db_path}? This cannot be undone. [y/N] ")
        if response.lower() != 'y':
            out.print("Cancelled.")
            return 0

    db_path.unlink()
    out.success("Database reset.")
    return 0


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog='brandkit',
        description='Brandkit - Brand Name Generator & Validator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate -n 10 --method japanese
  %(prog)s generate -n 20 --industry tech --save
  %(prog)s check "Voltix" --profile camping_rv --full
  %(prog)s score "Lumina" --category luxury -v
  %(prog)s hazards "Voltix" --markets german,french
  %(prog)s list --available --limit 20
  %(prog)s stats
  %(prog)s discover -n 50 --method blend --profile electronics
"""
    )

    parser.add_argument('--version', '-V', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress non-essential output')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # --- generate ---
    p = subparsers.add_parser('generate', aliases=['gen', 'g'], help='Generate brand names')
    p.add_argument('-n', '--count', type=int, default=10, help='Number of names (default: 10)')
    p.add_argument('--method', '-m', choices=GENERATION_METHODS, default='rule_based',
                   help='Generation method')
    p.add_argument('--industry', '-i', choices=INDUSTRIES, help='Generate for specific industry')
    p.add_argument('--archetype', '-a', help='Brand archetype (power, elegance, speed, etc.)')
    p.add_argument('--save', '-s', action='store_true', help='Save generated names to database')
    p.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')

    # --- check ---
    p = subparsers.add_parser('check', aliases=['c'], help='Check name availability')
    p.add_argument('name', help='Brand name to check')
    p.add_argument('--full', '-f', action='store_true', help='Full check (trademark + domain)')
    p.add_argument('--profile', '-p', help='Nice class profile')
    p.add_argument('--classes', '-c', help='Comma-separated Nice classes (e.g., 9,12)')

    # --- hazards ---
    p = subparsers.add_parser('hazards', aliases=['haz', 'h'], help='Check cross-linguistic hazards')
    p.add_argument('name', help='Brand name to check')
    p.add_argument('--markets', '-m', help='Comma-separated markets (e.g., german,french,spanish)')

    # --- score ---
    p = subparsers.add_parser('score', aliases=['s'], help='Get phonaesthetic score')
    p.add_argument('name', help='Brand name to score')
    p.add_argument('--category', '-c', help='Category for fit scoring (tech, luxury, power, etc.)')
    p.add_argument('--verbose', '-v', action='store_true', help='Show rhythm analysis')

    # --- stats ---
    subparsers.add_parser('stats', help='Show database statistics')

    # --- list ---
    p = subparsers.add_parser('list', aliases=['ls', 'l'], help='List names from database')
    p.add_argument('--status', choices=['new', 'candidate', 'shortlist', 'approved', 'rejected', 'blocked'])
    p.add_argument('--quality', choices=['excellent', 'good', 'acceptable', 'poor'])
    p.add_argument('--available', '-a', action='store_true', help='Show only available names')
    p.add_argument('--conflicts', action='store_true', help='Show only names with conflicts')
    p.add_argument('--limit', type=int, default=50, help='Max results (default: 50)')
    p.add_argument('--json', '-j', action='store_true', help='Output as JSON')

    # --- export ---
    p = subparsers.add_parser('export', help='Export database to JSON')
    p.add_argument('--output', '-o', help='Output file path')

    # --- save ---
    p = subparsers.add_parser('save', help='Save a name to database')
    p.add_argument('name', help='Brand name to save')
    p.add_argument('--status', default='candidate', help='Initial status (default: candidate)')
    p.add_argument('--method', help='Generation method used')

    # --- block ---
    p = subparsers.add_parser('block', help='Block a name')
    p.add_argument('name', help='Brand name to block')
    p.add_argument('--reason', '-r', help='Block reason')
    p.add_argument('--notes', '-n', help='Additional notes')

    # --- profiles ---
    subparsers.add_parser('profiles', help='List Nice class profiles')

    # --- industries ---
    subparsers.add_parser('industries', help='List available industries')

    # --- discover ---
    p = subparsers.add_parser('discover', aliases=['disc', 'd'], help='Automated discovery pipeline')
    p.add_argument('-n', '--count', type=int, default=None,
                   help='Names per batch (default: auto based on target, or 100)')
    p.add_argument('--method', '-m', choices=GENERATION_METHODS, default='blend',
                   help='Generation method (default: blend)')
    p.add_argument('--profile', '-p', default='camping_rv', help='Nice class profile')
    p.add_argument('--target', type=int, help='Target number of valid names (loops until reached)')
    p.add_argument('--min-quality', '-q', choices=['excellent', 'good', 'acceptable', 'poor'],
                   default='acceptable', help='Minimum quality threshold (default: acceptable)')
    p.add_argument('--max-rounds', type=int, default=50, help='Max generation rounds (default: 50)')
    p.add_argument('--top', '-t', type=int, default=30, help='Top candidates per batch (default: 30)')
    p.add_argument('--parallel', action='store_true', help='Enable parallel domain/trademark checking')
    p.add_argument('--max-concurrent', type=int, default=10, help='Max concurrent checks (default: 10)')
    p.add_argument('--profiling', action='store_true', help='Enable profiling to identify bottlenecks')
    p.add_argument('--profile-output', help='Save profiling data to JSON file')

    # --- reset ---
    p = subparsers.add_parser('reset', help='Reset database (delete all data)')
    p.add_argument('--force', '-f', action='store_true', help='Skip confirmation')

    # Parse
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Handle aliases
    cmd_map = {
        'gen': 'generate', 'g': 'generate',
        'c': 'check',
        'haz': 'hazards', 'h': 'hazards',
        's': 'score',
        'ls': 'list', 'l': 'list',
        'disc': 'discover', 'd': 'discover',
    }
    command = cmd_map.get(args.command, args.command)

    # Output handler
    out = Output(quiet=getattr(args, 'quiet', False))

    # Check API keys for certain commands
    if command == 'generate' and getattr(args, 'method', None) == 'llm':
        from brandkit import get_config
        if not get_config().has_anthropic:
            out.error("LLM generation requires ANTHROPIC_API_KEY in .env")
            return 1

    # Dispatch
    commands = {
        'generate': cmd_generate,
        'check': cmd_check,
        'hazards': cmd_hazards,
        'score': cmd_score,
        'stats': cmd_stats,
        'list': cmd_list,
        'export': cmd_export,
        'save': cmd_save,
        'block': cmd_block,
        'profiles': cmd_profiles,
        'industries': cmd_industries,
        'discover': cmd_discover,
        'reset': cmd_reset,
    }

    handler = commands.get(command)
    if handler:
        try:
            if command in ('profiles', 'industries', 'stats', 'reset', 'export'):
                return handler(args, out)
            else:
                return handler(args, out)
        except KeyboardInterrupt:
            out.print("\nCancelled.")
            return 130
        except Exception as e:
            out.error(str(e))
            if getattr(args, 'verbose', False):
                import traceback
                traceback.print_exc()
            return 1

    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
