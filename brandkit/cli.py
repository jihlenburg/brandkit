#!/usr/bin/env python3
"""
BrandKit CLI
============
Command-line interface for brand name generation and validation.

Usage:
    python -m brandkit generate -n 10 --method rule_based
    python -m brandkit check "Voltix"
    python -m brandkit list --status candidate
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from brandkit import BrandKit, get_config, list_profiles, NICE_PROFILES


def _get_name_score(name) -> float:
    """Get score from different name types (NameScore, TurkicName, etc.)."""
    if hasattr(name, 'total_score'):
        return name.total_score
    elif hasattr(name, 'score'):
        return name.score
    return 0.5


def _get_name_str(name) -> str:
    """Get name string from different name types."""
    if hasattr(name, 'name'):
        return name.name
    return str(name)


def cmd_generate(args, kit: BrandKit):
    """Generate brand names."""
    print(f"Generating {args.count} names using {args.method}...")
    print()

    try:
        names = kit.generate(
            count=args.count,
            method=args.method,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if args.method == 'rule_based':
        for i, name in enumerate(names, 1):
            print(f"{i:2}. {name.name:<15} [{name.total_score:.2f}]")
            if args.verbose:
                print(f"    DE:{name.pronounceability_de:.2f} EN:{name.pronounceability_en:.2f}")
                print(f"    Memo:{name.memorability:.2f} Euphony:{name.euphony:.2f}")
                if name.issues:
                    print(f"    Issues: {', '.join(name.issues)}")
                print()
    elif args.method == 'markov':
        for i, name in enumerate(names, 1):
            print(f"{i:2}. {name}")
    elif args.method == 'llm':
        for i, name in enumerate(names, 1):
            print(f"{i:2}. {name.name:<15} [{name.score_estimate:.2f}]")
            if args.verbose:
                print(f"    {name.explanation}")
                print(f"    Associations: {', '.join(name.associations)}")
                print()
    elif args.method in ('turkic', 'greek'):
        for i, name in enumerate(names, 1):
            print(f"{i:2}. {name.name:<15} [{name.score:.2f}]")
            if args.verbose:
                print(f"    Method: {name.method}")
                if name.roots_used:
                    print(f"    Roots: {', '.join(name.roots_used)}")
                if name.meaning_hints:
                    print(f"    Hints: {', '.join(name.meaning_hints)}")
                print()
    elif args.method == 'all':
        # Mixed output - handle different name types
        for i, name in enumerate(names, 1):
            if hasattr(name, 'name'):
                name_str = name.name
                score = _get_name_score(name)
            else:
                name_str = str(name)
                score = 0.5
            print(f"{i:2}. {name_str:<15} [{score:.2f}]")

    return 0


def cmd_profiles(args):
    """List available Nice class profiles."""
    print("Available Nice Class Profiles")
    print("=" * 60)
    print()
    for name, info in sorted(NICE_PROFILES.items()):
        classes_str = ', '.join(str(c) for c in info['classes'])
        print(f"  {name:<18} Classes: [{classes_str}]")
        print(f"  {'':<18} {info['description']}")
        print()
    return 0


def cmd_check(args, kit: BrandKit):
    """Check a brand name."""
    name = args.name

    # Resolve Nice classes from profile or direct list
    nice_classes = None
    if hasattr(args, 'profile') and args.profile:
        nice_classes = args.profile
    elif hasattr(args, 'classes') and args.classes:
        nice_classes = [int(c) for c in args.classes.split(',')]

    # Show what we're checking
    print(f"Checking: {name}")
    if nice_classes:
        if isinstance(nice_classes, str):
            profile_info = NICE_PROFILES.get(nice_classes, {})
            classes_str = ', '.join(str(c) for c in profile_info.get('classes', []))
            print(f"Profile: {nice_classes} (Classes: {classes_str})")
        else:
            print(f"Classes: {', '.join(str(c) for c in nice_classes)}")
    else:
        print("Classes: all (no filter)")
    print("=" * 50)

    result = kit.check(name, check_all=args.full, nice_classes=nice_classes)

    # Similarity
    sim = result['similarity']
    print(f"\nSimilarity Check: {'SAFE' if sim.is_safe else 'WARNING'}")
    if sim.similar_brands:
        for m in sim.similar_brands[:3]:
            flags = []
            if m.soundex_match:
                flags.append("Soundex")
            if m.metaphone_match:
                flags.append("Metaphone")
            flags.append(f"Sim:{m.text_similarity:.2f}")
            print(f"  - {m.known_brand}: [{', '.join(flags)}]")

    # Domain
    if 'domain' in result:
        dom = result['domain']
        print(f"\nDomain Check:")
        for tld, d in dom.domains.items():
            status = "AVAILABLE" if d.available else "taken"
            if d.has_website:
                status += " (has website)"
            print(f"  {d.domain}: {status}")

    # Trademark
    if 'trademark' in result:
        tm = result['trademark']
        if tm.get('euipo'):
            eu = tm['euipo']
            total = eu.exact_matches + eu.similar_matches
            print(f"\nEU Trademark: {total} matches ({eu.exact_matches} exact, {eu.similar_matches} similar)")
        if tm.get('uspto'):
            us = tm['uspto']
            total = us.exact_matches + us.similar_matches
            print(f"US Trademark: {total} matches ({us.exact_matches} exact, {us.similar_matches} similar)")
        if tm.get('euipo_error'):
            print(f"EU Trademark Error: {tm['euipo_error']}")
        if tm.get('uspto_error'):
            print(f"US Trademark Error: {tm['uspto_error']}")

    # Summary
    print(f"\n{'='*50}")
    if result['warnings']:
        print("Warnings:")
        for w in result['warnings']:
            print(f"  - {w}")
    else:
        print("No warnings.")

    return 0


def cmd_reset(args):
    """Reset database by deleting the database file."""
    from pathlib import Path

    db_path = Path(__file__).parent.parent / 'brandnames.db'

    if not db_path.exists():
        print("Database does not exist. Nothing to reset.")
        return 0

    if not args.force:
        response = input(f"Delete {db_path}? This cannot be undone. [y/N] ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0

    db_path.unlink()
    print("Database reset. A fresh database will be created on next use.")
    return 0


def cmd_discover(args, kit: BrandKit):
    """Automated pipeline: generate, check, and save viable names."""
    import time
    from brandkit import SimilarityChecker, DomainChecker
    from brandkit.db import NameStatus

    sim_checker = SimilarityChecker()
    domain_checker = DomainChecker()

    profile = args.profile or "camping_rv"
    profile_info = NICE_PROFILES.get(profile, {})
    classes_str = ', '.join(str(c) for c in profile_info.get('classes', []))

    print(f"BrandKit Discovery Pipeline")
    print(f"=" * 60)
    print(f"Generate: {args.count} names using {args.method}")
    print(f"Profile:  {profile} (Classes: {classes_str})")
    print(f"Top:      Check top {args.top} candidates for trademarks")
    print(f"=" * 60)

    # Stage 1: Generate
    print(f"\n[1/4] Generating {args.count} names...")
    names = kit.generate(count=args.count, method=args.method)
    print(f"      Generated {len(names)} names")

    # Stage 2: Similarity filter
    print(f"\n[2/4] Checking similarity to known brands...")
    passed_similarity = []
    for name in names:
        name_str = _get_name_str(name)
        result = sim_checker.check(name_str)
        if result.is_safe:
            passed_similarity.append(name)

    failed_sim = len(names) - len(passed_similarity)
    print(f"      Passed: {len(passed_similarity)}, Failed: {failed_sim}")

    if not passed_similarity:
        print("\nNo names passed similarity check.")
        return 0

    # Stage 3: Domain filter
    print(f"\n[3/4] Checking domain availability...")
    candidates = []
    for i, name in enumerate(passed_similarity):
        name_str = _get_name_str(name)
        result = domain_checker.check(name_str)
        available = [tld for tld, d in result.domains.items() if d.available]
        if available:
            candidates.append((name, available))
        if (i + 1) % 25 == 0:
            print(f"      Checked {i + 1}/{len(passed_similarity)}...")
        time.sleep(0.1)

    no_domains = len(passed_similarity) - len(candidates)
    print(f"      With domains: {len(candidates)}, No domains: {no_domains}")

    if not candidates:
        print("\nNo names have available domains.")
        return 0

    # Stage 4: Trademark check on top candidates
    candidates.sort(key=lambda x: _get_name_score(x[0]), reverse=True)
    top_candidates = candidates[:args.top]

    print(f"\n[4/4] Checking trademarks for top {len(top_candidates)} candidates...")
    viable = []
    conflicts = []

    for i, (name, domains) in enumerate(top_candidates):
        name_str = _get_name_str(name)
        result = kit.check(name_str, check_all=True, nice_classes=profile)

        tm_clear = True
        tm_info = ""

        if 'trademark' in result:
            tm = result['trademark']
            eu_matches = us_matches = 0

            if tm.get('euipo') and tm['euipo'].found:
                eu_matches = tm['euipo'].exact_matches + tm['euipo'].similar_matches
            if tm.get('uspto') and tm['uspto'].found:
                us_matches = tm['uspto'].exact_matches + tm['uspto'].similar_matches

            if eu_matches > 0 or us_matches > 0:
                tm_clear = False
                tm_info = f"EU:{eu_matches} US:{us_matches}"

        if tm_clear:
            viable.append((name, domains))
            status = "CLEAR"
        else:
            conflicts.append((name_str, tm_info))
            status = f"CONFLICT ({tm_info})"

        print(f"      {i+1:2}. {name_str:<15} [{_get_name_score(name):.2f}] {status}")
        time.sleep(0.5)

    # Results
    print(f"\n" + "=" * 60)
    print(f"RESULTS")
    print(f"=" * 60)
    print(f"\nPipeline: {len(names)} -> {len(passed_similarity)} -> {len(candidates)} -> {len(viable)} viable")

    if viable:
        print(f"\nVIABLE CANDIDATES ({len(viable)}):")
        print(f"{'Name':<15} {'Score':>6}  Domains")
        print("-" * 50)
        for name, domains in viable:
            name_str = _get_name_str(name)
            com_flag = "*" if "com" in domains else " "
            print(f"{name_str:<15} [{_get_name_score(name):.2f}] {com_flag} {', '.join(domains)}")
        print("\n* = .com available")

        # Save to database
        print(f"\nSaving {len(viable)} names to database as 'candidate'...")
        saved = 0
        for name, domains in viable:
            name_str = _get_name_str(name)
            try:
                kit._db.add(name=name_str, score=_get_name_score(name),
                           status=NameStatus.CANDIDATE, method=args.method)
                saved += 1
            except Exception:
                pass  # Already exists
        print(f"Saved {saved} new names.")
    else:
        print("\nNo viable candidates found. Try generating more names.")

    # Summary
    total_candidates = len(kit._db.get_by_status(NameStatus.CANDIDATE))
    print(f"\nTotal candidates in database: {total_candidates}")

    return 0


def cmd_list(args, kit: BrandKit):
    """List names from database."""
    from brandkit.db import NameStatus

    if args.status:
        # Convert string to NameStatus enum
        try:
            status_enum = NameStatus(args.status)
            names = kit.db.get_by_status(status_enum, limit=args.limit)
        except ValueError:
            print(f"Error: Invalid status '{args.status}'")
            return 1
    else:
        # Get all by searching with empty query (returns recent)
        names = kit.db.search("")[:args.limit] if args.limit else kit.db.search("")

    if not names:
        print("No names found.")
        return 0

    print(f"{'Name':<20} {'Status':<12} {'Score':>6}")
    print("-" * 40)
    for name in names:
        print(f"{name.name:<20} {name.status.value:<12} {name.score:>6.2f}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='BrandKit - Brand Name Generator & Validator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate -n 10                    Generate 10 names
  %(prog)s generate -n 5 --method markov     Generate using Markov chains
  %(prog)s check "Voltix"                    Check a name (all classes)
  %(prog)s check "Voltix" --profile camping_rv   Check for camping/RV products
  %(prog)s check "Voltix" --classes 9,12     Check specific Nice classes
  %(prog)s profiles                          List available Nice class profiles
  %(prog)s list --status candidate           List database entries
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate brand names')
    gen_parser.add_argument('-n', '--count', type=int, default=10,
                           help='Number of names to generate')
    gen_parser.add_argument('--method', choices=['rule_based', 'markov', 'llm', 'hybrid', 'turkic', 'greek', 'all'],
                           default='rule_based', help='Generation method (all = mix of all methods)')
    gen_parser.add_argument('-v', '--verbose', action='store_true',
                           help='Show detailed scores')

    # Check command
    check_parser = subparsers.add_parser('check', help='Check a brand name')
    check_parser.add_argument('name', help='Brand name to check')
    check_parser.add_argument('--full', action='store_true',
                             help='Full check (trademark + domain)')
    check_parser.add_argument('--profile', '-p', type=str,
                             choices=list(NICE_PROFILES.keys()),
                             help='Nice class profile (e.g., camping_rv, electronics)')
    check_parser.add_argument('--classes', '-c', type=str,
                             help='Comma-separated Nice class numbers (e.g., 9,12)')

    # Profiles command
    subparsers.add_parser('profiles', help='List available Nice class profiles')

    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset database (delete all data)')
    reset_parser.add_argument('--force', '-f', action='store_true',
                             help='Skip confirmation prompt')

    # List command
    list_parser = subparsers.add_parser('list', help='List database entries')
    list_parser.add_argument('--status', choices=['new', 'candidate', 'shortlist',
                                                  'approved', 'rejected', 'blocked'],
                            help='Filter by status')
    list_parser.add_argument('--limit', type=int, default=50, help='Max results')

    # Discover command (automated pipeline)
    disc_parser = subparsers.add_parser('discover',
                                        help='Automated pipeline: generate, check, save viable names')
    disc_parser.add_argument('-n', '--count', type=int, default=100,
                            help='Number of names to generate (default: 100)')
    disc_parser.add_argument('--method', choices=['rule_based', 'markov', 'hybrid', 'turkic', 'greek', 'all'],
                            default='rule_based', help='Generation method (all = mix of all methods)')
    disc_parser.add_argument('--profile', '-p', type=str,
                            choices=list(NICE_PROFILES.keys()), default='camping_rv',
                            help='Nice class profile (default: camping_rv)')
    disc_parser.add_argument('--top', '-t', type=int, default=30,
                            help='Number of top candidates to trademark check (default: 30)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Initialize BrandKit
    kit = BrandKit()

    # Check config
    cfg = get_config()
    if args.command == 'generate' and args.method == 'llm' and not cfg.has_anthropic:
        print("Error: LLM generation requires ANTHROPIC_API_KEY in .env")
        return 1

    # Execute command
    if args.command == 'profiles':
        return cmd_profiles(args)
    elif args.command == 'reset':
        return cmd_reset(args)
    elif args.command == 'generate':
        return cmd_generate(args, kit)
    elif args.command == 'check':
        return cmd_check(args, kit)
    elif args.command == 'list':
        return cmd_list(args, kit)
    elif args.command == 'discover':
        return cmd_discover(args, kit)

    return 0


if __name__ == '__main__':
    sys.exit(main())
