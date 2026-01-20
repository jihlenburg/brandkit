#!/usr/bin/env python3
"""
Integrated Brand Name Generator
================================
Combines multiple generation methods:
1. Rule-based phonetic generation
2. Markov chain generation with semantic seeding
3. EUIPO trademark checking

Usage:
    python generate.py --count 30 --method hybrid
    python generate.py --count 20 --method markov --temperature 0.8
    python generate.py --count 10 --check-euipo
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


def load_dotenv():
    """Load .env file if it exists (no external dependency needed)"""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())


# Load environment variables from .env
load_dotenv()

# Import our modules
from brand_generator import BrandNameGenerator, NameScorer, NameScore
from markov_generator import HybridMarkovGenerator, MarkovGenerator, MarkovTrainer, get_weighted_corpus
from euipo_checker import EUIPOChecker, TrademarkResult
from rapidapi_checker import RapidAPIChecker
from namedb import BrandNameDB, NameStatus, BlockReason, get_namedb


@dataclass
class GeneratedName:
    """A generated brand name with all metadata"""
    name: str
    score: float
    method: str  # 'rules', 'markov', 'hybrid'

    # Detailed scores
    pronounceability_de: float = 0.0
    pronounceability_en: float = 0.0
    memorability: float = 0.0
    euphony: float = 0.0
    rhythm: float = 0.0
    semantic_fit: float = 0.0
    uniqueness: float = 0.0

    # Semantic info
    associations: list = None
    generation_details: str = ""

    # EUIPO check results
    euipo_checked: bool = False
    euipo_found: bool = False
    euipo_exact_matches: int = 0
    euipo_url: str = ""

    # USPTO/WIPO check results (via RapidAPI)
    uspto_checked: bool = False
    uspto_found: bool = False
    uspto_exact_matches: int = 0

    def __post_init__(self):
        if self.associations is None:
            self.associations = []

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'score': round(self.score, 3),
            'method': self.method,
            'scores': {
                'pronounceability_de': round(self.pronounceability_de, 3),
                'pronounceability_en': round(self.pronounceability_en, 3),
                'memorability': round(self.memorability, 3),
                'euphony': round(self.euphony, 3),
                'rhythm': round(self.rhythm, 3),
                'semantic_fit': round(self.semantic_fit, 3),
                'uniqueness': round(self.uniqueness, 3),
            },
            'associations': self.associations,
            'generation_details': self.generation_details,
            'euipo': {
                'checked': self.euipo_checked,
                'found': self.euipo_found,
                'exact_matches': self.euipo_exact_matches,
                'search_url': self.euipo_url,
            } if self.euipo_checked else None,
            'uspto': {
                'checked': self.uspto_checked,
                'found': self.uspto_found,
                'exact_matches': self.uspto_exact_matches,
            } if self.uspto_checked else None
        }


class IntegratedGenerator:
    """
    Main generator combining all methods.

    Generation Methods:
    - 'rules': Pure rule-based phonetic generation
    - 'markov': Markov chain with semantic seeding
    - 'hybrid': Mix of both methods (recommended)
    """

    def __init__(self,
                 target_hardness: float = 0.4,
                 temperature: float = 0.9,
                 categories: list = None,
                 use_blocklist: bool = True):
        """
        Initialize the integrated generator.

        Args:
            target_hardness: Sound hardness for rule-based (0=soft, 1=hard)
            temperature: Creativity for Markov (0.5=safe, 1.5=creative)
            categories: Semantic categories to focus on
            use_blocklist: Filter out blocked names (default: True)
        """
        self.target_hardness = target_hardness
        self.temperature = temperature
        self.categories = categories or ['energy', 'travel', 'nature']
        self.use_blocklist = use_blocklist
        self.namedb = get_namedb() if use_blocklist else None

        # Initialize components
        self.rule_generator = BrandNameGenerator(
            target_hardness=target_hardness,
            prefer_semantic=True
        )
        self.scorer = NameScorer()

        # Lazy-load Markov generator (takes a moment to train)
        self._markov_generator = None

    @property
    def markov_generator(self) -> HybridMarkovGenerator:
        """Lazy-load Markov generator"""
        if self._markov_generator is None:
            print("Training Markov models...", file=sys.stderr)
            self._markov_generator = HybridMarkovGenerator()
        return self._markov_generator

    def generate_rules(self, count: int, min_score: float = 0.6) -> list[GeneratedName]:
        """Generate names using rule-based approach"""
        results = []
        raw_results = self.rule_generator.generate_batch(
            count=count,
            min_score=min_score,
            categories=self.categories
        )

        for score_result in raw_results:
            results.append(GeneratedName(
                name=score_result.name,
                score=score_result.total_score,
                method='rules',
                pronounceability_de=score_result.pronounceability_de,
                pronounceability_en=score_result.pronounceability_en,
                memorability=score_result.memorability,
                euphony=score_result.euphony,
                rhythm=score_result.rhythm,
                semantic_fit=score_result.semantic_fit,
                uniqueness=score_result.uniqueness,
                associations=score_result.semantic_associations,
            ))

        return results

    def generate_markov(self, count: int, min_score: float = 0.6) -> list[GeneratedName]:
        """Generate names using Markov chains"""
        results = []

        # Map our categories to markov categories
        markov_categories = []
        for cat in self.categories:
            if cat in ['energy', 'transformation']:
                markov_categories.append('energy')
            elif cat in ['travel']:
                markov_categories.append('travel')
            elif cat in ['nature']:
                markov_categories.append('nature')
            elif cat in ['tech']:
                markov_categories.append('tech')

        raw_results = self.markov_generator.generate(
            count=count * 2,  # Generate more, then filter by score
            temperature=self.temperature,
            categories=markov_categories if markov_categories else None
        )

        for name, method_detail in raw_results:
            # Score the name
            score_result = self.scorer.score_name(name, [])

            if score_result.total_score >= min_score and not score_result.issues:
                results.append(GeneratedName(
                    name=name,
                    score=score_result.total_score,
                    method='markov',
                    pronounceability_de=score_result.pronounceability_de,
                    pronounceability_en=score_result.pronounceability_en,
                    memorability=score_result.memorability,
                    euphony=score_result.euphony,
                    rhythm=score_result.rhythm,
                    semantic_fit=score_result.semantic_fit,
                    uniqueness=score_result.uniqueness,
                    generation_details=method_detail,
                ))

            if len(results) >= count:
                break

        return results

    def generate_hybrid(self, count: int, min_score: float = 0.6) -> list[GeneratedName]:
        """Generate names using both methods"""
        # Split between methods
        rule_count = count // 2
        markov_count = count - rule_count

        rule_results = self.generate_rules(rule_count, min_score)
        markov_results = self.generate_markov(markov_count, min_score)

        # Combine and deduplicate
        seen = set()
        results = []

        for item in rule_results + markov_results:
            if item.name.lower() not in seen:
                seen.add(item.name.lower())
                results.append(item)

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:count]

    def generate(self,
                 count: int = 30,
                 method: str = 'hybrid',
                 min_score: float = 0.6) -> list[GeneratedName]:
        """
        Generate brand names.

        Args:
            count: Number of names to generate
            method: 'rules', 'markov', or 'hybrid'
            min_score: Minimum score threshold

        Returns:
            List of GeneratedName objects
        """
        # Generate more than needed to account for blocklist filtering
        generate_count = count
        if self.use_blocklist and self.namedb.count_blocked() > 0:
            generate_count = int(count * 1.3)  # Generate 30% extra

        if method == 'rules':
            results = self.generate_rules(generate_count, min_score)
        elif method == 'markov':
            results = self.generate_markov(generate_count, min_score)
        else:  # hybrid
            results = self.generate_hybrid(generate_count, min_score)

        # Filter out blocked names
        if self.use_blocklist:
            results = [r for r in results if not self.namedb.is_blocked(r.name)]

        return results[:count]

    def check_euipo(self,
                    names: list[GeneratedName],
                    api_key: str = None,
                    api_secret: str = None) -> list[GeneratedName]:
        """
        Check names against EUIPO trademark database.

        Args:
            names: List of GeneratedName objects
            api_key: EUIPO API key (optional, generates URLs if not provided)
            api_secret: EUIPO API secret

        Returns:
            Names with EUIPO check results added
        """
        checker = EUIPOChecker(
            client_id=api_key,
            client_secret=api_secret
        )

        for item in names:
            result = checker.check(item.name)
            item.euipo_checked = True
            item.euipo_found = result.found
            item.euipo_exact_matches = result.exact_matches
            item.euipo_url = result.search_url

        return names

    def check_uspto(self,
                    names: list[GeneratedName],
                    api_key: str = None) -> list[GeneratedName]:
        """
        Check names against USPTO/WIPO via RapidAPI.

        Args:
            names: List of GeneratedName objects
            api_key: RapidAPI key (optional, uses env var if not provided)

        Returns:
            Names with USPTO check results added
        """
        checker = RapidAPIChecker(api_key=api_key)

        if not checker.has_api_access:
            print("Warning: No RapidAPI key. Skipping USPTO check.", file=sys.stderr)
            print("Get free key: https://rapidapi.com/Creativesdev/api/trademark-lookup-api", file=sys.stderr)
            return names

        for item in names:
            result = checker.check(item.name)
            item.uspto_checked = True
            item.uspto_found = result.found
            item.uspto_exact_matches = result.exact_matches

        return names


def main():
    parser = argparse.ArgumentParser(
        description='Generate phonetically pleasing brand names for DE/EN markets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --count 30 --method hybrid
  %(prog)s --count 20 --method markov --temperature 0.8
  %(prog)s --count 15 --check-euipo --min-score 0.75
  %(prog)s --categories energy travel --output names.json
        """
    )

    # Generation options
    parser.add_argument(
        '-n', '--count',
        type=int,
        default=30,
        help='Number of names to generate (default: 30)'
    )
    parser.add_argument(
        '-m', '--method',
        choices=['rules', 'markov', 'hybrid'],
        default='hybrid',
        help='Generation method (default: hybrid)'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.65,
        help='Minimum score threshold (0-1, default: 0.65)'
    )
    parser.add_argument(
        '-c', '--categories',
        nargs='+',
        choices=['energy', 'travel', 'nature', 'tech', 'transformation'],
        default=['energy', 'travel', 'nature'],
        help='Semantic categories to focus on'
    )

    # Sound options
    parser.add_argument(
        '--hardness',
        type=float,
        default=0.4,
        help='Sound hardness for rules (0=soft, 1=hard, default: 0.4)'
    )
    parser.add_argument(
        '-t', '--temperature',
        type=float,
        default=0.9,
        help='Creativity for Markov (0.5=safe, 1.5=creative, default: 0.9)'
    )

    # EUIPO options
    parser.add_argument(
        '--check-euipo',
        action='store_true',
        help='Check names against EUIPO trademark database'
    )
    parser.add_argument(
        '--euipo-key',
        type=str,
        help='EUIPO API Client ID (or set EUIPO_CLIENT_ID env var)'
    )
    parser.add_argument(
        '--euipo-secret',
        type=str,
        help='EUIPO API Client Secret (or set EUIPO_CLIENT_SECRET env var)'
    )

    # USPTO/RapidAPI options
    parser.add_argument(
        '--check-uspto',
        action='store_true',
        help='Check names against USPTO/WIPO via RapidAPI (free: 1000/month)'
    )
    parser.add_argument(
        '--rapidapi-key',
        type=str,
        help='RapidAPI key (or set RAPIDAPI_KEY env var)'
    )
    parser.add_argument(
        '--check-all',
        action='store_true',
        help='Check both EUIPO and USPTO databases'
    )

    # Blocklist options
    parser.add_argument(
        '--no-blocklist',
        action='store_true',
        help='Disable blocklist filtering'
    )
    parser.add_argument(
        '--block',
        nargs='+',
        metavar='NAME',
        help='Add name(s) to blocklist (e.g., --block Voltex Fluxon)'
    )
    parser.add_argument(
        '--block-reason',
        choices=['euipo_conflict', 'euipo_similar', 'phonetic_issue_de',
                 'phonetic_issue_en', 'negative_connotation', 'client_rejected',
                 'legal_issue', 'other'],
        default='other',
        help='Reason for blocking (default: other)'
    )
    parser.add_argument(
        '--blocklist-stats',
        action='store_true',
        help='Show database statistics and exit'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save generated names to database'
    )
    parser.add_argument(
        '--save-status',
        choices=['new', 'candidate'],
        default='new',
        help='Status for saved names (default: new)'
    )

    # Output options
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['text', 'json', 'csv', 'markdown'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed scoring information'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Handle blocklist commands first
    namedb = get_namedb()

    if args.blocklist_stats:
        stats = namedb.stats()
        print(f"\nDatabase Statistics:")
        print("=" * 40)
        print(f"Total names: {stats['total']}")
        print(f"Blocked: {stats['by_status'].get('blocked', 0)}")
        print(f"\nBy status:")
        for status, count in stats['by_status'].items():
            print(f"  {status}: {count}")
        if stats['by_block_reason']:
            print(f"\nBy block reason:")
            for reason, count in stats['by_block_reason'].items():
                print(f"  {reason}: {count}")
        return

    if args.block:
        reason = BlockReason(args.block_reason)
        for name in args.block:
            if namedb.block(name, reason):
                print(f"✓ Blocked: {name} ({reason.value})")
            else:
                print(f"⚠ Already blocked: {name}")
        return

    # Set random seed if provided
    if args.seed is not None:
        import random
        random.seed(args.seed)

    # Initialize generator
    generator = IntegratedGenerator(
        target_hardness=args.hardness,
        temperature=args.temperature,
        categories=args.categories,
        use_blocklist=not args.no_blocklist
    )

    # Print header
    print(f"Brand Name Generator", file=sys.stderr)
    blocked_count = namedb.count_blocked()
    if blocked_count > 0 and not args.no_blocklist:
        print(f"Blocked: {blocked_count} names filtered", file=sys.stderr)
    print(f"=" * 60, file=sys.stderr)
    print(f"Method: {args.method}", file=sys.stderr)
    print(f"Categories: {', '.join(args.categories)}", file=sys.stderr)
    print(f"Minimum score: {args.min_score}", file=sys.stderr)
    if args.method in ['markov', 'hybrid']:
        print(f"Temperature: {args.temperature}", file=sys.stderr)
    if args.method in ['rules', 'hybrid']:
        print(f"Sound hardness: {args.hardness}", file=sys.stderr)
    print(f"-" * 60, file=sys.stderr)

    # Generate names
    print(f"Generating {args.count} names...", file=sys.stderr)
    results = generator.generate(
        count=args.count,
        method=args.method,
        min_score=args.min_score
    )

    # Check EUIPO if requested
    if args.check_euipo or args.check_all:
        print(f"Checking EUIPO database...", file=sys.stderr)
        results = generator.check_euipo(
            results,
            api_key=args.euipo_key,
            api_secret=args.euipo_secret
        )

    # Check USPTO/WIPO if requested
    if args.check_uspto or args.check_all:
        print(f"Checking USPTO/WIPO database...", file=sys.stderr)
        results = generator.check_uspto(
            results,
            api_key=args.rapidapi_key
        )

    # Format output
    if args.format == 'json':
        output = json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False)

    elif args.format == 'csv':
        lines = ['name,score,method,pronounce_de,pronounce_en,memorability,euphony,euipo_found,euipo_url']
        for r in results:
            euipo_found = str(r.euipo_found) if r.euipo_checked else ''
            euipo_url = r.euipo_url if r.euipo_checked else ''
            lines.append(
                f'{r.name},{r.score:.3f},{r.method},{r.pronounceability_de:.3f},'
                f'{r.pronounceability_en:.3f},{r.memorability:.3f},{r.euphony:.3f},'
                f'{euipo_found},{euipo_url}'
            )
        output = '\n'.join(lines)

    elif args.format == 'markdown':
        lines = ['# Generated Brand Names\n']
        lines.append('| # | Name | Score | Method | Details |')
        lines.append('|---|------|-------|--------|---------|')
        for i, r in enumerate(results, 1):
            details = r.generation_details or ', '.join(r.associations) or '-'
            euipo = ""
            if r.euipo_checked:
                if r.euipo_exact_matches > 0:
                    euipo = f" ⚠️ EUIPO: {r.euipo_exact_matches} exact"
                else:
                    euipo = " ✓"
            lines.append(f'| {i} | **{r.name}** | {r.score:.2f} | {r.method} | {details}{euipo} |')

        if args.check_euipo:
            lines.append('\n## EUIPO Check URLs\n')
            for r in results:
                if r.euipo_checked:
                    status = "⚠️" if r.euipo_exact_matches > 0 else "✓"
                    lines.append(f'- {status} [{r.name}]({r.euipo_url})')

        output = '\n'.join(lines)

    else:  # text
        lines = []
        for i, r in enumerate(results, 1):
            if args.verbose:
                lines.append(f"\n{i:2}. {r.name:<15} Score: {r.score:.2f} [{r.method}]")
                lines.append(f"    DE: {r.pronounceability_de:.2f}  EN: {r.pronounceability_en:.2f}  "
                           f"Memory: {r.memorability:.2f}  Euphony: {r.euphony:.2f}")
                if r.generation_details:
                    lines.append(f"    Method: {r.generation_details}")
                if r.associations:
                    lines.append(f"    Associations: {', '.join(r.associations)}")
                if r.euipo_checked:
                    if r.euipo_exact_matches > 0:
                        lines.append(f"    ⚠️  EUIPO: {r.euipo_exact_matches} exact matches found!")
                    else:
                        lines.append(f"    ✓  EUIPO: No exact matches")
                    lines.append(f"    Check: {r.euipo_url}")
                if r.uspto_checked:
                    if r.uspto_exact_matches > 0:
                        lines.append(f"    ⚠️  USPTO: {r.uspto_exact_matches} exact matches found!")
                    else:
                        lines.append(f"    ✓  USPTO: No exact matches")
            else:
                details = r.generation_details or ', '.join(r.associations) or ''
                details_str = f" ({details})" if details else ""

                # Trademark check indicators
                tm_str = ""
                if r.euipo_checked:
                    if r.euipo_exact_matches > 0:
                        tm_str += f" ⚠️EU:{r.euipo_exact_matches}"
                    else:
                        tm_str += " ✓EU"
                if r.uspto_checked:
                    if r.uspto_exact_matches > 0:
                        tm_str += f" ⚠️US:{r.uspto_exact_matches}"
                    else:
                        tm_str += " ✓US"

                lines.append(f"{i:2}. {r.name:<15} {r.score:.2f} [{r.method}]{details_str}{tm_str}")

        output = '\n'.join(lines)

    # Output
    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f"\nResults saved to: {args.output}", file=sys.stderr)
    else:
        print(output)

    # Save to database if requested
    if args.save:
        status = NameStatus.NEW if args.save_status == 'new' else NameStatus.CANDIDATE
        saved_count = 0
        skipped_count = 0

        for r in results:
            # Check if name already exists
            existing = namedb.get(r.name)
            if existing:
                skipped_count += 1
                continue

            # Add to database
            associations = ', '.join(r.associations) if r.associations else None
            namedb.add(
                name=r.name,
                score=r.score,
                score_de=r.pronounceability_de,
                score_en=r.pronounceability_en,
                score_memorability=r.memorability,
                score_euphony=r.euphony,
                method=r.method,
                semantic_associations=associations,
                generation_details=r.generation_details,
                status=status
            )

            # Add method as tag
            namedb.add_tag(r.name, r.method)

            # Store EUIPO results if checked
            if r.euipo_checked:
                namedb.update(
                    r.name,
                    euipo_checked=True,
                    euipo_matches=r.euipo_exact_matches,
                    euipo_url=r.euipo_url
                )

            saved_count += 1

        print(f"\n💾 Saved {saved_count} names to database (status: {status.value})", file=sys.stderr)
        if skipped_count > 0:
            print(f"   Skipped {skipped_count} already existing", file=sys.stderr)

    # Summary
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Generated {len(results)} names with score >= {args.min_score}", file=sys.stderr)

    if args.check_euipo or args.check_all:
        conflicts = sum(1 for r in results if r.euipo_exact_matches > 0)
        if conflicts:
            print(f"⚠️  {conflicts} names have potential EUIPO conflicts", file=sys.stderr)
        else:
            print(f"✓  No exact EUIPO matches found", file=sys.stderr)

    if args.check_uspto or args.check_all:
        conflicts = sum(1 for r in results if r.uspto_exact_matches > 0)
        if conflicts:
            print(f"⚠️  {conflicts} names have potential USPTO conflicts", file=sys.stderr)
        else:
            print(f"✓  No exact USPTO matches found", file=sys.stderr)


if __name__ == '__main__':
    main()
