#!/usr/bin/env python3
"""
Phonaesthetic Score Calibration Study
======================================
Rigorous calibration of quality thresholds using validated brand name corpora.

Methodology:
1. Curate gold standard corpus of excellent/problematic/neutral brand names
2. Score all names with phonaesthetic_score()
3. Analyze distributions and separation metrics
4. Calculate optimal thresholds using ROC analysis
5. Cross-validate results

Data Sources:
- Interbrand Best Global Brands (market-validated success)
- Known rebranded/failed names (market-validated failure)
- Academic pseudoword studies (controlled experiments)
- Common words as neutral baseline

References:
- Klink (2000): Sound symbolism in brand names
- Yorkston & Menon (2004): Sound symbolism effects on perception
- Crystal: Research on phonaesthetic qualities of words
"""

import statistics
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from brandkit.generators.phonemes import phonaesthetic_score, is_pronounceable


# =============================================================================
# Brand Name Corpus
# =============================================================================

# TIER 1: TOP GLOBAL BRANDS
# Sources: Interbrand Best Global Brands, Forbes Most Valuable Brands
# These are market-validated successful brand names
EXCELLENT_GLOBAL_BRANDS = [
    # Tech Giants
    "Apple", "Google", "Amazon", "Microsoft", "Samsung", "Intel", "Cisco",
    "Oracle", "SAP", "Adobe", "Salesforce", "IBM", "Dell", "HP", "Sony",

    # Consumer Tech
    "Netflix", "Spotify", "Tesla", "Uber", "Airbnb", "PayPal", "Zoom",
    "Slack", "Stripe", "Square", "Shopify", "Twitch", "Discord",

    # Automotive
    "Toyota", "Mercedes", "BMW", "Honda", "Ford", "Audi", "Porsche",
    "Ferrari", "Lexus", "Nissan", "Mazda", "Subaru", "Volvo", "Jaguar",

    # Luxury & Fashion
    "Chanel", "Hermes", "Gucci", "Prada", "Dior", "Cartier", "Tiffany",
    "Burberry", "Versace", "Armani", "Rolex", "Omega",

    # Consumer Goods
    "Nike", "Adidas", "Puma", "Lego", "Ikea", "Zara", "Uniqlo",

    # Food & Beverage
    "Coca", "Pepsi", "Nestle", "Danone", "Heineken", "Corona", "Budweiser",
    "Starbucks", "McDonalds", "KFC", "Subway", "Dominos",

    # Beauty & Personal Care
    "Loreal", "Nivea", "Dove", "Olay", "Lancome", "Clinique", "Estee",

    # Finance
    "Visa", "Mastercard", "Amex", "Chase", "Citi", "HSBC",

    # Other Successful Brands
    "Canon", "Nikon", "Philips", "Siemens", "Bosch", "Dyson",
    "Nintendo", "Sega", "Atari", "Kodak", "Xerox", "Polaroid",
]

# TIER 2: AWARD-WINNING / ACCLAIMED BRAND NAMES
# Names specifically praised for their linguistic qualities
EXCELLENT_LINGUISTIC = [
    # Tech startups with acclaimed names
    "Figma", "Notion", "Airtable", "Asana", "Trello", "Canva",
    "Dropbox", "Evernote", "Mailchimp", "Hubspot", "Zendesk",

    # Pharmaceutical brands (carefully crafted)
    "Viagra", "Prozac", "Lipitor", "Advil", "Tylenol", "Aleve",
    "Claritin", "Zyrtec", "Nexium", "Celebrex",

    # Invented names that became iconic
    "Kodak", "Xerox", "Verizon", "Accenture", "Novartis",
    "Swiffer", "Febreze",

    # Clean, modern tech names
    "Stripe", "Plaid", "Ripple", "Stellar", "Palantir",
    "Databricks", "Snowflake", "Confluent",
]

# TIER 3: PROBLEMATIC / FAILED / REBRANDED NAMES
# Names that failed, were rebranded, or have known issues
PROBLEMATIC_BRANDS = [
    # Rebranded due to name issues
    "Datsun",       # Became Nissan (better international appeal)
    "Andersen",     # Became Accenture (post-Enron scandal, but name was also dated)
    "Backrub",      # Became Google
    "Cadabra",      # Became Amazon (sounded like "cadaver")
    "Confinity",    # Became PayPal

    # Awkward phonetics in English
    "Huawei",       # Pronunciation confusion
    "Xiaomi",       # Pronunciation confusion
    "Hyundai",      # Often mispronounced

    # Unfortunately named (real products)
    "Ayds",         # Diet candy, unfortunate AIDS similarity
    "Pschitt",      # French soft drink
    "Siri",         # Means "buttocks" in Japanese
    "Lumia",        # Means "prostitute" in Spanish
    "Nova",         # "No va" = "doesn't go" in Spanish
    "Pinto",        # Slang in Brazilian Portuguese

    # Harsh/difficult phonetics
    "Qwikster",     # Netflix spinoff that failed (name contributed)
    "Tronc",        # Tribune Publishing rebrand (widely mocked)
    "Xobni",        # "Inbox" backwards (awkward)

    # Pharmaceutical rejects/problematic
    "Xeljanz",      # Difficult pronunciation
    "Otezla",       # Awkward
    "Prolia",       # Confusion potential

    # Generic/forgettable
    "Quibi",        # Failed streaming service
]

# TIER 4: NEUTRAL BASELINE
# Common English words to establish baseline distribution
NEUTRAL_WORDS = [
    # Common nouns
    "table", "chair", "window", "garden", "mountain", "river", "ocean",
    "forest", "desert", "island", "bridge", "tower", "castle", "temple",
    "palace", "market", "harbor", "valley", "meadow", "stream",

    # Abstract nouns
    "power", "energy", "spirit", "wisdom", "courage", "honor", "glory",
    "wonder", "marvel", "mystery", "legend", "heritage", "legacy",

    # Action words
    "venture", "explore", "discover", "journey", "voyage", "quest",
    "ascend", "evolve", "ignite", "emerge", "flourish", "triumph",

    # Quality words
    "pristine", "serene", "vibrant", "radiant", "stellar", "premium",
    "elite", "prime", "apex", "zenith", "pinnacle", "summit",
]

# TIER 5: PSEUDOWORDS FROM LINGUISTIC RESEARCH
# Based on patterns from Klink (2000) and similar studies
RESEARCH_PSEUDOWORDS = {
    # Front vowel names (perceived as: smaller, lighter, faster, feminine)
    "front_vowel": [
        "Frish", "Tivex", "Kilar", "Premi", "Velim", "Silka", "Mirel",
        "Elira", "Vixen", "Zephir", "Tilera", "Krilex",
    ],
    # Back vowel names (perceived as: larger, heavier, powerful, masculine)
    "back_vowel": [
        "Groma", "Torax", "Volum", "Krona", "Boltar", "Domax", "Gorath",
        "Voltar", "Bromax", "Tundra", "Magnum", "Thorax",
    ],
    # Plosive-heavy (perceived as: powerful, decisive)
    "plosive_heavy": [
        "Kaptex", "Bloktar", "Grotik", "Tektron", "Paktum", "Diktron",
    ],
    # Sonorant-heavy (perceived as: smooth, elegant)
    "sonorant_heavy": [
        "Lumara", "Selena", "Mirella", "Lorena", "Melora", "Sonara",
    ],
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ScoredName:
    """A brand name with its phonaesthetic score and metadata."""
    name: str
    score: float
    quality: str
    category: str  # excellent, problematic, neutral, pseudoword
    subcategory: str  # more specific classification
    pronounceable: bool
    details: Dict


@dataclass
class CalibrationResult:
    """Results of the calibration study."""
    corpus_size: int
    category_stats: Dict[str, Dict]
    optimal_thresholds: Dict[str, float]
    separation_metrics: Dict
    validation_results: Dict
    recommendations: List[str]


# =============================================================================
# Calibration Engine
# =============================================================================

class CalibrationEngine:
    """
    Performs rigorous calibration of phonaesthetic score thresholds.
    """

    def __init__(self):
        self.scored_names: List[ScoredName] = []
        self.category_scores: Dict[str, List[float]] = defaultdict(list)

    def build_corpus(self) -> int:
        """Build and score the complete corpus."""
        corpus = []

        # Add excellent brands
        for name in EXCELLENT_GLOBAL_BRANDS:
            corpus.append((name, "excellent", "global_brand"))
        for name in EXCELLENT_LINGUISTIC:
            corpus.append((name, "excellent", "linguistic_acclaim"))

        # Add problematic brands
        for name in PROBLEMATIC_BRANDS:
            corpus.append((name, "problematic", "known_issues"))

        # Add neutral baseline
        for name in NEUTRAL_WORDS:
            corpus.append((name, "neutral", "common_word"))

        # Add research pseudowords
        for subcat, names in RESEARCH_PSEUDOWORDS.items():
            for name in names:
                corpus.append((name, "pseudoword", subcat))

        # Score all names
        for name, category, subcategory in corpus:
            self._score_name(name, category, subcategory)

        return len(self.scored_names)

    def _score_name(self, name: str, category: str, subcategory: str):
        """Score a single name and add to corpus."""
        # Check pronounceability
        is_pron, _ = is_pronounceable(name)

        # Get phonaesthetic score
        result = phonaesthetic_score(name)

        scored = ScoredName(
            name=name,
            score=result['score'],
            quality=result['quality'],
            category=category,
            subcategory=subcategory,
            pronounceable=is_pron,
            details=result
        )

        self.scored_names.append(scored)
        self.category_scores[category].append(result['score'])

    def analyze_distributions(self) -> Dict[str, Dict]:
        """Analyze score distributions by category."""
        stats = {}

        for category, scores in self.category_scores.items():
            if not scores:
                continue

            sorted_scores = sorted(scores)
            n = len(sorted_scores)

            stats[category] = {
                'count': n,
                'min': min(scores),
                'max': max(scores),
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'stdev': statistics.stdev(scores) if n > 1 else 0,
                'p10': sorted_scores[int(n * 0.10)] if n > 10 else sorted_scores[0],
                'p25': sorted_scores[int(n * 0.25)] if n > 4 else sorted_scores[0],
                'p75': sorted_scores[int(n * 0.75)] if n > 4 else sorted_scores[-1],
                'p90': sorted_scores[int(n * 0.90)] if n > 10 else sorted_scores[-1],
            }

        return stats

    def calculate_separation(self) -> Dict:
        """Calculate separation metrics between categories."""
        excellent = self.category_scores.get('excellent', [])
        problematic = self.category_scores.get('problematic', [])
        neutral = self.category_scores.get('neutral', [])

        metrics = {}

        # Cohen's d between excellent and problematic
        if excellent and problematic:
            mean_diff = statistics.mean(excellent) - statistics.mean(problematic)
            pooled_std = ((statistics.stdev(excellent) ** 2 + statistics.stdev(problematic) ** 2) / 2) ** 0.5
            metrics['cohens_d_exc_prob'] = mean_diff / pooled_std if pooled_std > 0 else 0

        # Cohen's d between excellent and neutral
        if excellent and neutral:
            mean_diff = statistics.mean(excellent) - statistics.mean(neutral)
            pooled_std = ((statistics.stdev(excellent) ** 2 + statistics.stdev(neutral) ** 2) / 2) ** 0.5
            metrics['cohens_d_exc_neut'] = mean_diff / pooled_std if pooled_std > 0 else 0

        # Overlap percentage
        if excellent and problematic:
            exc_min, exc_max = min(excellent), max(excellent)
            prob_min, prob_max = min(problematic), max(problematic)

            overlap_start = max(exc_min, prob_min)
            overlap_end = min(exc_max, prob_max)

            if overlap_start < overlap_end:
                overlap_range = overlap_end - overlap_start
                total_range = max(exc_max, prob_max) - min(exc_min, prob_min)
                metrics['overlap_pct'] = (overlap_range / total_range) * 100 if total_range > 0 else 0
            else:
                metrics['overlap_pct'] = 0

        return metrics

    def find_optimal_thresholds(self) -> Dict[str, float]:
        """
        Find optimal thresholds using multiple methods.
        """
        excellent = sorted(self.category_scores.get('excellent', []))
        problematic = sorted(self.category_scores.get('problematic', []))
        neutral = sorted(self.category_scores.get('neutral', []))
        all_scores = sorted(s.score for s in self.scored_names)

        thresholds = {}

        # Method 1: Percentile-based on excellent brands
        # "Excellent" threshold = 25th percentile of known excellent brands
        # This means 75% of validated excellent brands would pass
        if excellent:
            n = len(excellent)
            thresholds['excellent_p25'] = excellent[int(n * 0.25)] if n > 4 else excellent[0]
            thresholds['excellent_p10'] = excellent[int(n * 0.10)] if n > 10 else excellent[0]

        # Method 2: Separation-based
        # Find threshold that best separates excellent from problematic
        if excellent and problematic:
            best_threshold = 0
            best_separation = 0

            for threshold in [i/100 for i in range(50, 65)]:
                exc_above = sum(1 for s in excellent if s >= threshold) / len(excellent)
                prob_below = sum(1 for s in problematic if s < threshold) / len(problematic)
                separation = exc_above + prob_below  # Sum of true positive + true negative rates

                if separation > best_separation:
                    best_separation = separation
                    best_threshold = threshold

            thresholds['optimal_separation'] = best_threshold
            thresholds['separation_score'] = best_separation

        # Method 3: Distribution-based on all scores
        if all_scores:
            n = len(all_scores)
            thresholds['global_p90'] = all_scores[int(n * 0.90)]
            thresholds['global_p75'] = all_scores[int(n * 0.75)]
            thresholds['global_p50'] = all_scores[int(n * 0.50)]

        return thresholds

    def cross_validate(self, n_folds: int = 5) -> Dict:
        """
        Cross-validate threshold selection.
        """
        import random

        # Shuffle and split
        excellent = [s for s in self.scored_names if s.category == 'excellent']
        problematic = [s for s in self.scored_names if s.category == 'problematic']

        random.shuffle(excellent)
        random.shuffle(problematic)

        fold_results = []

        for fold in range(n_folds):
            # Simple holdout validation
            exc_train = excellent[:int(len(excellent) * 0.8)]
            exc_test = excellent[int(len(excellent) * 0.8):]
            prob_train = problematic[:int(len(problematic) * 0.8)]
            prob_test = problematic[int(len(problematic) * 0.8):]

            # Find threshold on training set
            train_exc_scores = [s.score for s in exc_train]
            train_prob_scores = [s.score for s in prob_train]

            if train_exc_scores:
                threshold = sorted(train_exc_scores)[int(len(train_exc_scores) * 0.25)]

                # Evaluate on test set
                if exc_test and prob_test:
                    test_exc_scores = [s.score for s in exc_test]
                    test_prob_scores = [s.score for s in prob_test]

                    true_pos = sum(1 for s in test_exc_scores if s >= threshold) / len(test_exc_scores)
                    true_neg = sum(1 for s in test_prob_scores if s < threshold) / len(test_prob_scores)

                    fold_results.append({
                        'threshold': threshold,
                        'true_positive_rate': true_pos,
                        'true_negative_rate': true_neg,
                        'accuracy': (true_pos + true_neg) / 2
                    })

            # Rotate for next fold
            random.shuffle(excellent)
            random.shuffle(problematic)

        if fold_results:
            avg_threshold = statistics.mean(r['threshold'] for r in fold_results)
            avg_accuracy = statistics.mean(r['accuracy'] for r in fold_results)

            return {
                'n_folds': n_folds,
                'avg_threshold': avg_threshold,
                'avg_accuracy': avg_accuracy,
                'fold_results': fold_results
            }

        return {'error': 'Insufficient data for cross-validation'}

    def generate_recommendations(self, stats: Dict, thresholds: Dict, separation: Dict) -> List[str]:
        """Generate calibration recommendations based on analysis."""
        recommendations = []

        # Check if excellent and problematic are well-separated
        cohens_d = separation.get('cohens_d_exc_prob', 0)
        if cohens_d < 0.2:
            recommendations.append(
                f"WARNING: Poor separation between excellent and problematic brands (Cohen's d = {cohens_d:.2f}). "
                "The scoring algorithm may need refinement."
            )
        elif cohens_d < 0.5:
            recommendations.append(
                f"MODERATE separation between categories (Cohen's d = {cohens_d:.2f}). "
                "Thresholds will have some overlap."
            )
        else:
            recommendations.append(
                f"GOOD separation between categories (Cohen's d = {cohens_d:.2f}). "
                "Thresholds should work well."
            )

        # Recommend thresholds
        exc_p25 = thresholds.get('excellent_p25', 0.58)
        opt_sep = thresholds.get('optimal_separation', 0.57)

        # Use the more conservative (lower) threshold to avoid excluding good brands
        recommended_excellent = min(exc_p25, opt_sep) - 0.01  # Small buffer

        recommendations.append(
            f"RECOMMENDED 'excellent' threshold: {recommended_excellent:.3f} "
            f"(based on 25th percentile of validated excellent brands)"
        )

        # Good threshold: median of excellent brands
        if 'excellent' in stats:
            exc_median = stats['excellent']['median']
            recommendations.append(
                f"RECOMMENDED 'good' threshold: {exc_median - 0.02:.3f} "
                f"(below median of excellent brands)"
            )

        return recommendations

    def run_full_calibration(self) -> CalibrationResult:
        """Run the complete calibration study."""
        print("Building corpus...")
        corpus_size = self.build_corpus()
        print(f"  Scored {corpus_size} names")

        print("\nAnalyzing distributions...")
        stats = self.analyze_distributions()
        for cat, s in stats.items():
            print(f"  {cat}: n={s['count']}, mean={s['mean']:.3f}, "
                  f"range=[{s['min']:.3f}, {s['max']:.3f}]")

        print("\nCalculating separation metrics...")
        separation = self.calculate_separation()
        for key, val in separation.items():
            print(f"  {key}: {val:.3f}")

        print("\nFinding optimal thresholds...")
        thresholds = self.find_optimal_thresholds()
        for key, val in thresholds.items():
            print(f"  {key}: {val:.3f}")

        print("\nCross-validating...")
        validation = self.cross_validate()
        if 'avg_accuracy' in validation:
            print(f"  Average accuracy: {validation['avg_accuracy']:.1%}")
            print(f"  Average threshold: {validation['avg_threshold']:.3f}")

        print("\nGenerating recommendations...")
        recommendations = self.generate_recommendations(stats, thresholds, separation)
        for rec in recommendations:
            print(f"  - {rec}")

        return CalibrationResult(
            corpus_size=corpus_size,
            category_stats=stats,
            optimal_thresholds=thresholds,
            separation_metrics=separation,
            validation_results=validation,
            recommendations=recommendations
        )

    def get_detailed_report(self) -> str:
        """Generate a detailed calibration report."""
        lines = []
        lines.append("=" * 70)
        lines.append("PHONAESTHETIC SCORE CALIBRATION REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Corpus summary
        lines.append("CORPUS SUMMARY")
        lines.append("-" * 40)
        for cat in ['excellent', 'problematic', 'neutral', 'pseudoword']:
            count = len([s for s in self.scored_names if s.category == cat])
            lines.append(f"  {cat}: {count} names")
        lines.append("")

        # Top scoring names by category
        lines.append("TOP SCORING NAMES BY CATEGORY")
        lines.append("-" * 40)
        for cat in ['excellent', 'problematic', 'neutral']:
            cat_names = sorted(
                [s for s in self.scored_names if s.category == cat],
                key=lambda x: x.score,
                reverse=True
            )[:5]
            lines.append(f"  {cat.upper()}:")
            for s in cat_names:
                lines.append(f"    {s.name}: {s.score:.3f}")
        lines.append("")

        # Bottom scoring excellent brands (important for threshold)
        lines.append("LOWEST SCORING 'EXCELLENT' BRANDS (threshold floor)")
        lines.append("-" * 40)
        exc_names = sorted(
            [s for s in self.scored_names if s.category == 'excellent'],
            key=lambda x: x.score
        )[:10]
        for s in exc_names:
            lines.append(f"  {s.name}: {s.score:.3f}")
        lines.append("")

        # Highest scoring problematic brands (threshold ceiling)
        lines.append("HIGHEST SCORING 'PROBLEMATIC' BRANDS (false positive risk)")
        lines.append("-" * 40)
        prob_names = sorted(
            [s for s in self.scored_names if s.category == 'problematic'],
            key=lambda x: x.score,
            reverse=True
        )[:10]
        for s in prob_names:
            lines.append(f"  {s.name}: {s.score:.3f}")

        return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================

def run_calibration():
    """Run the full calibration study and return results."""
    engine = CalibrationEngine()
    result = engine.run_full_calibration()

    print("\n")
    print(engine.get_detailed_report())

    return engine, result


if __name__ == "__main__":
    run_calibration()
