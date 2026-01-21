#!/usr/bin/env python3
"""
EU/German Trademark Law Compliance Module
==========================================

Implements additional checks based on EU Trademark Regulation (EUTMR) and
German Markengesetz (MarkenG) requirements.

Key Legal Principles:
1. Verwechslungsgefahr (likelihood of confusion) - § 9 MarkenG, Art. 8(1)(b) EUTMR
2. Klangliche Ähnlichkeit (phonetic similarity) - core to German case law
3. Begriffliche Ähnlichkeit (conceptual similarity) - meaning-based confusion
4. Bildliche Ähnlichkeit (visual similarity) - appearance-based
5. Bekannte Marken (well-known marks) - Art. 8(5) EUTMR, § 9(1) Nr. 3 MarkenG
6. Dienstleistungsähnlichkeit (service similarity) - related Nice classes

References:
- BGH GRUR 2004, 783 - "NEURO-VIBOLEX/NEURO-FIBRAFLEX"
- EuGH C-251/95 - SABEL/PUMA
- BGH GRUR 2012, 1040 - "pjur/pure"
"""

from typing import List, Dict, Tuple, Optional
import re


# =============================================================================
# Related Nice Classes (Dienstleistungsähnlichkeit)
# =============================================================================

# Classes that are often considered related/similar by EUIPO/DPMA
# Based on EUIPO TMclass similarity tool and German case law
RELATED_CLASSES = {
    # Technology & Software
    9: [35, 38, 41, 42],   # Software often conflicts with services
    42: [9, 35, 38, 41],   # IT services relate to software goods

    # Vehicles & Transport
    12: [35, 37, 39],      # Vehicles relate to repair, transport
    37: [12, 35, 39],      # Repair services relate to goods
    39: [12, 35, 37],      # Transport relates to vehicles

    # Food & Beverages
    29: [30, 31, 32, 33, 35, 43],
    30: [29, 31, 32, 33, 35, 43],
    31: [29, 30, 32, 35, 44],
    32: [29, 30, 33, 35, 43],
    33: [29, 30, 32, 35, 43],
    43: [29, 30, 32, 33, 35],

    # Fashion & Retail
    25: [18, 24, 26, 35],  # Clothing relates to accessories, retail
    18: [25, 26, 35],      # Leather goods relate to fashion
    35: [9, 12, 25, 29, 30, 31, 32, 33, 42, 43],  # Retail/advertising is broad

    # Health & Pharma
    5: [3, 10, 35, 42, 44],  # Pharma relates to medical, cosmetic
    10: [5, 35, 44],         # Medical devices
    44: [5, 10, 35, 42],     # Medical services

    # Finance & Business
    36: [35, 45],          # Insurance/finance relate to business
    45: [35, 36, 42],      # Legal services
}


def get_related_classes(nice_classes: List[int]) -> List[int]:
    """
    Get Nice classes that are potentially related (similar goods/services).

    Under EU law, marks in related classes can still create confusion
    if consumers might assume a common origin.

    Args:
        nice_classes: List of Nice classes to check

    Returns:
        List of related Nice classes (including original classes)
    """
    related = set(nice_classes)
    for cls in nice_classes:
        if cls in RELATED_CLASSES:
            related.update(RELATED_CLASSES[cls])
    return sorted(related)


def check_class_overlap(query_classes: List[int],
                       match_classes: List[int],
                       include_related: bool = True) -> Dict:
    """
    Check for overlap between query Nice classes and match Nice classes.

    Args:
        query_classes: Nice classes for the new mark
        match_classes: Nice classes of the existing trademark
        include_related: Also check related classes (recommended for EU)

    Returns:
        Dict with overlap info and risk assessment
    """
    if not query_classes or not match_classes:
        return {
            'direct_overlap': [],
            'related_overlap': [],
            'risk_level': 'UNKNOWN',
            'requires_detailed_analysis': True,
        }

    query_set = set(query_classes)
    match_set = set(match_classes)

    # Direct overlap - same classes
    direct_overlap = sorted(query_set & match_set)

    # Related overlap - classes considered similar
    related_overlap = []
    if include_related:
        query_related = set(get_related_classes(query_classes))
        related_overlap = sorted((query_related & match_set) - query_set)

    # Assess risk
    if direct_overlap:
        risk = 'CRITICAL' if len(direct_overlap) >= 2 else 'HIGH'
    elif related_overlap:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'

    return {
        'direct_overlap': direct_overlap,
        'related_overlap': related_overlap,
        'risk_level': risk,
        'requires_detailed_analysis': bool(related_overlap),
    }


# =============================================================================
# Well-Known Marks (Bekannte Marken)
# =============================================================================

# Sample of well-known marks that enjoy extended protection under Art. 8(5) EUTMR
# These marks can block registration even in unrelated classes
WELL_KNOWN_MARKS = {
    # Auto
    'VOLKSWAGEN', 'VW', 'BMW', 'MERCEDES', 'PORSCHE', 'AUDI', 'OPEL',
    'TOYOTA', 'HONDA', 'NISSAN', 'FORD', 'TESLA',
    # Tech
    'APPLE', 'GOOGLE', 'MICROSOFT', 'AMAZON', 'META', 'FACEBOOK',
    'SAMSUNG', 'SONY', 'INTEL', 'IBM', 'ORACLE', 'SAP',
    # Consumer
    'COCA-COLA', 'PEPSI', 'NIKE', 'ADIDAS', 'PUMA',
    # Pharma
    'BAYER', 'PFIZER', 'ROCHE', 'NOVARTIS',
    # German/EU specific
    'SIEMENS', 'BOSCH', 'ALLIANZ', 'DEUTSCHE BANK', 'LUFTHANSA',
    'HARIBO', 'MILKA', 'NIVEA', 'HENKEL',
}

# Common prefixes/suffixes of well-known marks (for partial matches)
WELL_KNOWN_ELEMENTS = {
    'VOLKS', 'BENZ', 'COCA', 'COLA',
}


def check_well_known_similarity(name: str) -> Dict:
    """
    Check if name is similar to well-known marks.

    Well-known marks enjoy extended protection under:
    - Art. 8(5) EUTMR (EU Trademark Regulation)
    - § 9(1) Nr. 3 MarkenG (German Trademark Act)

    This means they can block registration even in UNRELATED classes
    if the new mark would take unfair advantage or be detrimental.

    Args:
        name: Brand name to check

    Returns:
        Dict with similarity findings and risk
    """
    name_upper = name.upper()
    findings = []

    # Exact match
    if name_upper in WELL_KNOWN_MARKS:
        return {
            'is_similar': True,
            'matches': [name_upper],
            'match_type': 'EXACT',
            'risk_level': 'CRITICAL',
            'legal_basis': 'Art. 8(5) EUTMR, § 9(1) Nr. 3 MarkenG',
            'note': 'Exact match with well-known mark - extended protection applies',
        }

    # Partial/substring match
    for mark in WELL_KNOWN_MARKS:
        # Check if name contains well-known mark
        if mark in name_upper and len(mark) >= 4:
            findings.append((mark, 'CONTAINS', 0.9))
        # Check if well-known mark contains name
        elif name_upper in mark and len(name_upper) >= 4:
            findings.append((mark, 'CONTAINED_IN', 0.7))
        # Check prefix/suffix match
        elif name_upper.startswith(mark[:4]) or name_upper.endswith(mark[-4:]):
            findings.append((mark, 'PREFIX_SUFFIX', 0.5))

    # Check well-known elements
    for element in WELL_KNOWN_ELEMENTS:
        if element in name_upper:
            findings.append((element, 'ELEMENT', 0.6))

    if findings:
        # Sort by similarity score
        findings.sort(key=lambda x: x[2], reverse=True)
        best_match = findings[0]

        return {
            'is_similar': True,
            'matches': [f[0] for f in findings[:3]],
            'match_type': best_match[1],
            'similarity': best_match[2],
            'risk_level': 'HIGH' if best_match[2] >= 0.7 else 'MEDIUM',
            'legal_basis': 'Art. 8(5) EUTMR, § 9(1) Nr. 3 MarkenG',
            'note': 'Similarity to well-known mark - extended protection may apply',
        }

    return {
        'is_similar': False,
        'matches': [],
        'risk_level': 'LOW',
    }


# =============================================================================
# Conceptual Similarity (Begriffliche Ähnlichkeit)
# =============================================================================

# German words with clear meanings that could cause conceptual confusion
CONCEPTUAL_TERMS = {
    # Power/Energy
    'kraft': ['power', 'force', 'energy'],
    'energie': ['energy', 'power'],
    'volt': ['voltage', 'electric'],
    'strom': ['current', 'stream', 'electricity'],

    # Speed/Motion
    'schnell': ['fast', 'quick', 'rapid'],
    'blitz': ['lightning', 'flash'],
    'turbo': ['turbo', 'fast'],
    'jet': ['jet', 'stream'],

    # Quality
    'prima': ['prime', 'first', 'excellent'],
    'super': ['super', 'excellent'],
    'gold': ['gold', 'golden', 'premium'],
    'diamant': ['diamond'],

    # Nature
    'natur': ['nature', 'natural'],
    'bio': ['bio', 'organic'],
    'grün': ['green'],
    'öko': ['eco'],
}


def check_conceptual_similarity(name1: str, name2: str) -> Dict:
    """
    Check for conceptual (meaning-based) similarity between names.

    Under EU case law (SABEL/PUMA), conceptual similarity can
    contribute to likelihood of confusion even when marks
    differ phonetically and visually.

    Args:
        name1: First name (your brand)
        name2: Second name (existing trademark)

    Returns:
        Dict with conceptual similarity assessment
    """
    n1_lower = name1.lower()
    n2_lower = name2.lower()

    shared_concepts = []

    for term, concepts in CONCEPTUAL_TERMS.items():
        in_n1 = term in n1_lower
        in_n2 = term in n2_lower

        if in_n1 and in_n2:
            shared_concepts.append({
                'term': term,
                'concepts': concepts,
                'type': 'DIRECT',
            })
        elif in_n1:
            # Check if n2 contains a synonym
            for synonym in concepts:
                if synonym in n2_lower:
                    shared_concepts.append({
                        'term': term,
                        'synonym': synonym,
                        'concepts': concepts,
                        'type': 'SYNONYM',
                    })
                    break

    if shared_concepts:
        return {
            'is_similar': True,
            'shared_concepts': shared_concepts,
            'similarity_score': min(0.3 + 0.2 * len(shared_concepts), 0.8),
            'contributes_to_confusion': True,
            'legal_basis': 'EuGH C-251/95 SABEL/PUMA - begriffliche Ähnlichkeit',
        }

    return {
        'is_similar': False,
        'shared_concepts': [],
        'similarity_score': 0.0,
        'contributes_to_confusion': False,
    }


# =============================================================================
# German-Specific Phonetic Rules
# =============================================================================

def german_phonetic_equivalents(name: str) -> List[str]:
    """
    Generate phonetically equivalent spellings under German pronunciation.

    German has specific sound-letter correspondences that differ from English.
    Two marks can be phonetically identical despite different spellings.

    Examples:
    - "ei" and "ai" sound identical in German
    - "f" and "v" at word start often sound the same
    - "k" and "c" before a/o/u are identical
    - "ie" and long "i" sound the same

    Args:
        name: Brand name

    Returns:
        List of phonetically equivalent spellings
    """
    equivalents = [name]
    n = name.lower()

    # German sound equivalences
    substitutions = [
        ('ei', 'ai'),
        ('ai', 'ei'),
        ('ie', 'i'),  # Long i
        ('i', 'ie'),
        ('ph', 'f'),
        ('f', 'ph'),
        ('v', 'f'),   # Initial v sounds like f
        ('f', 'v'),
        ('c', 'k'),   # Before a/o/u
        ('k', 'c'),
        ('ck', 'k'),
        ('k', 'ck'),
        ('ks', 'x'),
        ('x', 'ks'),
        ('chs', 'x'),
        ('x', 'chs'),
        ('sch', 'sh'),
        ('sh', 'sch'),
        ('tz', 'z'),
        ('z', 'tz'),
        ('ss', 'ß'),
        ('ß', 'ss'),
        ('ae', 'ä'),
        ('ä', 'ae'),
        ('oe', 'ö'),
        ('ö', 'oe'),
        ('ue', 'ü'),
        ('ü', 'ue'),
    ]

    for old, new in substitutions:
        if old in n:
            variant = n.replace(old, new)
            if variant != n and variant not in equivalents:
                equivalents.append(variant)

    return equivalents


def german_phonetic_similarity(name1: str, name2: str) -> float:
    """
    Compute phonetic similarity with German pronunciation rules.

    This supplements the English-based Soundex/Metaphone with
    German-specific equivalences.

    Args:
        name1, name2: Names to compare

    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Get all phonetic equivalents for both names
    equiv1 = set(german_phonetic_equivalents(name1))
    equiv2 = set(german_phonetic_equivalents(name2))

    # Check for any overlap
    if equiv1 & equiv2:
        return 1.0

    # Check if any equivalent of name1 matches any equivalent of name2
    from .phonetic_similarity import normalized_levenshtein

    max_sim = 0.0
    for e1 in equiv1:
        for e2 in equiv2:
            sim = normalized_levenshtein(e1, e2)
            max_sim = max(max_sim, sim)

    return max_sim


# =============================================================================
# Comprehensive EU Trademark Check
# =============================================================================

def comprehensive_eu_check(query_name: str,
                          match_name: str,
                          match_status: str = None,
                          query_classes: List[int] = None,
                          match_classes: List[int] = None) -> Dict:
    """
    Comprehensive trademark conflict analysis for EU/German market.

    Applies all relevant legal tests:
    1. Phonetic similarity (klangliche Ähnlichkeit)
    2. Visual similarity (bildliche Ähnlichkeit)
    3. Conceptual similarity (begriffliche Ähnlichkeit)
    4. Class overlap including related classes
    5. Well-known mark check

    Args:
        query_name: Your proposed brand name
        match_name: Existing trademark name
        match_status: Status of existing mark (LIVE, PENDING, DEAD)
        query_classes: Nice classes you want to register
        match_classes: Nice classes of existing mark

    Returns:
        Comprehensive risk assessment dict
    """
    from .phonetic_similarity import (
        compute_phonetic_similarity,
        normalized_levenshtein,
        calculate_risk_level
    )

    result = {
        'query_name': query_name,
        'match_name': match_name,
        'analyses': {},
        'overall_risk': 'LOW',
        'recommendations': [],
    }

    # 1. Phonetic similarity (English + German)
    english_phon = compute_phonetic_similarity(query_name, match_name)
    german_phon = german_phonetic_similarity(query_name, match_name)
    phonetic_score = max(english_phon, german_phon)

    result['analyses']['phonetic'] = {
        'english_score': english_phon,
        'german_score': german_phon,
        'combined_score': phonetic_score,
        'is_similar': phonetic_score > 0.6,
    }

    # 2. Visual similarity
    visual_score = normalized_levenshtein(query_name, match_name)
    result['analyses']['visual'] = {
        'score': visual_score,
        'is_similar': visual_score > 0.7,
    }

    # 3. Conceptual similarity
    conceptual = check_conceptual_similarity(query_name, match_name)
    result['analyses']['conceptual'] = conceptual

    # 4. Class overlap
    if query_classes and match_classes:
        class_analysis = check_class_overlap(query_classes, match_classes)
        result['analyses']['class_overlap'] = class_analysis
    else:
        result['analyses']['class_overlap'] = {'risk_level': 'UNKNOWN'}

    # 5. Well-known mark check (for the match)
    well_known = check_well_known_similarity(match_name)
    result['analyses']['well_known'] = well_known

    # Calculate overall risk
    risks = []

    if phonetic_score >= 0.8:
        risks.append('CRITICAL')
        result['recommendations'].append(
            'HIGH phonetic similarity - likely Verwechslungsgefahr under BGH case law'
        )
    elif phonetic_score >= 0.6:
        risks.append('HIGH')
        result['recommendations'].append(
            'Significant phonetic similarity - detailed analysis recommended'
        )

    if visual_score >= 0.8:
        risks.append('HIGH')
        result['recommendations'].append(
            'High visual similarity - bildliche Ähnlichkeit likely found'
        )

    if conceptual.get('is_similar'):
        risks.append('MEDIUM')
        result['recommendations'].append(
            f"Conceptual similarity via: {[c['term'] for c in conceptual.get('shared_concepts', [])]}"
        )

    class_risk = result['analyses']['class_overlap'].get('risk_level', 'UNKNOWN')
    if class_risk in ('CRITICAL', 'HIGH'):
        risks.append(class_risk)
        direct = result['analyses']['class_overlap'].get('direct_overlap', [])
        if direct:
            result['recommendations'].append(
                f'Direct class overlap in classes: {direct}'
            )

    if well_known.get('is_similar'):
        risks.append(well_known.get('risk_level', 'MEDIUM'))
        result['recommendations'].append(
            f"Similarity to well-known mark: {well_known.get('matches')} - "
            f"extended protection under Art. 8(5) EUTMR may apply"
        )

    # Determine overall risk (highest of all)
    risk_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'UNKNOWN']
    result['overall_risk'] = min(risks, key=lambda r: risk_order.index(r)) if risks else 'LOW'

    # Status-based adjustment
    if match_status:
        status_upper = match_status.upper()
        if status_upper in ('DEAD', 'ABANDONED', 'CANCELLED', 'EXPIRED'):
            result['status_note'] = 'Mark is no longer active - reduced but not zero risk'
            # Don't downgrade CRITICAL, but can reduce HIGH to MEDIUM
            if result['overall_risk'] == 'HIGH':
                result['overall_risk'] = 'MEDIUM'
        elif status_upper in ('PENDING', 'FILED', 'UNDER EXAMINATION'):
            result['status_note'] = 'Mark is pending - could still fail or succeed'

    return result


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python eu_trademark_analysis.py <your_name> <existing_mark>")
        print("\nExample: python eu_trademark_analysis.py Voltix Voltic")
        sys.exit(1)

    query = sys.argv[1]
    match = sys.argv[2]

    # Optional Nice classes
    query_classes = [9, 12] if len(sys.argv) < 4 else [int(c) for c in sys.argv[3].split(',')]
    match_classes = [9, 12] if len(sys.argv) < 5 else [int(c) for c in sys.argv[4].split(',')]

    result = comprehensive_eu_check(
        query_name=query,
        match_name=match,
        match_status='LIVE',
        query_classes=query_classes,
        match_classes=match_classes,
    )

    print(f"\nEU/German Trademark Analysis: '{query}' vs '{match}'")
    print("=" * 60)
    print(json.dumps(result, indent=2))
