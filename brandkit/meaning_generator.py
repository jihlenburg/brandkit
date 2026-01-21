#!/usr/bin/env python3
"""
Brand Meaning Generator
=======================

Generates brand story/meaning explanations based on the phoneme components used.
Combines root meanings, suffix connotations, and cultural context into a
compelling brand narrative.

Usage:
    meaning = generate_brand_meaning(
        name="Voltara",
        culture="greek",
        roots=[("volt", "energy, electric")],
        suffix=("ara", "place, beauty"),
        prefix=None
    )
"""

from typing import List, Tuple, Optional


# Suffix connotations for brand meaning
SUFFIX_MEANINGS = {
    # Tech endings
    'ix': 'innovation and precision',
    'ex': 'excellence and expansion',
    'on': 'power and continuity',
    'ax': 'action and impact',
    'ox': 'strength and durability',
    'tek': 'technology and advancement',
    'tron': 'electronic precision',

    # Latin endings
    'us': 'classical authority',
    'um': 'completeness and unity',
    'is': 'essence and identity',
    'or': 'agency and action',
    'a': 'openness and accessibility',
    'o': 'strength and directness',

    # Nordic/European
    'en': 'belonging and identity',
    'er': 'one who acts',
    'heim': 'home and sanctuary',
    'gard': 'protection and enclosure',
    'berg': 'solidity and permanence',
    'vik': 'exploration and journey',

    # Elegant endings
    'ia': 'elegance and place',
    'io': 'dynamic energy',
    'ara': 'beauty and wonder',
    'ora': 'radiance and dawn',
    'ium': 'essence and element',
    'ella': 'grace and refinement',
    'ino': 'precision and craft',

    # Japanese-influenced
    'ra': 'community and harmony',
    'ka': 'transformation',
    'mi': 'beauty and essence',
    'ri': 'reason and clarity',
    'do': 'the way, mastery',

    # Celtic
    'wyn': 'purity and blessing',
    'wen': 'sacred whiteness',
    'an': 'small but mighty',

    # Turkic
    'kan': 'noble blood',
    'tan': 'dawn, new beginning',
    'lan': 'becoming, evolution',
    'chi': 'mastery and craft',
}

# Culture descriptions
CULTURE_DESCRIPTIONS = {
    'greek': 'Drawing from ancient Greek mythology and philosophy, this name evokes',
    'nordic': 'Inspired by Norse heritage and Scandinavian strength, this name embodies',
    'japanese': 'With roots in Japanese aesthetics and philosophy, this name captures',
    'latin': 'From the elegance of Latin and Romance languages, this name conveys',
    'celtic': 'Steeped in Celtic mysticism and natural wisdom, this name channels',
    'celestial': 'Reaching toward the cosmos and infinite possibilities, this name represents',
    'turkic': 'From the vast steppes and ancient nomadic traditions, this name carries',
    'animals': 'Channeling the power and majesty of the animal kingdom, this name embodies',
    'mythology': 'Drawing from global folklore and legendary creatures, this name captures',
    'landmarks': 'Inspired by majestic natural wonders and iconic landmarks, this name embodies',
    'rule_based': 'Crafted through phonetic optimization, this name combines',
    'blend': 'A fusion of multiple cultural influences, this name harmonizes',
}

# Brand archetype associations
ARCHETYPE_ASSOCIATIONS = {
    'power': 'strength, leadership, and decisive action',
    'light': 'illumination, clarity, and inspiration',
    'nature': 'authenticity, growth, and organic connection',
    'spirit': 'depth, meaning, and transcendence',
    'travel': 'exploration, freedom, and discovery',
    'celestial': 'aspiration, infinity, and wonder',
    'tech': 'innovation, precision, and future-forward thinking',
    'quality': 'excellence, refinement, and distinction',
    'abstract': 'ideas, concepts, and intellectual depth',
}


def generate_brand_meaning(
    name: str,
    culture: str = None,
    roots: List[Tuple[str, str]] = None,
    suffix: Tuple[str, str] = None,
    prefix: Tuple[str, str] = None,
    category: str = None,
) -> str:
    """
    Generate a brand meaning/story explanation.

    Args:
        name: The generated brand name
        culture: Cultural origin (greek, nordic, etc.)
        roots: List of (root_phoneme, meaning) tuples used
        suffix: (suffix, meaning) tuple if applicable
        prefix: (prefix, meaning) tuple if applicable
        category: Root category (power, light, nature, etc.)

    Returns:
        A 1-3 sentence brand meaning explanation
    """
    parts = []

    # Opening based on culture
    if culture and culture in CULTURE_DESCRIPTIONS:
        parts.append(CULTURE_DESCRIPTIONS[culture])
    else:
        parts.append("This distinctive name combines")

    # Root meanings
    if roots:
        root_meanings = [meaning for _, meaning in roots if meaning]
        if root_meanings:
            if len(root_meanings) == 1:
                parts.append(f"the concept of {root_meanings[0]}")
            else:
                parts.append(f"the concepts of {', '.join(root_meanings[:-1])} and {root_meanings[-1]}")

    # Prefix meaning
    if prefix and prefix[1]:
        parts.append(f"enhanced by the prefix suggesting {prefix[1]}")

    # Suffix connotation
    if suffix:
        suffix_str = suffix[0] if isinstance(suffix, tuple) else suffix
        if suffix_str in SUFFIX_MEANINGS:
            parts.append(f"with an ending that evokes {SUFFIX_MEANINGS[suffix_str]}")

    # Category association
    if category and category in ARCHETYPE_ASSOCIATIONS:
        parts.append(f"— embodying {ARCHETYPE_ASSOCIATIONS[category]}")

    # Build the narrative
    if len(parts) <= 1:
        return f"{name} is a distinctive brand name with strong phonetic appeal."

    # Join parts intelligently
    narrative = ' '.join(parts)

    # Clean up the narrative
    narrative = narrative.replace('  ', ' ')
    if not narrative.endswith('.'):
        narrative += '.'

    return narrative


def generate_meaning_from_components(
    name: str,
    method: str = None,
    root_info: dict = None,
) -> str:
    """
    Generate brand meaning from generation metadata.

    This is a simpler interface that takes the raw generation info
    and produces a brand story.

    Args:
        name: The brand name
        method: Generation method (greek, nordic, etc.)
        root_info: Dictionary with keys like 'roots', 'suffix', 'category'

    Returns:
        Brand meaning explanation
    """
    if not root_info:
        root_info = {}

    return generate_brand_meaning(
        name=name,
        culture=method,
        roots=root_info.get('roots', []),
        suffix=root_info.get('suffix'),
        prefix=root_info.get('prefix'),
        category=root_info.get('category'),
    )


def enrich_generated_name(name_obj, method: str = None) -> dict:
    """
    Enrich a generated name object with semantic meaning.

    Args:
        name_obj: A generated name (GreekName, NordicName, etc.)
        method: The generation method used

    Returns:
        Dictionary with name and semantic_meaning
    """
    name_str = name_obj.name if hasattr(name_obj, 'name') else str(name_obj)

    # Try to extract root info from the name object
    root_info = {}

    if hasattr(name_obj, 'roots_used'):
        roots = name_obj.roots_used
        if hasattr(name_obj, 'meaning_hints'):
            meanings = name_obj.meaning_hints
            root_info['roots'] = list(zip(roots, meanings)) if len(roots) == len(meanings) else [(r, '') for r in roots]
        else:
            root_info['roots'] = [(r, '') for r in roots]

    if hasattr(name_obj, 'category'):
        root_info['category'] = name_obj.category

    meaning = generate_meaning_from_components(
        name=name_str,
        method=method,
        root_info=root_info,
    )

    return {
        'name': name_str,
        'semantic_meaning': meaning,
    }


# =============================================================================
# LLM-Enhanced Meaning Generation
# =============================================================================

def generate_llm_meaning(
    name: str,
    culture: str = None,
    roots: list = None,
    category: str = None,
    api_key: str = None,
    industry: str = None,
) -> str:
    """
    Generate a rich brand meaning using Claude LLM.

    This provides more compelling, creative brand stories than the
    template-based approach. Requires an Anthropic API key.

    Args:
        name: The brand name
        culture: Cultural origin (greek, nordic, etc.)
        roots: List of (root, meaning) tuples
        category: Brand archetype/category
        api_key: Anthropic API key (uses env var if not provided)
        industry: Target industry context

    Returns:
        A compelling 2-3 sentence brand meaning explanation
    """
    import os

    # Get API key
    key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not key:
        # Fall back to template-based generation
        return generate_brand_meaning(name, culture, roots, category=category)

    try:
        import anthropic
    except ImportError:
        # Anthropic not installed, fall back to template
        return generate_brand_meaning(name, culture, roots, category=category)

    # Build context for LLM
    context_parts = []
    if culture:
        context_parts.append(f"Cultural origin: {culture}")
    if roots:
        root_info = ", ".join([f"{r[0]} ({r[1]})" for r in roots if len(r) >= 2])
        if root_info:
            context_parts.append(f"Root meanings: {root_info}")
    if category:
        context_parts.append(f"Brand archetype: {category}")
    if industry:
        context_parts.append(f"Target industry: {industry}")

    context = "\n".join(context_parts) if context_parts else "No specific context"

    prompt = f"""Create a compelling brand meaning/story for the brand name "{name}".

Context:
{context}

Guidelines:
- Write 2-3 sentences that explain the brand's essence
- Connect the name's sounds and origins to brand values
- Make it inspirational but authentic
- Avoid clichés and generic marketing language
- Focus on what makes this name distinctive

Brand meaning:"""

    try:
        client = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model="claude-3-haiku-20240307",  # Fast and cost-effective
            max_tokens=200,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text.strip()
    except Exception:
        # Any error, fall back to template
        return generate_brand_meaning(name, culture, roots, category=category)


def generate_meaning_with_llm_fallback(
    name: str,
    culture: str = None,
    roots: list = None,
    category: str = None,
    use_llm: bool = False,
    api_key: str = None,
    industry: str = None,
) -> str:
    """
    Generate brand meaning with optional LLM enhancement.

    By default uses fast template-based generation. Set use_llm=True
    for richer AI-generated meanings (requires API key).

    Args:
        name: The brand name
        culture: Cultural origin
        roots: List of (root, meaning) tuples
        category: Brand archetype
        use_llm: Whether to use LLM generation
        api_key: Anthropic API key
        industry: Target industry

    Returns:
        Brand meaning string
    """
    if use_llm:
        return generate_llm_meaning(
            name=name,
            culture=culture,
            roots=roots,
            category=category,
            api_key=api_key,
            industry=industry,
        )
    else:
        return generate_brand_meaning(
            name=name,
            culture=culture,
            roots=roots,
            category=category,
        )


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == '__main__':
    import sys

    # Test examples
    examples = [
        {
            'name': 'Voltara',
            'culture': 'greek',
            'roots': [('volt', 'energy, electric')],
            'suffix': ('ara', 'beauty, place'),
            'category': 'power',
        },
        {
            'name': 'Fenrikon',
            'culture': 'nordic',
            'roots': [('fenrir', 'giant wolf, power')],
            'suffix': ('on', 'power and continuity'),
            'category': 'beasts',
        },
        {
            'name': 'Hikaria',
            'culture': 'japanese',
            'roots': [('hikari', 'light')],
            'suffix': ('ia', 'elegance and place'),
            'category': 'light',
        },
        {
            'name': 'Turanix',
            'culture': 'turkic',
            'roots': [('turan', 'homeland, origin')],
            'suffix': ('ix', 'innovation and precision'),
            'category': 'concept',
        },
    ]

    print("Brand Meaning Examples")
    print("=" * 60)

    for ex in examples:
        meaning = generate_brand_meaning(**ex)
        print(f"\n{ex['name']}:")
        print(f"  {meaning}")
