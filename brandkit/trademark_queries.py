#!/usr/bin/env python3
"""
Trademark query variant generation.
"""

from __future__ import annotations

from typing import List

from brandkit.generators.phonemes import load_strategies


def _dedupe_letters(text: str) -> str:
    if not text:
        return text
    out = [text[0]]
    for ch in text[1:]:
        if ch != out[-1]:
            out.append(ch)
    return ''.join(out)


def _vowel_stripped(text: str) -> str:
    if not text:
        return text
    vowels = set("aeiou")
    out = [text[0]]
    for ch in text[1:]:
        if ch not in vowels:
            out.append(ch)
    return ''.join(out)


def _consonant_skeleton(text: str) -> str:
    vowels = set("aeiou")
    return ''.join(ch for ch in text if ch.isalpha() and ch not in vowels)


def generate_query_variants(name: str) -> List[str]:
    """
    Generate query variants for trademark searches based on strategies.yaml.
    """
    strategies = load_strategies()
    cfg = strategies.raw.get("trademark_queries", {})

    enabled = cfg.get("enabled", False)
    include_original = cfg.get("include_original", True)
    max_variants = int(cfg.get("max_variants", 3))
    min_length = int(cfg.get("min_length", 4))
    variant_steps = cfg.get("variants", [])

    if not enabled:
        return [name]

    base = name.lower().strip()
    variants = []

    if include_original:
        variants.append(name)

    def add_variant(value: str):
        if not value:
            return
        if len(value) < min_length:
            return
        if value.lower() == base:
            return
        if value not in variants:
            variants.append(value)

    for step in variant_steps:
        if len(variants) >= max_variants + (1 if include_original else 0):
            break
        if step == "vowel_stripped":
            add_variant(_vowel_stripped(base))
        elif step == "consonant_skeleton":
            add_variant(_consonant_skeleton(base))
        elif step == "dedupe_letters":
            add_variant(_dedupe_letters(base))

    return variants
