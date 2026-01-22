#!/usr/bin/env python3
"""
Quality filtering and diversity selection for generated brand names.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Any

from similarity_checker import SimilarityChecker, normalized_similarity
from brandkit.generators.base_generator import HazardChecker
from brandkit.generators.phonemes import is_pronounceable
from brandkit.settings import get_setting


@dataclass
class Candidate:
    """Normalized candidate representation."""
    obj: Any
    name: str
    score: float
    suffix2: str
    suffix3: str
    prefix2: str


def _extract_name(obj: Any) -> str:
    if hasattr(obj, "name"):
        return str(obj.name)
    return str(obj)


def _extract_score(obj: Any) -> float:
    for attr in ("total_score", "score", "score_estimate"):
        if hasattr(obj, attr):
            try:
                return float(getattr(obj, attr))
            except Exception:
                pass
    return 0.5


def _build_candidates(names: Iterable[Any]) -> List[Candidate]:
    candidates = []
    for obj in names:
        name = _extract_name(obj)
        if not name:
            continue
        name_lower = name.lower()
        suffix2 = name_lower[-2:] if len(name_lower) >= 2 else name_lower
        suffix3 = name_lower[-3:] if len(name_lower) >= 3 else name_lower
        prefix2 = name_lower[:2] if len(name_lower) >= 2 else name_lower
        candidates.append(
            Candidate(
                obj=obj,
                name=name,
                score=_extract_score(obj),
                suffix2=suffix2,
                suffix3=suffix3,
                prefix2=prefix2,
            )
        )
    return candidates


def filter_and_rank(
    names: Iterable[Any],
    target_count: int,
    markets: str = "en_de",
    min_score: float | None = None,
    similarity_threshold: float | None = None,
    max_suffix_pct: float | None = None,
    max_prefix_pct: float | None = None,
) -> List[Any]:
    """
    Filter and rank names by pronounceability, hazards, similarity, and diversity.
    """
    candidates = _build_candidates(names)
    if not candidates:
        return []

    # Pronounceability filter (EN/DE)
    pronounceable = []
    for c in candidates:
        ok, _ = is_pronounceable(c.name, markets=markets)
        if ok:
            pronounceable.append(c)

    if not pronounceable:
        return []

    # Hazard filter
    hazard_checker = HazardChecker()
    safe = []
    for c in pronounceable:
        result = hazard_checker.check(c.name)
        if result.severity in ("high", "critical"):
            continue
        safe.append(c)

    if not safe:
        return []

    cfg = get_setting("quality_filter", {}) or {}
    if min_score is None:
        min_score = cfg.get("min_score")
    if similarity_threshold is None:
        similarity_threshold = cfg.get("similarity_threshold")
    if max_suffix_pct is None:
        max_suffix_pct = cfg.get("max_suffix_pct")
    if max_prefix_pct is None:
        max_prefix_pct = cfg.get("max_prefix_pct")

    if min_score is None or similarity_threshold is None or max_suffix_pct is None or max_prefix_pct is None:
        raise ValueError("quality_filter settings must be set in app.yaml")

    diversity_penalties = cfg.get("diversity_penalties") or {}
    suffix3_penalty = diversity_penalties.get("suffix3")
    suffix2_penalty = diversity_penalties.get("suffix2")
    prefix2_penalty = diversity_penalties.get("prefix2")
    if suffix3_penalty is None or suffix2_penalty is None or prefix2_penalty is None:
        raise ValueError("quality_filter.diversity_penalties must be set in app.yaml")
    sim_checker = SimilarityChecker()
    distinct = []
    for c in safe:
        sim_result = sim_checker.check(c.name)
        if sim_result.is_safe:
            distinct.append(c)

    if not distinct:
        return []

    # Score filter
    filtered = [c for c in distinct if c.score >= min_score]
    if not filtered:
        filtered = distinct

    # Diversity ranking
    suffix2_counts = Counter(c.suffix2 for c in filtered)
    suffix3_counts = Counter(c.suffix3 for c in filtered)
    prefix2_counts = Counter(c.prefix2 for c in filtered)

    def diversity_penalty(c: Candidate) -> float:
        penalty = 0.0
        penalty += max(0, suffix3_counts[c.suffix3] - 1) * suffix3_penalty
        penalty += max(0, suffix2_counts[c.suffix2] - 1) * suffix2_penalty
        penalty += max(0, prefix2_counts[c.prefix2] - 1) * prefix2_penalty
        return penalty

    ranked = sorted(
        filtered,
        key=lambda c: (c.score - diversity_penalty(c), c.score),
        reverse=True,
    )

    # Greedy selection for diversity and similarity spacing
    selected = []
    suffix2_sel = Counter()
    suffix3_sel = Counter()
    prefix2_sel = Counter()
    max_suffix = max(1, int(target_count * max_suffix_pct))
    max_prefix = max(1, int(target_count * max_prefix_pct))

    for c in ranked:
        if suffix3_sel[c.suffix3] >= max_suffix:
            continue
        if suffix2_sel[c.suffix2] >= max_suffix:
            continue
        if prefix2_sel[c.prefix2] >= max_prefix:
            continue

        too_similar = False
        for s in selected:
            if normalized_similarity(c.name, _extract_name(s)) >= similarity_threshold:
                too_similar = True
                break
        if too_similar:
            continue

        selected.append(c.obj)
        suffix2_sel[c.suffix2] += 1
        suffix3_sel[c.suffix3] += 1
        prefix2_sel[c.prefix2] += 1

        if len(selected) >= target_count:
            break

    return selected
