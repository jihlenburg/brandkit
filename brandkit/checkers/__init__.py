#!/usr/bin/env python3
"""
Brand Name Checkers
===================
Provides validation and availability checking:
- Trademark: EUIPO and USPTO/WIPO trademark searches
- Domain: DNS-based domain availability
- Similarity: Phonetic similarity to known brands
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

# Import checkers from existing modules
from euipo_checker import (
    EUIPOChecker,
    TrademarkResult as EUIPOResult,
    TrademarkMatch as EUIPOMatch,
)
from rapidapi_checker import (
    RapidAPIChecker,
    TrademarkResult as USPTOResult,
    TrademarkMatch as USPTOMatch,
    get_rapidapi_checker,
)
from domain_checker import (
    DomainChecker,
    DomainResult,
    DomainCheckResult,
    get_domain_checker,
    check_domain,
)
from similarity_checker import (
    SimilarityChecker,
    SimilarityResult,
    SimilarityMatch,
    check_similarity,
    soundex,
    metaphone,
    levenshtein_distance,
    normalized_similarity,
    KNOWN_BRANDS,
)


class TrademarkChecker:
    """
    Unified trademark checker combining EUIPO and USPTO/WIPO searches.

    Usage:
        checker = TrademarkChecker()
        result = checker.check("Voltix")
        print(f"EU conflicts: {result['euipo'].total_matches}")
        print(f"US conflicts: {result['uspto'].total_matches}")
    """

    def __init__(self,
                 euipo_client_id: str = None,
                 euipo_client_secret: str = None,
                 rapidapi_key: str = None):
        """
        Initialize with API credentials.

        Args:
            euipo_client_id: EUIPO API client ID
            euipo_client_secret: EUIPO API client secret
            rapidapi_key: RapidAPI key for USPTO/WIPO
        """
        self._euipo = None
        self._rapidapi = None

        if euipo_client_id and euipo_client_secret:
            self._euipo = EUIPOChecker(euipo_client_id, euipo_client_secret)

        if rapidapi_key:
            self._rapidapi = RapidAPIChecker(rapidapi_key)

    @property
    def has_euipo(self) -> bool:
        return self._euipo is not None

    @property
    def has_uspto(self) -> bool:
        return self._rapidapi is not None

    def check(self, name: str, nice_classes: list = None) -> dict:
        """
        Check trademark availability in both EU and US.

        Args:
            name: Brand name to check
            nice_classes: Nice classification codes (e.g., [9, 12])

        Returns:
            Dictionary with 'euipo' and 'uspto' results
        """
        result = {
            'name': name,
            'euipo': None,
            'uspto': None,
            'is_available': True,
        }

        if self._euipo:
            try:
                result['euipo'] = self._euipo.check(name, nice_classes=nice_classes)
                if result['euipo'].found:
                    result['is_available'] = False
            except Exception as e:
                result['euipo_error'] = str(e)

        if self._rapidapi:
            try:
                result['uspto'] = self._rapidapi.check(name, nice_classes=nice_classes)
                if result['uspto'].found:
                    result['is_available'] = False
            except Exception as e:
                result['uspto_error'] = str(e)

        return result

    def check_batch(self, names: list, nice_classes: list = None) -> dict:
        """Check multiple names."""
        return {name: self.check(name, nice_classes) for name in names}


__all__ = [
    # Unified
    'TrademarkChecker',
    # EUIPO
    'EUIPOChecker',
    'EUIPOResult',
    'EUIPOMatch',
    # USPTO/WIPO
    'RapidAPIChecker',
    'USPTOResult',
    'USPTOMatch',
    'get_rapidapi_checker',
    # Domain
    'DomainChecker',
    'DomainResult',
    'DomainCheckResult',
    'get_domain_checker',
    'check_domain',
    # Similarity
    'SimilarityChecker',
    'SimilarityResult',
    'SimilarityMatch',
    'check_similarity',
    'soundex',
    'metaphone',
    'levenshtein_distance',
    'normalized_similarity',
    'KNOWN_BRANDS',
]
