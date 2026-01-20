#!/usr/bin/env python3
"""
Database Module
===============
Re-exports the name database functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from namedb import (
    BrandNameDB,
    NameStatus,
    BlockReason,
    QualityTier,
    Comment,
    BrandName,
    get_namedb,
)

# Aliases for convenience
NameDB = BrandNameDB
NameRecord = BrandName
get_db = get_namedb

__all__ = [
    'NameDB',
    'BrandNameDB',
    'NameStatus',
    'BlockReason',
    'QualityTier',
    'Comment',
    'BrandName',
    'NameRecord',
    'get_db',
    'get_namedb',
]
