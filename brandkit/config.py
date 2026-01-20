#!/usr/bin/env python3
"""
Configuration Management
========================
Loads API keys and settings from .env file.
Provides Nice classification profiles for trademark searches.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Nice Classification Profiles
# =============================================================================
# Predefined product category profiles mapping to relevant Nice classes.
# Users can use these or specify custom class lists.
#
# Reference: https://www.wipo.int/classifications/nice/nclpub/en/fr/

NICE_PROFILES = {
    # Electronics & Technology
    "electronics": {
        "classes": [9],
        "description": "Electrical and electronic apparatus, computers, software",
    },
    "software": {
        "classes": [9, 42],
        "description": "Computer software and related services",
    },

    # Energy & Power
    "energy": {
        "classes": [7, 9, 11],
        "description": "Power generation, electrical apparatus, energy systems",
    },
    "power_electronics": {
        "classes": [7, 9],
        "description": "DC/DC converters, inverters, power supplies",
    },

    # Vehicles & Transport
    "automotive": {
        "classes": [7, 9, 12],
        "description": "Vehicles, automotive parts and accessories",
    },
    "camping_rv": {
        "classes": [7, 9, 11, 12],
        "description": "Camping equipment, RV/caravan accessories, mobile power",
    },

    # Consumer Goods
    "clothing": {
        "classes": [25, 35],
        "description": "Clothing, footwear, headgear",
    },
    "household": {
        "classes": [11, 21],
        "description": "Household appliances and utensils",
    },
    "food_beverage": {
        "classes": [29, 30, 32, 33],
        "description": "Food products and beverages",
    },

    # Services
    "retail": {
        "classes": [35],
        "description": "Retail services, advertising, business management",
    },
    "consulting": {
        "classes": [35, 42],
        "description": "Business and technical consulting services",
    },
}


def get_nice_classes(profile_or_classes) -> list:
    """
    Resolve Nice classes from a profile name or class list.

    Args:
        profile_or_classes: Either:
            - str: Profile name (e.g., "camping_rv")
            - list: Direct list of class numbers (e.g., [9, 12])
            - None: Returns None (search all classes)

    Returns:
        List of Nice class numbers, or None for all classes

    Raises:
        ValueError: If profile name is not found
    """
    if profile_or_classes is None:
        return None

    if isinstance(profile_or_classes, str):
        profile = NICE_PROFILES.get(profile_or_classes)
        if profile is None:
            available = ', '.join(sorted(NICE_PROFILES.keys()))
            raise ValueError(
                f"Unknown profile '{profile_or_classes}'. "
                f"Available profiles: {available}"
            )
        return profile["classes"]

    if isinstance(profile_or_classes, (list, tuple)):
        return list(profile_or_classes)

    raise ValueError(f"Invalid nice_classes: {profile_or_classes}")


def list_profiles() -> dict:
    """List all available Nice class profiles with descriptions."""
    return {
        name: {
            "classes": p["classes"],
            "description": p["description"],
        }
        for name, p in NICE_PROFILES.items()
    }


# =============================================================================
# Application Configuration
# =============================================================================

@dataclass
class Config:
    """Application configuration"""
    euipo_client_id: Optional[str] = None
    euipo_client_secret: Optional[str] = None
    rapidapi_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    @property
    def has_euipo(self) -> bool:
        return bool(self.euipo_client_id and self.euipo_client_secret)

    @property
    def has_rapidapi(self) -> bool:
        return bool(self.rapidapi_key)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)


def load_env(env_path: Path = None) -> dict:
    """Load environment variables from .env file."""
    if env_path is None:
        # Look for .env in package parent directory
        env_path = Path(__file__).parent.parent / '.env'

    env_vars = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
                # Also set in os.environ for modules that use it directly
                os.environ.setdefault(key.strip(), value.strip())

    return env_vars


def get_config(env_path: Path = None) -> Config:
    """Get configuration from environment."""
    env = load_env(env_path)

    return Config(
        euipo_client_id=env.get('EUIPO_CLIENT_ID') or os.environ.get('EUIPO_CLIENT_ID'),
        euipo_client_secret=env.get('EUIPO_CLIENT_SECRET') or os.environ.get('EUIPO_CLIENT_SECRET'),
        rapidapi_key=env.get('RAPIDAPI_KEY') or os.environ.get('RAPIDAPI_KEY'),
        anthropic_api_key=env.get('ANTHROPIC_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'),
    )


# Singleton config
_config = None

def config() -> Config:
    """Get the singleton config instance."""
    global _config
    if _config is None:
        _config = get_config()
    return _config
