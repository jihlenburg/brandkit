#!/usr/bin/env python3
"""Standalone settings loader for BrandKit."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import os

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "brandkit" / "configs"
APP_CONFIG_PATH = CONFIG_DIR / "app.yaml"


@lru_cache(maxsize=1)
def load_app_config() -> dict:
    if not APP_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing app config: {APP_CONFIG_PATH}")
    data = yaml.safe_load(APP_CONFIG_PATH.read_text())
    return data or {}


def get_setting(path: str, default: Any = None) -> Any:
    """Get nested setting by dotted path."""
    data = load_app_config()
    current: Any = data
    for part in path.split('.'):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def resolve_path(value: str, base: Path | None = None) -> Path:
    """Resolve a path string relative to project root (unless absolute)."""
    if value is None:
        raise ValueError("path value is required")
    expanded = os.path.expanduser(str(value))
    path = Path(expanded)
    if not path.is_absolute():
        base = base or PROJECT_ROOT
        path = (base / path).resolve()
    return path


__all__ = [
    "load_app_config",
    "get_setting",
    "resolve_path",
    "PROJECT_ROOT",
    "APP_CONFIG_PATH",
]
