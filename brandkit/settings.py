#!/usr/bin/env python3
"""Package wrapper around the top-level settings loader."""

from settings import (
    load_app_config,
    get_setting,
    resolve_path,
    PROJECT_ROOT,
    APP_CONFIG_PATH,
)

__all__ = [
    "load_app_config",
    "get_setting",
    "resolve_path",
    "PROJECT_ROOT",
    "APP_CONFIG_PATH",
]
