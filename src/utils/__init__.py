"""HAR-WISDM-Advanced: Human Activity Recognition with Deep Learning."""

__version__ = "1.0.0"
__author__ = "Mahmoud Zaafan"

# src/utils/__init__.py
"""Utility modules for configuration and constants."""

from .config import Config, load_config, save_config
from .constants import ACTIVITY_MAPPING, ACTIVITY_NAMES, FEATURE_COLUMNS

__all__ = [
    'Config',
    'load_config',
    'save_config',
    'ACTIVITY_MAPPING',
    'ACTIVITY_NAMES',
    'FEATURE_COLUMNS'
]