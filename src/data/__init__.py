"""HAR-WISDM-Advanced: Human Activity Recognition with Deep Learning."""

__version__ = "1.0.0"
__author__ = "Mahmoud Zaafan"

# src/data/__init__.py
"""Data loading and augmentation modules."""

from .loader import WISDMDataLoader
from .augmentation import TimeSeriesAugmenter

__all__ = ['WISDMDataLoader', 'TimeSeriesAugmenter']