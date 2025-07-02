"""HAR-WISDM-Advanced: Human Activity Recognition with Deep Learning."""

__version__ = "1.0.0"
__author__ = "Mahmoud Zaafan"

__all__ = ['WISDMDataLoader', 'TimeSeriesAugmenter']

# src/training/__init__.py
"""Training utilities for HAR models."""

from .trainer import Trainer

__all__ = ['Trainer']