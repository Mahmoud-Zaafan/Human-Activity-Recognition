# src/__init__.py
"""HAR-WISDM-Advanced: Human Activity Recognition with Deep Learning."""

__version__ = "1.0.0"
__author__ = "Mahmoud Zaafan"


# src/evaluation/__init__.py
"""Model evaluation and visualization tools."""

from .metrics import ModelEvaluator
from .visualization import ModelVisualizer

__all__ = ['ModelEvaluator', 'ModelVisualizer']