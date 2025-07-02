"""HAR-WISDM-Advanced: Human Activity Recognition with Deep Learning."""

__version__ = "1.0.0"
__author__ = "Mahmoud Zaafan"

__all__ = ['WISDMDataLoader', 'TimeSeriesAugmenter']

# src/models/__init__.py
"""Deep learning model architectures for HAR."""

from .base_model import BaseHARModel
from .cnn_gru_attention import CNNGRUAttentionModel
from .cnn_transformer import CNNTransformerModel
from .cnn_bilstm import CNNBiLSTMModel
from .enhanced_cnn_bilstm import EnhancedCNNBiLSTMModel

__all__ = [
    'BaseHARModel',
    'CNNGRUAttentionModel',
    'CNNTransformerModel',
    'CNNBiLSTMModel',
    'EnhancedCNNBiLSTMModel'
]