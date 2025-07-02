"""
CNN-GRU with Attention mechanism for HAR.
"""
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D, 
    GRU, Dropout, Dense, MultiHeadAttention, 
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model

from .base_model import BaseHARModel


class CNNGRUAttentionModel(BaseHARModel):
    """
    CNN-GRU model with multi-head attention for HAR.
    
    Architecture:
    - CNN layers for feature extraction
    - GRU layers for temporal modeling
    - Multi-head attention mechanism
    - Global average pooling
    - Dense output layer
    """
    
    def build_model(self) -> Model:
        """
        Build the CNN-GRU-Attention model.
        
        Returns:
            Keras Model instance.
        """
        regularizer = self.get_regularizer()
        
        # Input layer
        inputs = Input(shape=self.config.input_shape)
        
        # CNN layers for feature extraction
        x = Conv1D(
            filters=self.config.cnn_filters,
            kernel_size=self.config.cnn_kernel_size,
            activation='relu',
            kernel_regularizer=regularizer
        )(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # GRU layers for capturing temporal dependencies
        x = GRU(
            units=self.config.rnn_units,
            return_sequences=True,
            kernel_regularizer=regularizer
        )(x)
        x = Dropout(self.config.dropout_rate)(x)
        
        x = GRU(
            units=self.config.rnn_units // 2,
            return_sequences=True,
            kernel_regularizer=regularizer
        )(x)
        x = Dropout(self.config.dropout_rate)(x)
        
        # Attention mechanism to focus on important time steps
        attention_output = MultiHeadAttention(
            num_heads=self.config.attention_heads,
            key_dim=self.config.attention_key_dim,
            dropout=self.config.dropout_rate * 0.5
        )(x, x)
        
        # Global average pooling
        attention_output = GlobalAveragePooling1D()(attention_output)
        
        # Output layer for classification
        outputs = Dense(
            units=self.config.num_classes,
            activation='softmax',
            kernel_regularizer=regularizer
        )(attention_output)
        
        # Create and return the model
        model = Model(inputs=inputs, outputs=outputs, name='CNN_GRU_Attention')
        return model