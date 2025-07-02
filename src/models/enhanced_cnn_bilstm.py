"""
Enhanced CNN-Bidirectional LSTM model with attention for HAR.
"""
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    Bidirectional, LSTM, Dropout, Dense, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D, Add
)
from tensorflow.keras.models import Model

from .base_model import BaseHARModel


class EnhancedCNNBiLSTMModel(BaseHARModel):
    """
    Enhanced CNN-Bidirectional LSTM model with attention mechanism.
    
    Features:
    - CNN layers for feature extraction
    - Bidirectional LSTM with residual connections
    - Multi-head attention mechanism
    - Layer normalization for stability
    - Increased regularization
    """
    
    def build_model(self) -> Model:
        """
        Build the Enhanced CNN-BiLSTM model.
        
        Returns:
            Keras Model instance.
        """
        # Use stronger regularization for enhanced model
        l2_reg = max(self.config.l2_regularization, 0.001)
        regularizer = self.get_regularizer() if l2_reg > 0 else None
        
        # Increased dropout rate
        dropout_rate = min(self.config.dropout_rate * 1.2, 0.7)
        
        # Input layer
        inputs = Input(shape=self.config.input_shape)
        
        # CNN layers for feature extraction with L2 regularization
        x = Conv1D(
            filters=self.config.cnn_filters,
            kernel_size=self.config.cnn_kernel_size,
            activation='relu',
            kernel_regularizer=regularizer
        )(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # First Bidirectional LSTM layer
        x = Bidirectional(
            LSTM(
                units=self.config.rnn_units,
                return_sequences=True,
                kernel_regularizer=regularizer
            )
        )(x)
        x = Dropout(dropout_rate)(x)
        
        # Second Bidirectional LSTM layer
        x = Bidirectional(
            LSTM(
                units=self.config.rnn_units // 2,
                return_sequences=True,
                kernel_regularizer=regularizer
            )
        )(x)
        x = Dropout(dropout_rate)(x)
        
        # Store for residual connection
        residual = x
        
        # Third Bidirectional LSTM layer
        x = Bidirectional(
            LSTM(
                units=self.config.rnn_units // 2,
                return_sequences=True,
                kernel_regularizer=regularizer
            )
        )(x)
        x = Dropout(dropout_rate)(x)
        
        # Residual connection
        x = Add()([residual, x])
        x = LayerNormalization()(x)
        
        # Attention mechanism with residual connection
        attn_output = MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=dropout_rate * 0.5
        )(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        
        # Another residual connection
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Output layer
        outputs = Dense(
            units=self.config.num_classes,
            activation='softmax',
            kernel_regularizer=regularizer
        )(x)
        
        # Create and return the model
        model = Model(inputs=inputs, outputs=outputs, name='Enhanced_CNN_BiLSTM')
        return model