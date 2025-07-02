"""
CNN-Transformer model for HAR.
"""
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    Dropout, Dense, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Add
)
from tensorflow.keras.models import Model

from .base_model import BaseHARModel


class CNNTransformerModel(BaseHARModel):
    """
    CNN-Transformer hybrid model for HAR.
    
    Architecture:
    - CNN layers for local feature extraction
    - Transformer encoder for global context
    - Global average pooling
    - Dense output layer
    """
    
    def transformer_encoder(
        self, 
        inputs, 
        head_size: int, 
        num_heads: int, 
        ff_dim: int, 
        dropout: float = 0.1
    ):
        """
        Transformer encoder block.
        
        Args:
            inputs: Input tensor.
            head_size: Size of each attention head.
            num_heads: Number of attention heads.
            ff_dim: Hidden layer size in feed forward network.
            dropout: Dropout rate.
            
        Returns:
            Output tensor.
        """
        # Normalization and Attention
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(
            key_dim=head_size, 
            num_heads=num_heads, 
            dropout=dropout
        )(x, x)
        x = Dropout(dropout)(x)
        res = Add()([x, inputs])  # Residual connection
        
        # Feed Forward Part
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Dense(ff_dim, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        x = Add()([x, res])  # Residual connection
        
        return x
    
    def build_model(self) -> Model:
        """
        Build the CNN-Transformer model.
        
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
        x = Dropout(self.config.dropout_rate)(x)
        
        # Prepare for Transformer Encoder
        x = Conv1D(
            filters=128,
            kernel_size=3,
            padding='same',
            kernel_regularizer=regularizer
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config.dropout_rate)(x)
        
        # Apply Transformer Encoder
        x = self.transformer_encoder(
            x,
            head_size=64,
            num_heads=4,
            ff_dim=128,
            dropout=self.config.dropout_rate
        )
        
        # Global average pooling and output
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(
            units=self.config.num_classes,
            activation='softmax',
            kernel_regularizer=regularizer
        )(x)
        
        # Create and return the model
        model = Model(inputs=inputs, outputs=outputs, name='CNN_Transformer')
        return model