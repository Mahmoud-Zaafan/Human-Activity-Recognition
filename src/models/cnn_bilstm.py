"""
CNN-Bidirectional LSTM model for HAR.
"""
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    Bidirectional, LSTM, Dropout, Dense
)
from tensorflow.keras.models import Model

from .base_model import BaseHARModel


class CNNBiLSTMModel(BaseHARModel):
    """
    CNN-Bidirectional LSTM model for HAR.
    
    Architecture:
    - CNN layers for feature extraction
    - Bidirectional LSTM layers for temporal modeling
    - Dropout for regularization
    - Dense output layer
    """
    
    def build_model(self) -> Model:
        """
        Build the CNN-BiLSTM model.
        
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
        
        # Bidirectional LSTM layers
        x = Bidirectional(
            LSTM(
                units=self.config.rnn_units,
                return_sequences=True,
                kernel_regularizer=regularizer
            )
        )(x)
        x = Dropout(self.config.dropout_rate)(x)
        
        x = Bidirectional(
            LSTM(
                units=self.config.rnn_units // 2,
                kernel_regularizer=regularizer
            )
        )(x)
        x = Dropout(self.config.dropout_rate)(x)
        
        # Output layer for classification
        outputs = Dense(
            units=self.config.num_classes,
            activation='softmax',
            kernel_regularizer=regularizer
        )(x)
        
        # Create and return the model
        model = Model(inputs=inputs, outputs=outputs, name='CNN_BiLSTM')
        return model