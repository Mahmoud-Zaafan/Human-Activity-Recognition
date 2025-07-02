"""
Base model class for Human Activity Recognition models.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from ..utils.config import ModelConfig


class BaseHARModel(ABC):
    """
    Abstract base class for HAR models.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the base model.
        
        Args:
            config: Model configuration object.
        """
        self.config = config
        self.model = None
        
    @abstractmethod
    def build_model(self) -> Model:
        """
        Build and return the model architecture.
        
        Returns:
            Keras Model instance.
        """
        pass
    
    def compile_model(
        self, 
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        loss: str = 'categorical_crossentropy',
        metrics: list = ['accuracy']
    ):
        """
        Compile the model.
        
        Args:
            optimizer: Optimizer instance. If None, uses Adam with config learning rate.
            loss: Loss function name.
            metrics: List of metrics to track.
        """
        if self.model is None:
            self.model = self.build_model()
        
        if optimizer is None:
            optimizer = Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def get_model(self) -> Model:
        """
        Get the compiled model.
        
        Returns:
            Compiled Keras Model.
        """
        if self.model is None:
            self.model = self.build_model()
            self.compile_model()
        return self.model
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.model = self.build_model()
        self.model.summary()
    
    def save(self, filepath: str):
        """
        Save the model.
        
        Args:
            filepath: Path to save the model.
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        self.model.save(filepath)
    
    def load(self, filepath: str):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model.
        """
        self.model = tf.keras.models.load_model(filepath)
    
    def get_regularizer(self) -> Optional[tf.keras.regularizers.Regularizer]:
        """
        Get L2 regularizer if configured.
        
        Returns:
            L2 regularizer or None.
        """
        if self.config.l2_regularization > 0:
            return l2(self.config.l2_regularization)
        return None