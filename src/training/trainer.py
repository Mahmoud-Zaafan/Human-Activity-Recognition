"""
Model training utilities for HAR.
"""
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger, Callback
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from ..utils.config import TrainingConfig
from ..models.base_model import BaseHARModel

logger = logging.getLogger(__name__)


class Trainer:
    """
    Model trainer for HAR models.
    """
    
    def __init__(
        self,
        model: BaseHARModel,
        config: TrainingConfig,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: HAR model instance.
            config: Training configuration.
            experiment_name: Name for the experiment.
        """
        self.model = model
        self.config = config
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.history = None
        
        # Setup directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for the experiment."""
        self.experiment_dir = Path('experiments') / self.experiment_name
        self.checkpoint_dir = self.experiment_dir / 'checkpoints'
        self.log_dir = self.experiment_dir / 'logs'
        self.results_dir = self.experiment_dir / 'results'
        
        for directory in [self.checkpoint_dir, self.log_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_callbacks(self) -> List[Callback]:
        """
        Get training callbacks.
        
        Returns:
            List of Keras callbacks.
        """
        callbacks = []
        
        # Early stopping
        if self.config.early_stopping_patience > 0:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Model checkpoint
        checkpoint_path = self.checkpoint_dir / 'best_model.h5'
        model_checkpoint = ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_loss',
            save_best_only=self.config.save_best_only,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # Learning rate reduction
        if self.config.reduce_lr_patience > 0:
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_lr,
                verbose=1
            )
            callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard = TensorBoard(
            log_dir=str(self.log_dir),
            histogram_freq=0,
            write_graph=True,
            write_images=False
        )
        callbacks.append(tensorboard)
        
        # CSV logger
        csv_logger = CSVLogger(
            str(self.log_dir / 'training_log.csv'),
            append=False
        )
        callbacks.append(csv_logger)
        
        return callbacks
    
    def compile_model(self, optimizer: Optional[Any] = None):
        """
        Compile the model.
        
        Args:
            optimizer: Optimizer instance. If None, uses config settings.
        """
        if optimizer is None:
            if self.config.optimizer == 'adam':
                optimizer = Adam(learning_rate=self.config.learning_rate)
            else:
                raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        self.model.compile_model(optimizer=optimizer)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weights: Optional[Dict[int, float]] = None,
        callbacks: Optional[List[Callback]] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training data.
            y_train: Training labels.
            X_val: Validation data.
            y_val: Validation labels.
            class_weights: Class weights dictionary.
            callbacks: Additional callbacks. If None, uses default callbacks.
            
        Returns:
            Training history.
        """
        # Compile model if not already compiled
        if self.model.model is None:
            self.compile_model()
        
        # Get callbacks
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        # Prepare class weights
        if class_weights is None and self.config.use_class_weights:
            logger.warning("Class weights enabled but not provided. Training without class weights.")
        
        # Log training info
        logger.info(f"Starting training for experiment: {self.experiment_name}")
        logger.info(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
        logger.info(f"Batch size: {self.config.batch_size}, Epochs: {self.config.epochs}")
        
        # Train the model
        keras_model = self.model.get_model()
        self.history = keras_model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=(X_val, y_val),
            class_weight=class_weights if self.config.use_class_weights else None,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        self.save_history()
        
        return self.history.history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            X_test: Test data.
            y_test: Test labels.
            batch_size: Batch size for evaluation.
            
        Returns:
            Tuple of (test_loss, test_accuracy).
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        keras_model = self.model.get_model()
        test_loss, test_accuracy = keras_model.evaluate(
            X_test, y_test,
            batch_size=batch_size,
            verbose=0
        )
        
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Save evaluation results
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy)
        }
        
        import json
        with open(self.results_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return test_loss, test_accuracy
    
    def predict(
        self,
        X: np.ndarray,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input data.
            batch_size: Batch size for prediction.
            
        Returns:
            Predictions array.
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        keras_model = self.model.get_model()
        predictions = keras_model.predict(X, batch_size=batch_size)
        
        return predictions
    
    def save_history(self):
        """Save training history to file."""
        if self.history is None:
            logger.warning("No training history to save.")
            return
        
        import json
        history_dict = {
            key: [float(v) for v in values] 
            for key, values in self.history.history.items()
        }
        
        with open(self.results_dir / 'training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)
    
    def save_model(self, filepath: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model. If None, saves to experiment directory.
        """
        if filepath is None:
            filepath = self.checkpoint_dir / 'final_model.h5'
        
        self.model.save(str(filepath))
        logger.info(f"Model saved to: {filepath}")