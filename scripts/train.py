"""
Main training script for HAR models.
"""
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import logging
from datetime import datetime

from src.utils.config import load_config, save_config
from src.utils.constants import ACTIVITY_NAMES
from src.data.loader import WISDMDataLoader
from src.data.augmentation import TimeSeriesAugmenter
from src.models.cnn_gru_attention import CNNGRUAttentionModel
from src.models.cnn_transformer import CNNTransformerModel
from src.models.cnn_bilstm import CNNBiLSTMModel
from src.models.enhanced_cnn_bilstm import EnhancedCNNBiLSTMModel
from src.training.trainer import Trainer
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualization import ModelVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_model(model_type: str, config):
    """Get model instance based on type."""
    model_classes = {
        'cnn_gru_attention': CNNGRUAttentionModel,
        'cnn_transformer': CNNTransformerModel,
        'cnn_bilstm': CNNBiLSTMModel,
        'enhanced_cnn_bilstm': EnhancedCNNBiLSTMModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_classes[model_type](config.model)


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config.model.model_type = args.model
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    # Set experiment name
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    else:
        config.experiment_name = f"{config.model.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Model type: {config.model.model_type}")
    
    # Save configuration
    experiment_dir = Path('experiments') / config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, experiment_dir / 'config.yaml')
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    data_loader = WISDMDataLoader(config.data)
    
    # Check if processed data exists
    processed_data_path = Path(config.data.processed_data_path) / 'sequences.npz'
    if processed_data_path.exists() and not args.reprocess:
        logger.info("Loading preprocessed data...")
        X_train, X_test, y_train, y_test = data_loader.load_processed_data()
        
        # Calculate class weights
        from sklearn.utils import class_weight
        y_train_integers = np.argmax(y_train, axis=1)
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_integers),
            y=y_train_integers
        )
        class_weights_dict = {i: class_weights[i] for i in range(len(ACTIVITY_NAMES))}
    else:
        logger.info("Processing data from scratch...")
        X_train, X_test, y_train, y_test, class_weights_dict = data_loader.prepare_data()
        
        # Save processed data
        data_loader.save_processed_data(X_train, X_test, y_train, y_test)
    
    logger.info(f"Data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Apply data augmentation if enabled
    if config.data.augmentation_enabled and not args.no_augmentation:
        logger.info("Applying data augmentation...")
        augmenter = TimeSeriesAugmenter(seed=config.seed)
        
        # Identify minority classes
        from collections import Counter
        y_train_integers = np.argmax(y_train, axis=1)
        class_counts = Counter(y_train_integers)
        mean_count = np.mean(list(class_counts.values()))
        minority_classes = [cls for cls, count in class_counts.items() if count < mean_count]
        
        logger.info(f"Minority classes: {[ACTIVITY_NAMES[i] for i in minority_classes]}")
        
        # Augment minority classes
        X_train, y_train = augmenter.augment_minority_classes(
            X_train, y_train,
            minority_class_codes=minority_classes,
            augmentation_functions=config.data.augmentation_functions or None,
            augmentations_per_sample=3
        )
        
        # Recalculate class weights
        y_train_integers = np.argmax(y_train, axis=1)
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_integers),
            y=y_train_integers
        )
        class_weights_dict = {i: class_weights[i] for i in range(len(ACTIVITY_NAMES))}
        
        logger.info(f"Augmented data shape: {X_train.shape}")
    
    # Create model
    logger.info(f"Creating {config.model.model_type} model...")
    model = get_model(config.model.model_type, config)
    model.summary()
    
    # Create trainer
    trainer = Trainer(model, config.training, config.experiment_name)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        X_train, y_train,
        X_test, y_test,
        class_weights=class_weights_dict if config.training.use_class_weights else None
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_loss, test_accuracy = trainer.evaluate(X_test, y_test)
    
    # Get predictions for detailed evaluation
    y_pred_proba = trainer.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate detailed metrics
    evaluator = ModelEvaluator(ACTIVITY_NAMES)
    metrics = evaluator.evaluate_predictions(y_true, y_pred, y_pred_proba)
    
    # Print evaluation summary
    evaluator.print_evaluation_summary(metrics)
    
    # Save metrics
    evaluator.save_metrics(metrics, experiment_dir / 'results' / 'metrics.json')
    
    # Create visualizations
    if not args.no_visualize:
        logger.info("Creating visualizations...")
        visualizer = ModelVisualizer(
            ACTIVITY_NAMES,
            save_dir=experiment_dir / 'results' / 'plots'
        )
        
        visualizer.create_evaluation_report(
            metrics,
            history=history,
            model_name=config.model.model_type
        )
    
    # Save final model
    trainer.save_model()
    
    logger.info(f"Experiment completed! Results saved to: {experiment_dir}")
    logger.info(f"Final test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HAR models")
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['cnn_gru_attention', 'cnn_transformer', 'cnn_bilstm', 'enhanced_cnn_bilstm'],
        help='Model type to train'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Name for the experiment'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for training'
    )
    parser.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip creating visualizations'
    )
    parser.add_argument(
        '--reprocess',
        action='store_true',
        help='Reprocess data even if cached version exists'
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)