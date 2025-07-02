"""
Script to evaluate trained HAR models.
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
import json
import tensorflow as tf

from src.utils.config import load_config
from src.utils.constants import ACTIVITY_NAMES
from src.data.loader import WISDMDataLoader
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualization import ModelVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_experiment(experiment_name: str):
    """Load experiment configuration and model."""
    experiment_dir = Path('experiments') / experiment_name
    
    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory not found: {experiment_dir}")
    
    # Load configuration
    config_path = experiment_dir / 'config.yaml'
    if not config_path.exists():
        raise ValueError(f"Configuration file not found: {config_path}")
    
    config = load_config(str(config_path))
    
    # Find model checkpoint
    checkpoint_dir = experiment_dir / 'checkpoints'
    model_files = list(checkpoint_dir.glob('*.h5')) + list(checkpoint_dir.glob('*.keras'))
    
    if not model_files:
        raise ValueError(f"No model checkpoint found in: {checkpoint_dir}")
    
    # Use the most recent model file
    model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    
    return config, model_path, experiment_dir


def main(args):
    """Main evaluation function."""
    # Load experiment
    logger.info(f"Loading experiment: {args.experiment_name}")
    config, model_path, experiment_dir = load_experiment(args.experiment_name)
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()
    
    # Load data
    logger.info("Loading test data...")
    data_loader = WISDMDataLoader(config.data)
    
    if args.use_full_dataset:
        # Prepare data from scratch
        X_train, X_test, y_train, y_test, _ = data_loader.prepare_data()
    else:
        # Load preprocessed data
        processed_path = Path(config.data.processed_data_path) / 'sequences.npz'
        if not processed_path.exists():
            logger.error("Preprocessed data not found. Run prepare_data.py first.")
            sys.exit(1)
        
        _, X_test, _, y_test = data_loader.load_processed_data()
    
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=args.batch_size, verbose=1)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    logger.info("Making predictions...")
    y_pred_proba = model.predict(X_test, batch_size=args.batch_size, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate detailed metrics
    evaluator = ModelEvaluator(ACTIVITY_NAMES)
    metrics = evaluator.evaluate_predictions(y_true, y_pred, y_pred_proba)
    
    # Print evaluation summary
    evaluator.print_evaluation_summary(metrics)
    
    # Save evaluation results
    results_dir = experiment_dir / 'evaluation_results'
    results_dir.mkdir(exist_ok=True)
    
    # Save metrics
    evaluator.save_metrics(metrics, results_dir / 'metrics.json')
    
    # Save summary
    summary = {
        'experiment_name': args.experiment_name,
        'model_path': str(model_path),
        'test_samples': len(X_test),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'model_type': config.model.model_type,
        'evaluation_timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create visualizations
    if not args.no_visualize:
        logger.info("Creating visualizations...")
        visualizer = ModelVisualizer(ACTIVITY_NAMES, save_dir=results_dir / 'plots')
        
        # Load training history if available
        history_path = experiment_dir / 'results' / 'training_history.json'
        history = None
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
        
        # Create evaluation report
        visualizer.create_evaluation_report(
            metrics,
            history=history,
            model_name=f"{config.model.model_type}_evaluation"
        )
        
        # Plot sample predictions
        if args.plot_samples:
            num_samples = min(args.plot_samples, len(X_test))
            sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
            
            visualizer.plot_sample_predictions(
                X_test[sample_indices],
                y_true[sample_indices],
                y_pred[sample_indices],
                num_samples=num_samples,
                save_name='sample_predictions'
            )
    
    # Generate classification errors analysis
    if args.analyze_errors:
        logger.info("Analyzing classification errors...")
        
        # Find misclassified samples
        misclassified_mask = y_true != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]
        
        logger.info(f"\nTotal misclassifications: {len(misclassified_indices)}")
        
        # Analyze error patterns
        error_matrix = np.zeros((len(ACTIVITY_NAMES), len(ACTIVITY_NAMES)))
        for idx in misclassified_indices:
            true_class = y_true[idx]
            pred_class = y_pred[idx]
            error_matrix[true_class, pred_class] += 1
        
        # Save error analysis
        error_analysis = {
            'total_errors': int(len(misclassified_indices)),
            'error_rate': float(len(misclassified_indices) / len(y_true)),
            'error_matrix': error_matrix.tolist(),
            'most_confused_pairs': []
        }
        
        # Find most confused pairs
        for i in range(len(ACTIVITY_NAMES)):
            for j in range(len(ACTIVITY_NAMES)):
                if i != j and error_matrix[i, j] > 0:
                    error_analysis['most_confused_pairs'].append({
                        'true_class': ACTIVITY_NAMES[i],
                        'predicted_class': ACTIVITY_NAMES[j],
                        'count': int(error_matrix[i, j]),
                        'percentage': float(error_matrix[i, j] / np.sum(error_matrix) * 100)
                    })
        
        # Sort by count
        error_analysis['most_confused_pairs'].sort(key=lambda x: x['count'], reverse=True)
        
        with open(results_dir / 'error_analysis.json', 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        logger.info("\nTop 5 Most Confused Pairs:")
        for pair in error_analysis['most_confused_pairs'][:5]:
            logger.info(f"  {pair['true_class']} → {pair['predicted_class']}: "
                       f"{pair['count']} ({pair['percentage']:.2f}%)")
    
    logger.info(f"\nEvaluation completed! Results saved to: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained HAR models")
    
    parser.add_argument(
        'experiment_name',
        type=str,
        help='Name of the experiment to evaluate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--use-full-dataset',
        action='store_true',
        help='Reprocess the full dataset instead of using cached sequences'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip creating visualizations'
    )
    parser.add_argument(
        '--plot-samples',
        type=int,
        default=5,
        help='Number of sample predictions to plot (0 to disable)'
    )
    parser.add_argument(
        '--analyze-errors',
        action='store_true',
        help='Perform detailed error analysis'
    )
    
    args = parser.parse_args()
    
    # Add pandas import for timestamp
    import pandas as pd
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)