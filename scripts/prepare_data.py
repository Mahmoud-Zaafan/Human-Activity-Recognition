"""
Script to prepare and preprocess WISDM data.
"""
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import logging
import numpy as np
from src.utils.config import load_config
from src.utils.constants import ACTIVITY_NAMES, FEATURE_COLUMNS
from src.data.loader import WISDMDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Main data preparation function."""
    # Load configuration
    config = load_config(args.config)
    
    logger.info("Starting data preparation...")
    
    # Create data loader
    data_loader = WISDMDataLoader(config.data)
    
    # Check if raw data exists
    raw_data_path = Path(config.data.raw_data_path)
    if not raw_data_path.exists():
        logger.error(f"Raw data file not found: {raw_data_path}")
        logger.error("Please download the WISDM dataset and place it in the data/raw/ directory")
        sys.exit(1)
    
    # Load and fix raw data
    logger.info("Loading and fixing raw data...")
    data = data_loader.load_data()
    
    # Save cleaned CSV
    cleaned_path = Path(config.data.processed_data_path) / 'WISDM_cleaned.csv'
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(cleaned_path, index=False)
    logger.info(f"Saved cleaned data to: {cleaned_path}")
    
    # Display data statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total samples: {len(data):,}")
    logger.info(f"Number of users: {data['user'].nunique()}")
    logger.info(f"Number of activities: {data['activity'].nunique()}")
    
    # Activity distribution
    logger.info("\nActivity Distribution:")
    activity_counts = data['activity'].value_counts()
    for activity, count in activity_counts.items():
        percentage = (count / len(data)) * 100
        logger.info(f"  {activity}: {count:,} samples ({percentage:.2f}%)")
    
    # Prepare sequences if requested
    if args.create_sequences:
        logger.info("\nCreating sequences...")
        X_train, X_test, y_train, y_test, class_weights = data_loader.prepare_data()
        
        # Save processed sequences
        save_path = Path(config.data.processed_data_path) / 'sequences.npz'
        data_loader.save_processed_data(X_train, X_test, y_train, y_test, save_path)
        
        logger.info(f"\nSequence Statistics:")
        logger.info(f"Training sequences: {X_train.shape}")
        logger.info(f"Test sequences: {X_test.shape}")
        logger.info(f"Sequence length: {X_train.shape[1]} time steps")
        logger.info(f"Features per time step: {X_train.shape[2]}")
        
        # Display class distribution in sequences
        y_train_classes = np.argmax(y_train, axis=1)
        logger.info("\nTraining Set Class Distribution:")
        unique, counts = np.unique(y_train_classes, return_counts=True)
        for class_idx, count in zip(unique, counts):
            percentage = (count / len(y_train_classes)) * 100
            logger.info(f"  {ACTIVITY_NAMES[class_idx]}: {count} sequences ({percentage:.2f}%)")
        
        logger.info(f"\nClass weights: {class_weights}")
    
    # Create visualization of data samples if requested
    if args.visualize:
        logger.info("\nCreating data visualizations...")
        import matplotlib.pyplot as plt
        
        # Create output directory
        viz_dir = Path('data/processed/visualizations')
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot sample data for each activity
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, activity in enumerate(ACTIVITY_NAMES):
            ax = axes[i]
            activity_data = data[data['activity'] == activity].head(1000)
            
            if len(activity_data) > 0:
                time_range = range(len(activity_data))
                ax.plot(time_range, activity_data['x'], label='X', alpha=0.7)
                ax.plot(time_range, activity_data['y'], label='Y', alpha=0.7)
                ax.plot(time_range, activity_data['z'], label='Z', alpha=0.7)
                ax.set_title(f'{activity} (n={len(activity_data)})')
                ax.set_xlabel('Time')
                ax.set_ylabel('Acceleration')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'activity_samples.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to: {viz_dir / 'activity_samples.png'}")
        
        # Plot acceleration distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, axis in enumerate(['x', 'y', 'z']):
            ax = axes[i]
            for activity in ACTIVITY_NAMES:
                activity_data = data[data['activity'] == activity][axis]
                ax.hist(activity_data, bins=50, alpha=0.5, label=activity, density=True)
            
            ax.set_title(f'{axis.upper()}-axis Acceleration Distribution')
            ax.set_xlabel('Acceleration')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'acceleration_distributions.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to: {viz_dir / 'acceleration_distributions.png'}")
    
    logger.info("\nData preparation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare WISDM data for training")
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--create-sequences',
        action='store_true',
        default=True,
        help='Create sequences for model training'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create data visualizations'
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        sys.exit(1)