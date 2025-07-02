"""
Data loading and preprocessing for WISDM dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from ..utils.constants import (
    COLUMN_NAMES, FEATURE_COLUMNS, ACTIVITY_MAPPING,
    TIME_STEPS, STEP_SIZE
)
from ..utils.config import DataConfig

logger = logging.getLogger(__name__)


class WISDMDataLoader:
    """
    Data loader for WISDM dataset.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize the data loader.
        
        Args:
            config: Data configuration object.
        """
        self.config = config
        self.data = None
        self.scaler = None
        
    def fix_raw_data(self, input_path: str, output_path: str) -> Dict[str, int]:
        """
        Fix the raw WISDM dataset by handling missing fields and formatting issues.
        
        Args:
            input_path: Path to raw data file.
            output_path: Path to save fixed data.
            
        Returns:
            Dictionary with statistics about fixed lines.
        """
        total_lines = 0
        fixed_lines = 0
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(input_path, 'r') as file, open(output_path, 'w') as fixed_file:
            for line in file:
                total_lines += 1
                # Clean the line
                line = line.strip().rstrip(';')
                if not line:
                    continue
                
                # Split into fields
                parts = line.split(',')
                
                # Check for missing or extra fields
                if len(parts) != 6:
                    fixed_lines += 1
                    if len(parts) < 6:
                        parts += ['0'] * (6 - len(parts))
                    else:
                        parts = parts[:6]
                
                # Validate and fix each field
                try:
                    parts[0] = str(int(parts[0]))  # User ID
                except ValueError:
                    fixed_lines += 1
                    parts[0] = "0"
                
                parts[1] = str(parts[1])  # Activity
                
                try:
                    parts[2] = str(int(parts[2]))  # Timestamp
                except ValueError:
                    fixed_lines += 1
                    parts[2] = "0"
                
                for i in range(3, 6):
                    try:
                        parts[i] = str(float(parts[i]))  # Accelerations
                    except ValueError:
                        fixed_lines += 1
                        parts[i] = "0.0"
                
                # Write the fixed line
                fixed_file.write(','.join(parts) + '\n')
        
        logger.info(f"Processed {total_lines} lines, fixed {fixed_lines} lines")
        return {"total_lines": total_lines, "fixed_lines": fixed_lines}
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to data file. If None, uses config path.
            
        Returns:
            Loaded DataFrame.
        """
        if file_path is None:
            file_path = self.config.raw_data_path
        
        # Check if we need to fix the raw data first
        if file_path.endswith('_raw.txt'):
            fixed_path = file_path.replace('_raw.txt', '_fixed.txt')
            if not Path(fixed_path).exists():
                logger.info("Fixing raw data...")
                self.fix_raw_data(file_path, fixed_path)
            file_path = fixed_path
        
        # Load the data
        self.data = pd.read_csv(file_path, header=None, names=COLUMN_NAMES)
        
        # Map activities to codes
        self.data['activity_code'] = self.data['activity'].map(ACTIVITY_MAPPING)
        
        logger.info(f"Loaded {len(self.data)} samples")
        return self.data
    
    def create_sequences(
        self, 
        data: Optional[pd.DataFrame] = None,
        time_steps: Optional[int] = None,
        step_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from time series data.
        
        Args:
            data: DataFrame with time series data. If None, uses self.data.
            time_steps: Length of each sequence.
            step_size: Step size for sliding window.
            
        Returns:
            Tuple of (sequences, labels).
        """
        if data is None:
            data = self.data
        if time_steps is None:
            time_steps = self.config.time_steps
        if step_size is None:
            step_size = self.config.step_size
        
        sequences = []
        labels = []
        
        def create_sequences_for_segment(segment_data, time_steps, step):
            segment_sequences = []
            segment_labels = []
            for start in range(0, len(segment_data) - time_steps, step):
                end = start + time_steps
                segment_sequences.append(segment_data[FEATURE_COLUMNS].values[start:end])
                segment_labels.append(segment_data['activity_code'].values[end - 1])
            return segment_sequences, segment_labels
        
        # Process data by user and activity to maintain continuity
        for user_id in data['user'].unique():
            user_data = data[data['user'] == user_id]
            for activity in user_data['activity'].unique():
                activity_data = user_data[user_data['activity'] == activity]
                if len(activity_data) >= time_steps:
                    seqs, labs = create_sequences_for_segment(
                        activity_data, time_steps, step_size
                    )
                    sequences.extend(seqs)
                    labels.extend(labs)
        
        X = np.array(sequences)
        y = np.array(labels)
        
        logger.info(f"Created {X.shape[0]} sequences of shape {X.shape[1:]}")
        return X, y
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize accelerometer data using StandardScaler.
        
        Args:
            data: DataFrame with accelerometer data.
            
        Returns:
            Normalized DataFrame.
        """
        from sklearn.preprocessing import StandardScaler
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            data[FEATURE_COLUMNS] = self.scaler.fit_transform(data[FEATURE_COLUMNS])
        else:
            data[FEATURE_COLUMNS] = self.scaler.transform(data[FEATURE_COLUMNS])
        
        return data
    
    def prepare_data(
        self, 
        normalize: bool = True,
        create_sequences: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, float]]:
        """
        Complete data preparation pipeline.
        
        Args:
            normalize: Whether to normalize the data.
            create_sequences: Whether to create sequences.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, class_weights_dict).
        """
        # Load data if not already loaded
        if self.data is None:
            self.load_data()
        
        # Normalize data
        if normalize:
            self.data = self.normalize_data(self.data)
        
        # Create sequences
        if create_sequences:
            X, y = self.create_sequences()
        else:
            X = self.data[FEATURE_COLUMNS].values
            y = self.data['activity_code'].values
        
        # One-hot encode labels
        from tensorflow.keras.utils import to_categorical
        y_encoded = to_categorical(y, num_classes=len(ACTIVITY_MAPPING))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=y_encoded
        )
        
        # Calculate class weights
        y_train_integers = np.argmax(y_train, axis=1)
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_integers),
            y=y_train_integers
        )
        class_weights_dict = {i: class_weights[i] for i in range(len(ACTIVITY_MAPPING))}
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"Class weights: {class_weights_dict}")
        
        return X_train, X_test, y_train, y_test, class_weights_dict
    
    def save_processed_data(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray,
        y_train: np.ndarray, 
        y_test: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Save processed data to disk.
        
        Args:
            X_train: Training sequences.
            X_test: Test sequences.
            y_train: Training labels.
            y_test: Test labels.
            save_path: Path to save the data.
        """
        if save_path is None:
            save_path = Path(self.config.processed_data_path) / 'sequences.npz'
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            save_path,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
        
        logger.info(f"Saved processed data to {save_path}")
    
    def load_processed_data(
        self, 
        load_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load processed data from disk.
        
        Args:
            load_path: Path to load the data from.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        if load_path is None:
            load_path = Path(self.config.processed_data_path) / 'sequences.npz'
        
        data = np.load(load_path)
        return data['X_train'], data['X_test'], data['y_train'], data['y_test']