"""
Data augmentation techniques for time series data.
"""
import numpy as np
from typing import List, Tuple, Callable, Optional
from scipy.interpolate import CubicSpline
import logging

logger = logging.getLogger(__name__)


class TimeSeriesAugmenter:
    """
    Time series data augmentation for HAR.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the augmenter.
        
        Args:
            seed: Random seed for reproducibility.
        """
        np.random.seed(seed)
        self.augmentation_functions = {
            'add_noise': self.add_noise,
            'scale_data': self.scale_data,
            'time_mask': self.time_mask,
            'time_warp': self.time_warp,
            'window_slice': self.window_slice,
            'window_warp': self.window_warp,
            'magnitude_warp': self.magnitude_warp,
            'permute': self.permute,
            'random_sampling': self.random_sampling
        }
    
    @staticmethod
    def add_noise(data_sequence: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """Add random Gaussian noise to the data."""
        noise = np.random.randn(*data_sequence.shape) * noise_factor
        return data_sequence + noise
    
    @staticmethod
    def scale_data(data_sequence: np.ndarray, scaling_factor: float = 0.1) -> np.ndarray:
        """Randomly scale the data."""
        scaling = np.random.uniform(1 - scaling_factor, 1 + scaling_factor)
        return data_sequence * scaling
    
    @staticmethod
    def time_mask(data_sequence: np.ndarray, mask_ratio: float = 0.1) -> np.ndarray:
        """Randomly mask a portion of the data."""
        data_copy = data_sequence.copy()
        n_mask = int(data_sequence.shape[0] * mask_ratio)
        n_mask = max(n_mask, 1)
        mask_indices = np.random.choice(data_sequence.shape[0], n_mask, replace=False)
        data_copy[mask_indices] = 0
        return data_copy
    
    @staticmethod
    def time_warp(data_sequence: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
        """Apply time warping to the data sequence."""
        orig_steps = np.arange(data_sequence.shape[0])
        random_warp = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2))
        warp_steps = np.linspace(0, data_sequence.shape[0]-1, num=knot+2)
        warper = CubicSpline(warp_steps, random_warp)(orig_steps)
        warped_steps = (orig_steps * warper).astype(int)
        warped_steps = np.clip(warped_steps, 0, data_sequence.shape[0]-1)
        return data_sequence[warped_steps]
    
    @staticmethod
    def window_slice(data_sequence: np.ndarray, slice_ratio: float = 0.9) -> np.ndarray:
        """Randomly slice a window from the data sequence."""
        sequence_length = data_sequence.shape[0]
        slice_length = int(np.ceil(sequence_length * slice_ratio))
        start = np.random.randint(0, sequence_length - slice_length + 1)
        end = start + slice_length
        sliced_sequence = data_sequence[start:end]
        return np.resize(sliced_sequence, data_sequence.shape)
    
    def window_warp(self, data_sequence: np.ndarray, window_ratio: float = 0.1, sigma: float = 0.2) -> np.ndarray:
        """Apply time warping to a random window of the data sequence."""
        sequence_length = data_sequence.shape[0]
        window_length = int(np.ceil(sequence_length * window_ratio))
        start = np.random.randint(0, sequence_length - window_length + 1)
        end = start + window_length
        window = data_sequence[start:end]
        warped_window = self.time_warp(window, sigma)
        augmented_sequence = np.copy(data_sequence)
        augmented_sequence[start:end] = warped_window
        return augmented_sequence
    
    @staticmethod
    def magnitude_warp(data_sequence: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
        """Apply magnitude warping to the data sequence."""
        orig_steps = np.arange(data_sequence.shape[0])
        random_warp = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, data_sequence.shape[1]))
        warp_steps = np.linspace(0, data_sequence.shape[0]-1, num=knot+2)
        warper = np.array([CubicSpline(warp_steps, random_warp[:, dim])(orig_steps) 
                           for dim in range(data_sequence.shape[1])]).T
        return data_sequence * warper
    
    @staticmethod
    def permute(data_sequence: np.ndarray, n_perm: int = 4, min_seg_length: int = 10) -> np.ndarray:
        """Randomly permute segments of the data sequence."""
        sequence_length = data_sequence.shape[0]
        idx = np.random.permutation(n_perm)
        segments = []
        start = 0
        
        for i in range(n_perm):
            seg_length = np.random.randint(min_seg_length, sequence_length // n_perm + 1)
            end = start + seg_length
            if end > sequence_length or i == n_perm - 1:
                end = sequence_length
            segments.append(data_sequence[start:end])
            start = end
            if start >= sequence_length:
                break
        
        permuted_sequence = np.concatenate([segments[i] for i in idx[:len(segments)]], axis=0)
        return np.resize(permuted_sequence, data_sequence.shape)
    
    @staticmethod
    def random_sampling(data_sequence: np.ndarray, sampling_ratio: float = 0.8) -> np.ndarray:
        """Randomly sample and interpolate the data sequence."""
        sequence_length = data_sequence.shape[0]
        sampled_length = int(sequence_length * sampling_ratio)
        sampled_indices = sorted(np.random.choice(np.arange(sequence_length), sampled_length, replace=False))
        sampled_sequence = data_sequence[sampled_indices]
        
        augmented_sequence = np.array([
            np.interp(np.arange(sequence_length), sampled_indices, sampled_sequence[:, dim])
            for dim in range(data_sequence.shape[1])
        ]).T
        return augmented_sequence
    
    def augment_batch(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        augmentation_functions: Optional[List[str]] = None,
        augmentation_prob: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment a batch of data.
        
        Args:
            X: Input sequences of shape (batch_size, time_steps, features).
            y: Labels of shape (batch_size, num_classes).
            augmentation_functions: List of augmentation function names to apply.
            augmentation_prob: Probability of applying augmentation to each sample.
            
        Returns:
            Tuple of (augmented_X, augmented_y).
        """
        if augmentation_functions is None:
            augmentation_functions = list(self.augmentation_functions.keys())
        
        augmented_X = []
        augmented_y = []
        
        for i in range(len(X)):
            # Add original sample
            augmented_X.append(X[i])
            augmented_y.append(y[i])
            
            # Apply augmentation with probability
            if np.random.random() < augmentation_prob:
                # Randomly select an augmentation function
                aug_func_name = np.random.choice(augmentation_functions)
                aug_func = self.augmentation_functions[aug_func_name]
                
                try:
                    aug_sample = aug_func(X[i])
                    if aug_sample.shape == X[i].shape:
                        augmented_X.append(aug_sample)
                        augmented_y.append(y[i])
                except Exception as e:
                    logger.warning(f"Augmentation {aug_func_name} failed: {e}")
        
        return np.array(augmented_X), np.array(augmented_y)
    
    def augment_minority_classes(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        minority_class_codes: List[int],
        augmentation_functions: Optional[List[str]] = None,
        augmentations_per_sample: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment only minority classes to balance the dataset.
        
        Args:
            X_train: Training sequences.
            y_train: Training labels (one-hot encoded).
            minority_class_codes: List of minority class indices.
            augmentation_functions: List of augmentation functions to use.
            augmentations_per_sample: Number of augmentations per minority sample.
            
        Returns:
            Tuple of (augmented_X_train, augmented_y_train).
        """
        if augmentation_functions is None:
            augmentation_functions = list(self.augmentation_functions.keys())
        
        X_augmented = []
        y_augmented = []
        
        for i in range(len(X_train)):
            label = np.argmax(y_train[i])
            if label in minority_class_codes:
                # Apply multiple augmentations for minority classes
                for _ in range(augmentations_per_sample):
                    # Randomly select an augmentation function
                    aug_func_name = np.random.choice(augmentation_functions)
                    aug_func = self.augmentation_functions[aug_func_name]
                    
                    try:
                        aug_sample = aug_func(X_train[i])
                        if aug_sample.shape == X_train[i].shape:
                            X_augmented.append(aug_sample)
                            y_augmented.append(y_train[i])
                    except Exception as e:
                        logger.warning(f"Augmentation {aug_func_name} failed: {e}")
        
        if len(X_augmented) > 0:
            X_augmented = np.array(X_augmented)
            y_augmented = np.array(y_augmented)
            
            # Combine original and augmented data
            X_combined = np.concatenate((X_train, X_augmented), axis=0)
            y_combined = np.concatenate((y_train, y_augmented), axis=0)
            
            # Shuffle the combined data
            from sklearn.utils import shuffle
            X_combined, y_combined = shuffle(X_combined, y_combined, random_state=42)
            
            logger.info(f"Added {len(X_augmented)} augmented samples for minority classes")
            return X_combined, y_combined
        else:
            logger.warning("No augmented samples were created")
            return X_train, y_train