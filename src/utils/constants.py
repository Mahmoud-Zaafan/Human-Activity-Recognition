"""
Constants for the Human Activity Recognition project.
"""

# Activity mappings
ACTIVITY_MAPPING = {
    'Walking': 0,
    'Jogging': 1,
    'Upstairs': 2,
    'Downstairs': 3,
    'Sitting': 4,
    'Standing': 5
}

ACTIVITY_NAMES = list(ACTIVITY_MAPPING.keys())

# Data columns
COLUMN_NAMES = ['user', 'activity', 'timestamp', 'x', 'y', 'z']
FEATURE_COLUMNS = ['x', 'y', 'z']

# Sequence parameters
TIME_STEPS = 90
STEP_SIZE = 45  # 50% overlap

# Model parameters
NUM_CLASSES = len(ACTIVITY_MAPPING)

# File paths
RAW_DATA_PATH = 'data/raw/WISDM_ar_v1.1_raw.txt'
FIXED_DATA_PATH = 'data/processed/WISDM_fixed.txt'
CLEANED_DATA_PATH = 'data/processed/WISDM_cleaned.csv'
PROCESSED_DATA_PATH = 'data/processed/sequences.npz'

# Training parameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 0.001

# Augmentation parameters
AUGMENTATION_FUNCTIONS = [
    'add_noise',
    'scale_data',
    'time_mask',
    'time_warp',
    'window_slice',
    'window_warp',
    'magnitude_warp',
    'permute',
    'random_sampling'
]