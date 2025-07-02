"""
Configuration management for the HAR project.
"""
import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data processing."""
    raw_data_path: str = 'data/raw/WISDM_ar_v1.1_raw.txt'
    processed_data_path: str = 'data/processed'
    augmented_data_path: str = 'data/augmented'
    time_steps: int = 90
    step_size: int = 45
    test_size: float = 0.2
    random_state: int = 42
    augmentation_enabled: bool = True
    augmentation_functions: list = field(default_factory=list)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_type: str = 'cnn_gru_attention'
    input_shape: tuple = (90, 3)
    num_classes: int = 6
    
    # CNN parameters
    cnn_filters: int = 64
    cnn_kernel_size: int = 3
    
    # RNN parameters
    rnn_units: int = 64
    rnn_type: str = 'gru'  # 'gru', 'lstm', 'bilstm'
    
    # Regularization
    dropout_rate: float = 0.5
    l2_regularization: float = 0.001
    
    # Attention parameters
    attention_heads: int = 2
    attention_key_dim: int = 16


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    
    # Callbacks
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Class weighting
    use_class_weights: bool = True
    
    # Checkpointing
    checkpoint_dir: str = 'experiments/checkpoints'
    save_best_only: bool = True


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = 'default'
    seed: int = 42


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default config.
        
    Returns:
        Config object with loaded settings.
    """
    if config_path is None:
        return Config()
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config objects from dictionary
    config = Config()
    
    if 'data' in config_dict:
        config.data = DataConfig(**config_dict['data'])
    if 'model' in config_dict:
        config.model = ModelConfig(**config_dict['model'])
    if 'training' in config_dict:
        config.training = TrainingConfig(**config_dict['training'])
    
    if 'experiment_name' in config_dict:
        config.experiment_name = config_dict['experiment_name']
    if 'seed' in config_dict:
        config.seed = config_dict['seed']
    
    return config


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save.
        save_path: Path where to save the configuration.
    """
    config_dict = {
        'experiment_name': config.experiment_name,
        'seed': config.seed,
        'data': {
            'raw_data_path': config.data.raw_data_path,
            'processed_data_path': config.data.processed_data_path,
            'augmented_data_path': config.data.augmented_data_path,
            'time_steps': config.data.time_steps,
            'step_size': config.data.step_size,
            'test_size': config.data.test_size,
            'random_state': config.data.random_state,
            'augmentation_enabled': config.data.augmentation_enabled,
            'augmentation_functions': config.data.augmentation_functions
        },
        'model': {
            'model_type': config.model.model_type,
            'input_shape': list(config.model.input_shape),
            'num_classes': config.model.num_classes,
            'cnn_filters': config.model.cnn_filters,
            'cnn_kernel_size': config.model.cnn_kernel_size,
            'rnn_units': config.model.rnn_units,
            'rnn_type': config.model.rnn_type,
            'dropout_rate': config.model.dropout_rate,
            'l2_regularization': config.model.l2_regularization,
            'attention_heads': config.model.attention_heads,
            'attention_key_dim': config.model.attention_key_dim
        },
        'training': {
            'batch_size': config.training.batch_size,
            'epochs': config.training.epochs,
            'learning_rate': config.training.learning_rate,
            'optimizer': config.training.optimizer,
            'early_stopping_patience': config.training.early_stopping_patience,
            'reduce_lr_patience': config.training.reduce_lr_patience,
            'reduce_lr_factor': config.training.reduce_lr_factor,
            'min_lr': config.training.min_lr,
            'use_class_weights': config.training.use_class_weights,
            'checkpoint_dir': config.training.checkpoint_dir,
            'save_best_only': config.training.save_best_only
        }
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)