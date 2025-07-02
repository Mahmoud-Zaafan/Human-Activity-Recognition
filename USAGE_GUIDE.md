# HAR Project - Quick Usage Guide

## 🚀 Getting Started

### 1. Initial Setup

```bash
# Clone and enter the project
cd HAR-WISDM-Advanced

# Set up the environment (using Makefile)
make setup

# Or manually:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare The Data

Place the WISDM dataset (`WISDM_ar_v1.1_raw.txt`) in `data/raw/`, then:

```bash
# Prepare data with visualizations
make prepare

# Or manually:
python scripts/prepare_data.py --create-sequences --visualize
```

### 3. Train Models

#### Quick Training (using Makefile)

```bash
# Train default model
make train

# Train specific model
make train MODEL=enhanced_cnn_bilstm

# Train all models
make train-all
```

#### Manual Training with Options

```bash
# Basic training
python scripts/train.py

# Train specific model with custom settings
python scripts/train.py \
    --model enhanced_cnn_bilstm \
    --epochs 100 \
    --batch-size 128 \
    --experiment-name my_experiment

# Train with custom config
python scripts/train.py --config configs/model_configs/enhanced_cnn_bilstm.yaml

# Train without augmentation
python scripts/train.py --no-augmentation

# Train without creating visualizations
python scripts/train.py --no-visualize
```

### 4. Evaluate Models

```bash
# Evaluate with error analysis
make evaluate EXP=enhanced_cnn_bilstm_20231210_120000

# Or manually:
python scripts/evaluate.py enhanced_cnn_bilstm_20231210_120000 \
    --analyze-errors \
    --plot-samples 10
```

### 5. View Results

```bash
# Start TensorBoard
make tensorboard

# Open Jupyter notebooks
make notebook
```

## 📁 Key Directories

- **experiments/**: All training outputs
  - `{experiment_name}/checkpoints/`: Model files
  - `{experiment_name}/logs/`: Training logs
  - `{experiment_name}/results/`: Metrics and plots

- **data/**:
  - `raw/`: Original WISDM data
  - `processed/`: Cleaned data and sequences
  - `augmented/`: Augmented sequences

## ⚙️ Configuration

### Creating Custom Configurations

```bash
# Create new config from template
make new-config NAME=my_custom_config

# Edit the config
nano configs/my_custom_config.yaml
```

### Key Configuration Options

```yaml
# Model architecture
model:
  model_type: enhanced_cnn_bilstm  # or: cnn_gru_attention, cnn_transformer, cnn_bilstm
  dropout_rate: 0.6
  l2_regularization: 0.001

# Training settings
training:
  batch_size: 64
  epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 5

# Data augmentation
data:
  augmentation_enabled: true
  augmentation_functions:
    - time_warp
    - magnitude_warp
    - window_slice
```

## 🧪 Development

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test
```

### Quick Test Run

```bash
# Train for 2 epochs to test setup
make test-run
```

## 📊 Using the Modular Code

### Example: Custom Training Script

```python
from src.utils.config import load_config
from src.data.loader import WISDMDataLoader
from src.models.enhanced_cnn_bilstm import EnhancedCNNBiLSTMModel
from src.training.trainer import Trainer

# Load config
config = load_config('configs/default.yaml')

# Prepare data
data_loader = WISDMDataLoader(config.data)
X_train, X_test, y_train, y_test, class_weights = data_loader.prepare_data()

# Create model
model = EnhancedCNNBiLSTMModel(config.model)

# Train
trainer = Trainer(model, config.training)
history = trainer.train(X_train, y_train, X_test, y_test, class_weights)

# Evaluate
test_loss, test_accuracy = trainer.evaluate(X_test, y_test)
```

### Example: Custom Data Augmentation

```python
from src.data.augmentation import TimeSeriesAugmenter

# Create augmenter
augmenter = TimeSeriesAugmenter()

# Augment specific samples
augmented_X, augmented_y = augmenter.augment_batch(
    X_batch, y_batch,
    augmentation_functions=['time_warp', 'magnitude_warp'],
    augmentation_prob=0.5
)
```

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size in config
2. **Slow Training**: Disable augmentation or reduce epochs
3. **Poor Performance**: Check class weights, increase augmentation
4. **Missing Data**: Ensure WISDM data is in `data/raw/`

### Clean Up

```bash
# Clean Python cache files
make clean

# Clean processed data (careful!)
make clean-data

# Clean all experiments (very careful!)
make clean-experiments
```

## 📈 Model Comparison

| Command | Model | Expected Accuracy | Training Time |
|---------|-------|-------------------|---------------|
| `make train MODEL=cnn_gru_attention` | CNN-GRU-Attention | ~99.0% | Fast |
| `make train MODEL=cnn_transformer` | CNN-Transformer | ~98.2% | Medium |
| `make train MODEL=cnn_bilstm` | CNN-BiLSTM | ~99.0% | Medium |
| `make train MODEL=enhanced_cnn_bilstm` | Enhanced CNN-BiLSTM | ~99.3% | Slow |

## 🎯 Tips for Best Results

1. **Use Data Augmentation**: Especially for minority classes
2. **Monitor Training**: Use TensorBoard to track progress
3. **Experiment with Configs**: Try different hyperparameters
4. **Analyze Errors**: Use evaluation script with `--analyze-errors`
5. **Save Best Models**: Early stopping automatically saves best weights

Happy experimenting! 🚀