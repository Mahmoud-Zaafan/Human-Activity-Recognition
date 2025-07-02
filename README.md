# Human Activity Recognition with Advanced Deep Learning Models

This project implements state-of-the-art deep learning models for Human Activity Recognition (HAR) using the WISDM dataset. The codebase features a modular architecture, advanced data augmentation techniques, and comprehensive evaluation tools.

## 🚀 Features

- **Multiple Model Architectures**:
  - CNN-GRU with Attention Mechanism
  - CNN-Transformer Hybrid
  - CNN-Bidirectional LSTM
  - Enhanced CNN-BiLSTM with Multi-Head Attention

- **Advanced Data Processing**:
  - Time series-specific augmentation techniques
  - Automatic class balancing
  - Efficient sequence generation with sliding windows

- **Comprehensive Evaluation**:
  - Detailed metrics (precision, recall, F1-score)
  - Confusion matrices and visualization tools
  - Per-class performance analysis

- **Modular Architecture**:
  - Clean separation of concerns
  - Easy to extend and customize
  - Configuration-driven experiments

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.10+
- See `requirements.txt` for complete list

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Mahmoud-Zaafan/Human-Activity-Recognition.git
cd Human-Activity-Recognition
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install the package in development mode
```bash
pip install -e .
```

## 📊 Dataset

Download the WISDM v1.1 dataset from the [WISDM Lab Website](https://www.cis.fordham.edu/wisdm/dataset.php) and place `WISDM_ar_v1.1_raw.txt` in the `data/raw/` directory.

The dataset contains accelerometer data for 6 activities:
- Walking
- Jogging
- Upstairs
- Downstairs
- Sitting
- Standing

## 🚄 Quick Start

### 1. Prepare the data
```bash
python scripts/prepare_data.py
```

### 2. Train a model
```bash
# Train with default configuration
python scripts/train.py

# Train a specific model
python scripts/train.py --model enhanced_cnn_bilstm --epochs 50

# Train with custom configuration
python scripts/train.py --config configs/model_configs/enhanced_cnn_bilstm.yaml
```

### 3. Evaluate the model
```bash
python scripts/evaluate.py --experiment-name your_experiment_name
```

## 📁 Project Structure

```
HAR-WISDM-Advanced/
├── src/
│   ├── data/              # Data loading and augmentation
│   ├── models/            # Model architectures
│   ├── training/          # Training utilities
│   ├── evaluation/        # Metrics and visualization
│   └── utils/             # Configuration and constants
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebook
└── data/                  # Dataset directory
```

## 🔧 Configuration

The project uses YAML configuration files. See `configs/default.yaml` for available options:

```yaml
data:
  time_steps: 90           # Sequence length
  step_size: 45            # Sliding window step
  augmentation_enabled: true

model:
  model_type: cnn_gru_attention
  dropout_rate: 0.5
  l2_regularization: 0.001

training:
  batch_size: 64
  epochs: 50
  learning_rate: 0.001
```

## 📈 Model Performance

| Model | Test Accuracy |
|-------|--------------|
| CNN-GRU-Attention | 99.17% |
| CNN-Transformer | 98.78% |
| CNN-BiLSTM | 99.44% |
| Enhanced CNN-BiLSTM | 99.56% |

## 🧪 Advanced Features

### Data Augmentation
The project includes 9 time series-specific augmentation techniques:
- Time Warping
- Window Slicing
- Magnitude Warping
- Random Permutation
- And more...

### Class Balancing
Automatic handling of imbalanced classes through:
- Class weight computation
- Targeted augmentation for minority classes
- Stratified train-test splits

### Experiment Tracking
Each experiment automatically saves:
- Model checkpoints
- Training history
- Evaluation metrics
- Visualizations
- Configuration used

## 📊 Visualization

The project generates comprehensive visualizations:
- Training/validation curves
- Confusion matrices
- Per-class performance metrics
- Precision-recall curves
- Sample predictions


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- WISDM Lab for providing the dataset
- TensorFlow/Keras team for the deep learning framework
- Contributors and researchers in the HAR field

## 📧 Contact

For questions or suggestions, please open an issue or contact [zaafan.info@gmail.com]

---

**Note**: This is a research/educational project. For production use, additional optimization and testing may be required.
