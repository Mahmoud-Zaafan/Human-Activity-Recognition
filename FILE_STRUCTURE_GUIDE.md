# Complete File Structure Guide - Where to Place Each File

## 📂 Project Directory Structure

Here's exactly where each file should be placed:

```
HAR-WISDM-Advanced/
│
├── 📁 src/                          # Main source code directory
│   ├── __init__.py                  # Package initializer
│   │
│   ├── 📁 data/                     # Data processing modules
│   │   ├── __init__.py
│   │   ├── loader.py                # WISDMDataLoader class
│   │   └── augmentation.py          # TimeSeriesAugmenter class
│   │
│   ├── 📁 models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── base_model.py            # BaseHARModel abstract class
│   │   ├── cnn_gru_attention.py    # CNNGRUAttentionModel class
│   │   ├── cnn_transformer.py      # CNNTransformerModel class
│   │   ├── cnn_bilstm.py           # CNNBiLSTMModel class
│   │   └── enhanced_cnn_bilstm.py  # EnhancedCNNBiLSTMModel class
│   │
│   ├── 📁 training/                 # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py               # (optional) Custom loss functions
│   │
│   ├── 📁 evaluation/               # Evaluation tools
│   │   ├── __init__.py
│   │   ├── metrics.py               # ModelEvaluator class
│   │   └── visualization.py         # ModelVisualizer class
│   │
│   └── 📁 utils/                    # Utilities
│       ├── __init__.py
│       ├── config.py                # Config classes and functions
│       └── constants.py               # Project constants
│
├── 📁 configs/                      # Configuration files
│   ├── default.yaml                 # Default configuration
│   └── 📁 model_configs/
│       ├── cnn_gru_attention.yaml
│       ├── cnn_transformer.yaml
│       ├── cnn_bilstm.yaml
│       └── enhanced_cnn_bilstm.yaml
│
├── 📁 scripts/                      # Executable scripts
│   ├── train.py                     # Main training script
│   ├── evaluate.py                  # Evaluation script
│   └── prepare_data.py              # Data preparation script
│
├── 📁 notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── (your original notebook).ipynb
│
├── 📁 data/                         # Data directory
│   ├── 📁 raw/
│   │   └── WISDM_ar_v1.1_raw.txt   # ⚠️ Place your raw data here!
│   ├── 📁 processed/                # Auto-created by scripts
│   └── 📁 augmented/                # Auto-created if needed
│
├── 📁 experiments/                  # Auto-created during training
│   └── (experiment folders will be created here automatically)
│
├── 📄 requirements.txt              # Python dependencies
├── 📄 setup.py                      # Package setup
├── 📄 README.md                     # Main documentation
├── 📄 USAGE_GUIDE.md                # Quick usage guide
├── 📄 FILE_STRUCTURE_GUIDE.md       # This file
├── 📄 Makefile                      # Automation commands
├── 📄 .gitignore                    # Git ignore rules
└── 📄 LICENSE                       # License file
```

## 🛠️ Step-by-Step Setup Instructions

### Step 1: Create the Main Directory
```bash
mkdir HAR-WISDM-Advanced
cd HAR-WISDM-Advanced
```

### Step 2: Create the Directory Structure
```bash
# Create all directories
mkdir -p src/{data,models,training,evaluation,utils}
mkdir -p configs/model_configs
mkdir -p scripts
mkdir -p notebooks
mkdir -p data/{raw,processed,augmented}
mkdir -p experiments
mkdir -p tests/{test_data,test_models}
mkdir -p docs
```

### Step 3: Create the Files

#### 3.1 Root Directory Files
Place these files in the root directory (`HAR-WISDM-Advanced/`):
- `requirements.txt`
- `setup.py`
- `README.md`
- `USAGE_GUIDE.md`
- `FILE_STRUCTURE_GUIDE.md` (this file)
- `Makefile`
- `.gitignore`

#### 3.2 Source Code Files (`src/`)

**In `src/`:**
- Create `__init__.py` with the main package init content

**In `src/data/`:**
- Create `__init__.py` with the data package init content
- Create `loader.py` with the WISDMDataLoader class
- Create `augmentation.py` with the TimeSeriesAugmenter class

**In `src/models/`:**
- Create `__init__.py` with the models package init content
- Create `base_model.py` with the BaseHARModel class
- Create `cnn_gru_attention.py` with the CNNGRUAttentionModel class
- Create `cnn_transformer.py` with the CNNTransformerModel class
- Create `cnn_bilstm.py` with the CNNBiLSTMModel class
- Create `enhanced_cnn_bilstm.py` with the EnhancedCNNBiLSTMModel class

**In `src/training/`:**
- Create `__init__.py` with the training package init content
- Create `trainer.py` with the Trainer class

**In `src/evaluation/`:**
- Create `__init__.py` with the evaluation package init content
- Create `metrics.py` with the ModelEvaluator class
- Create `visualization.py` with the ModelVisualizer class

**In `src/utils/`:**
- Create `__init__.py` with the utils package init content
- Create `config.py` with the configuration classes
- Create `constants.py` with project constants

#### 3.3 Configuration Files (`configs/`)
- Create `default.yaml` in `configs/`
- Create `enhanced_cnn_bilstm.yaml` in `configs/model_configs/`

#### 3.4 Script Files (`scripts/`)
- Create `train.py` with the training script
- Create `evaluate.py` with the evaluation script
- Create `prepare_data.py` with the data preparation script

### Step 4: Place Your Data
⚠️ **IMPORTANT**: Place your `WISDM_ar_v1.1_raw.txt` file in `data/raw/`

### Step 5: Create __init__.py Files
For the package init files, here's what goes in each:

```python
# src/__init__.py
"""HAR-WISDM-Advanced: Human Activity Recognition with Deep Learning."""
__version__ = "1.0.0"

# src/data/__init__.py
from .loader import WISDMDataLoader
from .augmentation import TimeSeriesAugmenter
__all__ = ['WISDMDataLoader', 'TimeSeriesAugmenter']

# src/models/__init__.py
from .base_model import BaseHARModel
from .cnn_gru_attention import CNNGRUAttentionModel
from .cnn_transformer import CNNTransformerModel
from .cnn_bilstm import CNNBiLSTMModel
from .enhanced_cnn_bilstm import EnhancedCNNBiLSTMModel
__all__ = ['BaseHARModel', 'CNNGRUAttentionModel', 'CNNTransformerModel', 
          'CNNBiLSTMModel', 'EnhancedCNNBiLSTMModel']

# src/training/__init__.py
from .trainer import Trainer
__all__ = ['Trainer']

# src/evaluation/__init__.py
from .metrics import ModelEvaluator
from .visualization import ModelVisualizer
__all__ = ['ModelEvaluator', 'ModelVisualizer']

# src/utils/__init__.py
from .config import Config, load_config, save_config
from .constants import ACTIVITY_MAPPING, ACTIVITY_NAMES, FEATURE_COLUMNS
__all__ = ['Config', 'load_config', 'save_config', 
          'ACTIVITY_MAPPING', 'ACTIVITY_NAMES', 'FEATURE_COLUMNS']
```

## 🚀 Quick Setup Commands

Once you have the structure in place, run these commands:

```bash
# 1. Navigate to project root
cd HAR-WISDM-Advanced

# 2. Create virtual environment and install dependencies
make setup
# OR manually:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# 3. Verify your data is in place
ls data/raw/
# Should show: WISDM_ar_v1.1_raw.txt

# 4. Prepare the data
make prepare

# 5. Train your first model
make train MODEL=enhanced_cnn_bilstm
```

## 📝 File Creation Tips

### Using a Text Editor
You can create all files using any text editor (VS Code, Sublime, nano, vim, etc.):

```bash
# Example with nano
nano src/utils/constants.py
# Paste the content, save with Ctrl+O, exit with Ctrl+X
```

### Using Command Line
You can also create files with echo or cat:

```bash
# Create a simple __init__.py
echo '"""Package initialization."""' > src/__init__.py

# Or use cat for multi-line content
cat > src/utils/__init__.py << 'EOF'
from .config import Config, load_config, save_config
from .constants import ACTIVITY_MAPPING, ACTIVITY_NAMES, FEATURE_COLUMNS
__all__ = ['Config', 'load_config', 'save_config', 
          'ACTIVITY_MAPPING', 'ACTIVITY_NAMES', 'FEATURE_COLUMNS']
EOF
```

## ✅ Verification Checklist

After setting up, verify your structure:

```bash
# Check directory structure
tree -L 3 HAR-WISDM-Advanced/

# Verify key files exist
ls src/models/
ls scripts/
ls configs/

# Check if data is in place
ls data/raw/

# Test imports (after activating venv and installing)
python -c "from src.models import EnhancedCNNBiLSTMModel; print('✓ Imports working!')"
```

## 🔧 Troubleshooting

### Common Issues:

1. **Import errors**: Make sure you run `pip install -e .` in the project root
2. **File not found**: Check you're in the correct directory
3. **Permission denied**: Use `chmod +x scripts/*.py` if needed
4. **Missing data**: Ensure `WISDM_ar_v1.1_raw.txt` is in `data/raw/`

### Need Help?
- Check file paths are exactly as shown
- Ensure all `__init__.py` files are created
- Verify you're in the virtual environment
- Check the README.md and USAGE_GUIDE.md for more details