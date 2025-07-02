# Makefile for HAR-WISDM-Advanced project

.PHONY: help setup clean prepare train evaluate test lint format

# Default target
help:
	@echo "Human Activity Recognition - Makefile Commands"
	@echo "============================================="
	@echo "setup      - Set up the development environment"
	@echo "prepare    - Prepare and preprocess the data"
	@echo "train      - Train a model (use MODEL=<model_type>)"
	@echo "evaluate   - Evaluate a trained model (use EXP=<experiment_name>)"
	@echo "test       - Run unit tests"
	@echo "lint       - Run code linting"
	@echo "format     - Format code with black"
	@echo "clean      - Clean up generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make train MODEL=enhanced_cnn_bilstm"
	@echo "  make evaluate EXP=enhanced_cnn_bilstm_20231210_120000"

# Setup development environment
setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -e .
	@echo "Setup complete! Activate virtual environment with: source venv/bin/activate"

# Prepare data
prepare:
	python scripts/prepare_data.py --create-sequences --visualize

# Train model
train:
ifdef MODEL
	python scripts/train.py --model $(MODEL)
else
	python scripts/train.py
endif

# Train all models
train-all:
	python scripts/train.py --model cnn_gru_attention
	python scripts/train.py --model cnn_transformer
	python scripts/train.py --model cnn_bilstm
	python scripts/train.py --model enhanced_cnn_bilstm

# Evaluate model
evaluate:
ifndef EXP
	@echo "Error: Please specify experiment name with EXP=<experiment_name>"
	@exit 1
endif
	python scripts/evaluate.py $(EXP) --analyze-errors

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html

# Lint code
lint:
	flake8 src/ scripts/ --max-line-length=100 --ignore=E203,W503
	mypy src/ scripts/ --ignore-missing-imports

# Format code
format:
	black src/ scripts/ tests/ --line-length=100

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	@echo "Cleanup complete!"

# Clean data files (use with caution)
clean-data:
	rm -rf data/processed/* data/augmented/*
	@echo "Data files cleaned!"

# Clean experiments (use with caution)
clean-experiments:
	@echo "This will delete all experiment results. Are you sure? [y/N]"
	@read -r response; \
	if [ "$$response" = "y" ]; then \
		rm -rf experiments/*; \
		echo "Experiments cleaned!"; \
	else \
		echo "Cancelled."; \
	fi

# Install development dependencies
dev-install:
	pip install -e ".[dev,notebook]"

# Start Jupyter notebook
notebook:
	jupyter notebook notebooks/

# Create new experiment config
new-config:
ifndef NAME
	@echo "Error: Please specify config name with NAME=<config_name>"
	@exit 1
endif
	cp configs/default.yaml configs/$(NAME).yaml
	@echo "Created new config: configs/$(NAME).yaml"

# Run tensorboard
tensorboard:
	tensorboard --logdir experiments/

# Quick test run (small epochs for testing)
test-run:
	python scripts/train.py --epochs 2 --experiment-name test_run
