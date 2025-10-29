"""
BTC Prediction Project - Configuration Settings
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data_collection" / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = LOGS_DIR / "models"

# Data paths
FEATURE_SETS = {
    'A0': FEATURES_DIR / 'A0.parquet',
    'A1': FEATURES_DIR / 'A1.parquet',
    'A1_pruned': FEATURES_DIR / 'A1_pruned.parquet',
    'A2': FEATURES_DIR / 'A2.parquet',
    'A3': FEATURES_DIR / 'A3.parquet',
    'A4': FEATURES_DIR / 'A4.parquet',
    'A4_Pruned': FEATURES_DIR / 'A4_Pruned.parquet'
}

TARGET_FILE = FEATURES_DIR / 'y.parquet'
RESULTS_FILE = LOGS_DIR / 'experiment_results.csv'

# Data split configuration
TRAIN_START = '2020-05-12'
TRAIN_END = '2024-04-19'
TEST_START = '2024-04-21'
TEST_END = '2025-09-19'
# 2025-10-19T23:59:59Z

# TimeSeriesSplit configuration
N_SPLITS = 5
RANDOM_STATE = 42

# Model training configuration
CLASS_WEIGHTS = 'balanced'  # Handle class imbalance
N_JOBS = -1  # Use all available cores

# Experiment configuration
EXPERIMENT_IDS = ['A0', 'A1', 'A2', 'A3', 'A4_Pruned']

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Ensure directories exist
for directory in [LOGS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
