# BTC 'Sell' Signal Prediction - Ablation Study

## 🎯 Project Overview

This project conducts a **systematic Ablation Study** to identify optimal Multi-Timeframe (MTF) and Historical Lag feature combinations for BTC 'Sell' signal prediction. The goal is to answer two key research questions through controlled experiments.

**Research Questions:**
- **RQ1 (Current timepoint):** What is the optimal MTF combination when expanding H4→D1→W1→M1?
- **RQ2 (Historical timepoint):** Do systematic historical lag features improve prediction?

## 🏗️ Architecture

**Stacking Ensemble for Fair Comparison:**
- **Level 0 Models:** XGBoost, Random Forest, Logistic Regression
- **Level 1 Meta-Model:** Logistic Regression
- **Validation:** TimeSeriesSplit (n_splits=5) with fallback to Rolling Window
- **Target:** Binary classification (Sell vs Rest) - 30-day -15% price drops
- **Experiments:** A0 (Baseline) → A1 (MTF-1) → A2 (MTF-2) → A3 (MTF-3) → A4_Pruned (Historical)

## 🚀 Key Features

- **Systematic Ablation:** Controlled experiments to isolate MTF and historical contributions
- **Modular Design:** Separate Python modules for each model component
- **Advanced Tuning:** Bayesian optimization for complex models, GridSearchCV for simple models
- **Time Series Aware:** Proper handling of temporal data with no data leakage
- **Feature Pruning:** XGBoost-based importance selection for A4_Pruned
- **Research Focus:** Data-driven answers to specific research questions

## 📁 Project Structure

```
btc_prediction/
├── data_collection/              # Step 1: Raw data collection
│   ├── scripts/
│   │   └── btc_collector.py
│   ├── config/
│   │   └── exchange_config.py
│   ├── data/                     # Raw OHLCV data (H4, D1, W1, M1)
│   └── logs/
├── config/                       # Global configuration
│   ├── settings.py               # Data paths, constants
│   └── model_params.py           # Hyperparameter grids
├── data_processing/              # Step 2: Feature engineering
│   ├── 01_data_collector.py      # Data collection wrapper
│   └── 02_feature_engineer.py    # Create A0-A4 feature sets
├── training/                     # Steps 4-5: Experiment execution
│   ├── 03_run_experiment.py      # Main experiment runner (takes exp_id)
│   ├── train_l0.py               # L0 model training utilities
│   └── train_l1.py               # L1 meta-model training utilities
├── evaluation/                   # Step 5: Results analysis
│   └── 04_evaluate_results.py    # Analyze experiment_results.csv
├── models/                       # Step 3: Model implementations
│   ├── level0/
│   │   ├── xgboost_model.py
│   │   ├── random_forest_model.py
│   │   └── logistic_regression_model.py
│   ├── level1/
│   │   └── meta_model.py
│   └── ensemble/
│       └── stacking_ensemble.py
├── utils/                        # Shared utilities
│   ├── data_utils.py
│   ├── cv_utils.py               # TimeSeriesSplit utilities
│   └── evaluation_utils.py
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_results_analysis.ipynb
├── logs/                         # Training logs and artifacts
│   ├── experiment_results.csv    # Main results table
│   └── models/                   # Saved model artifacts
├── features/                     # Step 2 output: Feature sets
│   ├── A0.parquet
│   ├── A1.parquet
│   ├── A2.parquet
│   ├── A3.parquet
│   ├── A4.parquet
│   ├── A4_Pruned.parquet
│   └── y.parquet
├── requirements.txt
├── plan.md                       # Implementation plan
└── README.md                     # This file
```

## 🔬 Ablation Study Workflow

1. **Data Collection (Step 1):** Collect H4, D1, W1, M1 timeframes (2020-03-01 to 2025-10-19)
2. **Feature Engineering (Step 2):** Create 5 feature sets (A0→A4)
3. **Environment Setup (Step 3):** Build modular codebase
4. **Experiment Loop (Step 4):** Run A0→A1→A2→A3→A4_Pruned sequentially
5. **Analysis (Step 5):** Answer RQ1 (MTF) and RQ2 (Historical lags)

## 🛠️ Technical Stack

- **Data Collection:** ccxt
- **ML Libraries:** scikit-learn, xgboost, scikit-optimize
- **Interpretability:** shap
- **Development:** Python 3.13, Jupyter
- **Environment:** Conda environment "csml"

## 🚀 Getting Started

### Environment Setup
```bash
# Create conda environment with Python 3.13
conda create -n csml python=3.13
conda activate csml

# Install requirements
pip install -r requirements.txt
```

### Running Ablation Study

```bash
# Step 1: Collect data (if not done)
python data_collection/scripts/btc_collector.py

# Step 2: Create feature sets
python data_processing/02_feature_engineer.py

# Step 3: Run ablation experiments
python training/03_run_experiment.py --exp_id A0
python training/03_run_experiment.py --exp_id A1
python training/03_run_experiment.py --exp_id A2
python training/03_run_experiment.py --exp_id A3

# Step 4: Feature pruning and final experiment
python training/03_run_experiment.py --prune_a4  # Creates A4_Pruned.parquet
python training/03_run_experiment.py --exp_id A4_Pruned

# Step 5: Analyze results
python evaluation/04_evaluate_results.py
```

## 📈 Results

*Results will be documented here after model training and evaluation.*

## 🎯 Conclusion

*Conclusion will be added after project completion.*


