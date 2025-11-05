# BTC Prediction - Jupyter Notebooks

## 📓 Notebook Overview

This directory contains Jupyter notebooks for the BTC 'Sell' signal prediction project. Each notebook focuses on a specific aspect of the machine learning pipeline.

## 🚀 Quick Commands

### **Data Collection & Processing**
```bash
# 1. Collect BTC data from Bitstamp (2011-2025)
conda activate csml
python data_collection/scripts/btc_collector.py

# 2. Run main pipeline (feature engineering)
python -m data_processing.main_pipeline

# 3. Generate rolling window targets
python -m data_processing.target.rolling_window_target
```

### **Model Training & Tuning**
```bash
# 4. XGBoost hyperparameter tuning (A4 features)
python models/level0/tune_xgboost.py

# 5. XGBoost tuning with pruned features (top 24 features)
python models/level0/tune_xgboost_pruned.py
```

### **Complete Workflow**
```bash
# Step-by-step execution:
conda activate csml
python data_collection/scripts/btc_collector.py      # Collect data
python -m data_processing.main_pipeline              # Create features (A0-A5)
python -m data_processing.target.rolling_window_target  # Create target variable
python models/level0/tune_xgboost.py                # Tune A4 features
python models/level0/tune_xgboost_pruned.py         # Tune pruned features
```

## 📋 Notebook Structure

### **01_data_exploration.ipynb**
**Purpose**: Explore and understand the collected BTC data
- Load and examine data from different timeframes (4h, 1d, 1w)
- Data quality assessment
- Basic statistical analysis
- Price and volume visualization
- Data quality reports

### **02_feature_engineering.ipynb**
**Purpose**: Create technical indicators and features
- Technical indicators (RSI, MACD, Moving Averages, Ichimoku)
- Price-based features (OHLCV derivatives)
- Target variable creation (Sell signal detection)
- Feature selection and preprocessing

### **03_model_development.ipynb**
**Purpose**: Develop the stacking ensemble model
- Level 0 models (XGBoost, Random Forest, Logistic Regression)
- Level 1 meta-model (Logistic Regression)
- TimeSeriesSplit cross-validation
- Hyperparameter tuning and model evaluation

### **04_model_evaluation.ipynb**
**Purpose**: Evaluate the final model performance
- Final test set evaluation (2024-present)
- Performance metrics analysis
- Model interpretability with SHAP
- Backtesting and risk assessment

## 🚀 Getting Started

### Prerequisites
```bash
# Activate conda environment
conda activate csml

# Install additional dependencies if needed
pip install talib shap
```

### Running Notebooks
1. **Start Jupyter**: `jupyter lab` or `jupyter notebook`
2. **Navigate**: Open notebooks in order (01 → 02 → 03 → 04)
3. **Execute**: Run cells sequentially for best results

## 📊 Expected Workflow

```
01_data_exploration.ipynb
    ↓ (data understanding)
02_feature_engineering.ipynb
    ↓ (feature creation)
03_model_development.ipynb
    ↓ (model training)
04_model_evaluation.ipynb
    ↓ (final results)
```

## 🔧 Configuration

### Data Paths
- **Input**: `../data_collection/data/` (parquet files)
- **Output**: `../models/` (trained models)
- **Logs**: `../logs/` (training logs)

### Model Artifacts
- **Level 0 Models**: `../models/level0/`
- **Level 1 Model**: `../models/level1/`
- **Ensemble**: `../models/ensemble/`

## 📈 Success Metrics

### Primary Target
- **F1-Score**: ≥ 0.70 on final test set
- **ROC-AUC**: ≥ 0.75
- **Precision**: ≥ 0.65 (minimize false positives)

### Secondary Metrics
- **Recall**: ≥ 0.70 (catch most sell signals)
- **False Positive Rate**: ≤ 0.15
- **Model Interpretability**: SHAP analysis

## 🛠️ Troubleshooting

### Common Issues
1. **Import Errors**: Ensure conda environment is activated
2. **Data Not Found**: Run data collection first
3. **Memory Issues**: Use smaller data samples for testing
4. **Model Loading**: Check file paths and model artifacts

### Performance Tips
- Use `%time` magic for timing cell execution
- Monitor memory usage with `%memit`
- Save intermediate results to avoid recomputation
- Use `n_jobs=-1` for parallel processing

## 📝 Notes

- **Execution Order**: Run notebooks sequentially
- **Data Dependencies**: Ensure data collection is complete
- **Model Persistence**: Save models after training
- **Version Control**: Commit notebooks with outputs cleared
