# BTC 'Sell' Signal Prediction

## 🎯 Project Overview

This project develops a **stacking ensemble model** to predict BTC 'Sell' signals using machine learning techniques. The goal is to identify when BTC price is likely to drop by 15% or more within a 30-day window, helping traders make informed decisions.

## 🏗️ Architecture

**Binary Classification with Stacking Ensemble:**
- **Level 0 Models:** XGBoost, Random Forest, Logistic Regression
- **Level 1 Meta-Model:** Logistic Regression
- **Validation:** TimeSeriesSplit with fallback to Rolling Window
- **Target:** Predict BTC 'Sell' signals (30-day -15% price drops)

## 🚀 Key Features

- **Modular Design:** Separate Python modules for each model component
- **Advanced Tuning:** Bayesian optimization for complex models, GridSearchCV for simple models
- **Time Series Aware:** Proper handling of temporal data with no data leakage
- **Class Imbalance:** Handled using class weights and threshold strategies
- **Interpretability:** SHAP analysis for model explainability


## 🛠️ Technical Stack

- **Data Collection:** ccxt
- **ML Libraries:** scikit-learn, xgboost, scikit-optimize
- **Interpretability:** shap
- **Development:** Python 3.8+, Jupyter
- **Environment:** Conda environment "csml"

## 🚀 Getting Started

### Environment Setup
```bash
# Create conda environment with Python 3.13 (compatible with all packages)
conda create -n csml python=3.13

# Activate environment
conda activate csml

# Install requirements
pip install -r requirements.txt
```

**Note:** If you're using Python 3.14, please create a new environment with Python 3.13 as some packages don't support Python 3.14 yet.

### Project Execution
1. **Data Collection:** Run `01_data_collector.py` to collect BTC historical data
2. **Feature Engineering:** Run `02_feature_engineer.py` to create technical indicators
3. **Model Training:** Run `03_train_l0.py` and `04_train_l1.py` for model training
4. **Evaluation:** Run `05_evaluate_final.py` for final performance assessment

## 📈 Results

*Results will be documented here after model training and evaluation.*

## 🎯 Conclusion

*Conclusion will be added after project completion.*


