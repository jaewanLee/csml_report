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

## 📈 Results

*Results will be documented here after model training and evaluation.*

## 🎯 Conclusion

*Conclusion will be added after project completion.*


