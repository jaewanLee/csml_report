# BTC Prediction Project Overview

## Project Purpose
Develop a stacking ensemble model to predict BTC 'Sell' signals (30-day -15% price drops) using machine learning techniques.

## Tech Stack
- **Language:** Python 3.13
- **Environment:** Conda "csml"
- **Data Collection:** ccxt library for cryptocurrency data
- **ML Libraries:** scikit-learn, xgboost, scikit-optimize, shap
- **Data Processing:** pandas, numpy
- **Development:** Jupyter, black, pyright, pytest

## Code Style & Conventions
- **Type hints:** Used throughout (typing module)
- **Docstrings:** Google-style docstrings for classes and methods
- **Naming:** snake_case for variables/functions, PascalCase for classes
- **Error handling:** Comprehensive try-catch with logging
- **Modular design:** Separate classes for different functionalities

## Project Structure
```
btc_prediction/
├── data_collection/          # BTC data collection (ccxt)
├── data_processing/          # Feature engineering & preprocessing  
├── training/                # Model training scripts
├── evaluation/              # Final model evaluation
├── models/                  # Modular model classes
├── config/                  # Configuration files
├── utils/                   # Utility functions
├── notebooks/               # Jupyter notebooks
└── logs/                    # Training logs and artifacts
```

## Key Commands
- **Environment:** `conda activate csml`
- **Install:** `pip install -r requirements.txt`
- **Format:** `black .`
- **Lint:** `pyright`
- **Test:** `pytest`