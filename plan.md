# BTC 'Sell' Signal Prediction - Implementation Plan

## 🎯 Project Overview
**Goal:** Develop a stacking ensemble model to predict BTC 'Sell' signals (30-day -10% price drops) using refined methods from `refine_methods.md`.

**Architecture:** Binary classification with stacking ensemble (XGBoost + Random Forest + Logistic Regression) + Meta-model

---

## 📋 Implementation Steps

### **Step 1: Data Collection (ccxt)**
**Duration:** 1-2 days  
**Objective:** Collect comprehensive BTC historical data

#### Tasks:
- [ ] **1.1** Set up ccxt library and configure exchange connections
- [ ] **1.2** Create data collection folder structure (`data_collection/`)
- [ ] **1.3** Collect OHLCV data for multiple timeframes (H4, D1, W1)
- [ ] **1.4** Data period: 2020-05-12 to 2024-04-20 (training) + 2024-04-20 to present (testing)
- [ ] **1.5** Implement data validation and quality checks
- [ ] **1.6** Store data in Parquet format (better compression and performance)
- [ ] **1.7** Create data collection pipeline with error handling and logging

#### Deliverables:
- Raw BTC OHLCV datasets for all timeframes (Parquet format)
- Data quality report
- Modular data collection pipeline script
- Configuration files for exchange settings

---

### **Step 2: Data Refinement**
**Duration:** 2-3 days  
**Objective:** Clean data and engineer features for model training

#### Tasks:
- [ ] **2.1** Data exploration and threshold analysis:
  - Analyze actual data to determine optimal threshold (-15% vs -10%)
  - Count Sell labels for different thresholds
  - Validate data quality and completeness
- [ ] **2.1.1** TimeSeriesSplit fold analysis (based on refine_methods.md):
  - Test different fold numbers (3, 5, 7, 10) for data distribution
  - Ensure each fold has sufficient Sell labels (minimum 50 per fold)
  - Validate temporal order and data leakage prevention
  - Use TimeSeriesSplit(n_splits=5) as baseline (refine_methods.md)
  - Determine optimal fold number based on data distribution
- [ ] **2.2** Data cleaning (missing values, outliers, duplicates)
- [ ] **2.3** Feature engineering - Technical indicators:
  - RSI (14-period) - based on CLOSE prices
  - MACD (12,26,9) - based on CLOSE prices
  - Moving Averages (7,14,20,60,120) - based on CLOSE prices
  - Ichimoku Cloud components - based on CLOSE prices
  - OHLCV derivatives - all calculations use CLOSE prices
- [ ] **2.4** Multi-timeframe feature alignment (H4, D1, W1)
- [ ] **2.5** Target variable creation:
  - Binary classification: Sell vs Rest
  - Threshold strategy: Start with -15%, adjust to -10% if insufficient labels
  - Method: Current bar close price → scan 30-day window for first threshold breach
  - Priority order: First detected threshold (chronological order)
  - Price calculation: 
    - SELL: (lowest_low - current_close) / current_close ≤ -15%
    - BUY: (highest_high - current_close) / current_close ≥ +5%
    - First detected threshold determines label (chronological order)
  - Example: Day 10: +5% detected → BUY, Day 20: -15% detected → Still BUY (first seen)
  - Example: Day 10: -15% detected → SELL, Day 20: +5% detected → Still SELL (first seen)
  - Handle class imbalance with class weights
- [ ] **2.6** Data leakage prevention:
  - Use completed candles only (e.g., 2020-01-01 00:00-04:00 candle available at 04:01)
  - No future data usage in feature engineering
  - Proper temporal alignment for multi-timeframe features
- [ ] **2.7** Train/Validation/Test split using TimeSeriesSplit (based on refine_methods.md):
  - Train/Validation: 2020-2024 (split using TimeSeriesSplit)
  - Final Test: 2024-present (never use during training)
  - Ensure temporal order: past → future prediction
  - No shuffle: maintain time series data order

#### Deliverables:
- Data exploration report with threshold analysis
- TimeSeriesSplit fold analysis report (optimal fold number)
- Cleaned and feature-engineered dataset
- Target variable distribution analysis
- Feature importance analysis
- Data preprocessing pipeline

---

### **Step 3: Model Development**
**Duration:** 3-4 days  
**Objective:** Build the stacking ensemble architecture with modular design

#### Tasks:
- [ ] **3.1** Create base model interface and common functionality:
  - Implement base model class with common methods (fit, predict, predict_proba, save, load)
  - Create hyperparameter tuning interface
  - Implement performance tracking and logging methods
- [ ] **3.2** Implement Level 0 model classes in `models/level0/`:
  - `xgboost_model.py`: XGBoost model class with configurable parameters
  - `random_forest_model.py`: Random Forest model class with configurable parameters
  - `logistic_regression_model.py`: Logistic Regression model class with configurable parameters
  - Each model inherits from base model class
- [ ] **3.3** Implement Level 1 meta-model class in `models/level1/`:
  - `meta_model.py`: Logistic Regression meta-model class
  - Input: [xgb_prob, rf_prob, lr_prob]
  - Output: Final sell probability
  - Inherits from base model class
- [ ] **3.4** Create ensemble wrapper in `models/ensemble/`:
  - `stacking_ensemble.py`: Two-level (L0, L1) architecture
  - **Final ensemble performance target** defined in Success Metrics section (e.g., F1 ≥ 0.70 on final Test Set)
  - Modular design for easy retraining and model replacement
- [ ] **3.5** Create TimeSeriesSplit cross-validation framework in `utils/`:
  - Use optimal fold number determined in Step 2.1.1
  - Implement TimeSeriesSplit(n_splits=5) as baseline
  - Implement fold-specific performance tracking
  - Ensure temporal order and no data leakage
- [ ] **3.6** Add comprehensive error handling and logging:
  - **Log all training attempts and cross-validation performance metrics for each model**
  - Implement early stopping for XGBoost (50 rounds)
  - Modular logging system for each model type
- [ ] **3.7** Model persistence and configuration:
  - Save Level 0 models using joblib (XGBoost compatible)
  - Save Level 1 model using pickle (sklearn compatible)
  - Store hyperparameters and performance metrics in JSON
  - Modular save/load functionality for each model

#### Deliverables:
- **Modular model architecture with separate .py files for each model**
- **Base model interface and common functionality**
- **Level 0 model modules (XGBoost, Random Forest, Logistic Regression)**
- **Level 1 meta-model module**
- **Stacking ensemble wrapper module**
- TimeSeriesSplit cross-validation framework with optimal fold number
- Meta-feature generation pipeline
- Model persistence system (Level 0 + Level 1 models)
- Comprehensive error handling and logging system
- Model architecture documentation with performance metrics

---

### **Step 4: Development Environment Setup**
**Duration:** 1 day  
**Objective:** Set up optimal development environment for model training

#### Tasks:
- [ ] **4.1** Local development environment (Mac M1 Pro):
  - Set up Python virtual environment
  - Install required packages (ccxt, pandas, scikit-learn, xgboost, etc.)
  - Configure Jupyter notebook environment
  - Set up project structure with proper imports
- [ ] **4.2** Git repository management:
  - Initialize git repository
  - Create .gitignore for Python/ML projects
  - Set up branch strategy (main, develop, feature branches)
  - Configure remote repository (GitHub/GitLab)
- [ ] **4.3** Modular project structure for model development:
  - Create folder structure for modular model development:
    ```
    btc_prediction/
    ├── data_collection/          # Already created
    ├── config/
    │   ├── settings.py           # Data paths, API keys, constants
    │   └── model_params.py       # All model hyperparameter grids
    ├── data_processing/
    │   ├── 01_data_collector.py  # ccxt data collection (Step 1)
    │   └── 02_feature_engineer.py # Technical indicators, target creation (Step 2)
    ├── training/
    │   ├── 03_train_l0.py        # L0 model tuning, meta-feature generation (Step 5)
    │   └── 04_train_l1.py        # L1 meta-model tuning (Step 6)
    ├── evaluation/
    │   └── 05_evaluate_final.py  # Final Test Set evaluation (Step 6.4)
    ├── models/
    │   ├── level0/
    │   │   ├── xgboost_model.py
    │   │   ├── random_forest_model.py
    │   │   └── logistic_regression_model.py
    │   ├── level1/
    │   │   └── meta_model.py
    │   └── ensemble/
    │       └── stacking_ensemble.py
    ├── utils/
    │   ├── data_utils.py
    │   ├── feature_utils.py
    │   └── evaluation_utils.py
    ├── main.py                   # Execute steps 1-5 in sequence
    ├── notebooks/
    │   ├── data_exploration.ipynb
    │   ├── model_development.ipynb
    │   └── evaluation.ipynb
    └── logs/
        ├── training_logs/
        └── model_artifacts/
    ```
- [ ] **4.4** Model module development:
  - Create base model class with common interface
  - Implement Level 0 model classes (XGBoost, Random Forest, Logistic Regression)
  - Implement Level 1 meta-model class
  - Create ensemble wrapper class
  - Set up proper import structure for modular usage
- [ ] **4.5** Basic monitoring and logging:
  - Set up logging configuration (Python logging module)
  - Create performance tracking (simple CSV/JSON logs)
  - Set up basic error handling and notifications
  - Configure model artifact storage (local filesystem)

#### Deliverables:
- Local development environment ready
- Git repository with proper structure
- **Modular project folder structure with model modules**
- **Base model classes and interfaces**
- **Level 0 and Level 1 model modules**
- Basic logging and monitoring setup

---

### **Step 5: Level 0 Models Tuning & Training**
**Duration:** 2-3 days  
**Objective:** Tune hyperparameters first, then train Level 0 models using modular architecture

#### Tasks:
- [ ] **5.1** Level 0 hyperparameter tuning using TimeSeriesSplit:
  - Use TimeSeriesSplit(n_splits=5) to find **optimal hyperparameters**
  - **Optimal criteria: Parameter combination with highest average F1 score from TimeSeriesSplit cross-validation** (no fixed threshold criteria)
  - XGBoost: Tune max_depth, learning_rate, n_estimators, scale_pos_weight
  - Random Forest: Tune n_estimators, max_depth, min_samples_split, class_weight
  - Logistic Regression: Tune C, penalty, solver, class_weight
  - **Store best_params (not best_estimator) for each model**
  - **Hyperparameter Tuning Framework (summary):**
    - **Menu (param_grid/param_space):** Define candidate values for each model's parameters.
    - **Method Selection (Hybrid Approach):**
      - **XGBoost & Random Forest:** Use **Bayesian Optimization** (complex models with many parameters)
      - **Logistic Regression:** Use **GridSearchCV** (simple model with few parameters)
    - **Judge:** Set `cv=TimeSeriesSplit(...)` and `scoring='f1'` to evaluate each combination on time series folds.
    - **Implementation Examples:**
      - **XGBoost (Bayesian):** `BayesSearchCV(xgb, param_space, cv=tscv, scoring='f1', n_iter=50)`
      - **Random Forest (Bayesian):** `BayesSearchCV(rf, param_space, cv=tscv, scoring='f1', n_iter=50)`
      - **Logistic Regression (Grid):** `GridSearchCV(pipeline, param_grid, cv=tscv, scoring='f1', n_jobs=-1)`
      - **Pipeline:** Use `Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(...))])` with keys like `model__C`, `model__penalty`, `model__solver`.
- [ ] **5.2** Generate meta-features using SAME TimeSeriesSplit:
  - Use SAME TimeSeriesSplit(n_splits=5) as in 5.1
  - **Create NEW models with best_params from Step 5.1 (not using best_estimator)**
  - **For each fold: Train new model on fold_train → Predict on fold_val**
  - Predict on validation folds to create meta-features: [xgb_prob, rf_prob, lr_prob]
  - Store meta-features and targets for Level 1 training
- [ ] **5.3** Final Level 0 model training:
  - **Create NEW models with best_params from Step 5.1**
  - **Train each L0 model on full Train/Validation dataset (2020-2024)**
  - **Use modular model classes from Step 3 for training and saving**
  - **Manually save models: `joblib.dump(model, 'final_xgb_model.pkl')`**
  - (These models will be used to generate predictions for Final Test Set)
- [ ] **5.4** Level 0 model evaluation:
  - Individual model performance analysis
  - Feature importance and SHAP analysis
  - **Cross-validation performance (from 5.1) and full-train-set performance (from 5.3) logging**

#### Deliverables:
- Optimized Level 0 models (XGBoost, Random Forest, Logistic Regression)
- Meta-features dataset for Level 1 training
- Level 0 model performance reports
- Feature importance and SHAP analysis
- Model artifacts and checkpoints

---

### **Step 6: Level 1 Meta-Model Tuning & Training**
**Duration:** 2-3 days  
**Objective:** Tune meta-model hyperparameters first, then train and evaluate complete stacking ensemble

#### Tasks:
- [ ] **6.1** Level 1 meta-model hyperparameter tuning:
  - **Use modular meta-model class from Step 3**
  - Use TimeSeriesSplit(n_splits=5) on meta-features from Step 5.2
  - **Store best_params (not best_estimator) for meta-model**
  - **Meta-Model Tuning Framework (summary):**
    - **Input:** Meta-features from Step 5.2 (`X_meta_train`, `y_meta_train`), i.e., L0 predictions `[xgb_prob, rf_prob, lr_prob]`.
    - **Judge:** Use `TimeSeriesSplit(n_splits=5)` on meta-features (same as L0) and `scoring='f1'`.
    - **Method Selection:** Use **GridSearchCV** (Logistic Regression meta-model has few parameters)
    - **Implementation:** `GridSearchCV(pipeline, param_grid, cv=tscv, scoring='f1', n_jobs=-1)`
    - **Pipeline:** `Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(...))])` with keys like `model__C`, `model__penalty`, `model__solver`.
    - **Note:** Meta-features are time-ordered; apply TimeSeriesSplit to avoid leakage.
- [ ] **6.2** Meta-model analysis and refinement loop:
  - **Meta-model coefficient analysis:** Check if any L0 model receives low weights (e.g., 0.05) → indicates low ensemble contribution
  - **L0 CV score review (from Step 5.1):** Cross-check if low-weight models also had low CV average scores
  - **Hypothesis and iteration:**
    - **Action 1:** Re-tune hyperparameters for underperforming L0 models → **Restart from Step 5.1**
    - **Action 2:** Remove persistently underperforming L0 models from ensemble (e.g., remove LR) → **Restart from Step 5.1**
    - **Action 3:** Replace L0 models (e.g., RF → LightGBM) → **Restart from Step 5.1**
    - **Action 4:** If L0 models are good, replace L1 meta-model (e.g., LR → LightGBM) → **Only repeat Step 6.1**
- [ ] **6.3** Final meta-model training:
  - **Create NEW meta-model with best_params from Step 6.1**
  - **Train meta-model on full meta-feature set (from Step 5.2)**
  - **Manually save meta-model: `joblib.dump(meta_model, 'final_meta_model.pkl')`**
  - Ensure no data leakage in final training
- [ ] **6.4** Complete ensemble evaluation:
  - **Evaluate on Final Test Set (2024-present) only once for final performance**
  - (Combine L0 models from Step 5.3 and L1 model from Step 6.3 for prediction)
  - Compare individual Level 0 models vs ensemble
  - Calculate comprehensive performance metrics (F1, Precision, Recall, ROC-AUC)
  - **Check if ensemble target performance (F1 ≥ 0.70) is achieved**
- [ ] **6.5** Final analysis and interpretability:
  - **Final meta-model coefficient analysis:** Identify which L0 model has the greatest influence on ensemble decisions
  - Ensemble weight analysis
  - SHAP analysis for complete ensemble
  - Walk-forward analysis and backtesting (on Test Set)
- [ ] **6.6** Performance comparison and reporting:
  - Individual model performance vs ensemble performance
  - Before vs after optimization comparison
  - Generate comprehensive final evaluation report

#### Deliverables:
- Trained and optimized Level 1 meta-model
- Complete stacking ensemble
- Final performance evaluation report
- Ensemble interpretability analysis
- Backtesting results and performance comparison
- Complete model artifacts and documentation

---

### **Step 7: Conclusion & Documentation**
**Duration:** 1-2 days  
**Objective:** Finalize results and create comprehensive documentation

#### Tasks:
- [ ] **7.1** Final performance evaluation
- [ ] **7.2** Compare with baseline models
- [ ] **7.3** Statistical significance testing
- [ ] **7.4** Create comprehensive results report
- [ ] **7.5** Document lessons learned and limitations
- [ ] **7.6** Prepare model deployment guidelines
- [ ] **7.7** Create user documentation

#### Deliverables:
- Final results report
- Model deployment package
- Comprehensive documentation
- Lessons learned document

---

## 🛠️ Technical Stack

### **Data & ML Libraries:**
- `ccxt` - Cryptocurrency data collection
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - ML algorithms and validation
- `xgboost` - Gradient boosting
- `shap` - Model interpretability
- `scikit-optimize` - Bayesian optimization (for BayesSearchCV)

### **Cloud & Infrastructure:**
- Cloud provider (AWS/GCP/Azure)
- `mlflow` - Experiment tracking
- `docker` - Containerization
- `kubernetes` - Orchestration (optional)

### **Development:**
- `python 3.8+`
- `jupyter` - Interactive development
- `black`, `pyright` - Code quality
- `pytest` - Testing

---

## 📊 Success Metrics

### **Model Performance:**
- **Primary:** F1-Score for 'Sell' class
- **Secondary:** Precision, Recall, ROC-AUC
- **Risk Management:** False positive rate for sell signals

### **Technical:**
- Model training time < 2 hours
- Prediction latency < 100ms
- Model size < 100MB

### **Business:**
- Sell signal accuracy > 70%
- False positive rate < 20%
- Model interpretability through SHAP

---

## 🚨 Risk Mitigation

### **Data Risks:**
- Missing data handling strategies
- Data quality validation
- Temporal consistency checks

### **Model Risks:**
- Overfitting prevention through proper validation
- Class imbalance handling
- Data leakage prevention

### **Infrastructure Risks:**
- Cloud cost monitoring
- Data backup strategies
- Model versioning

---

## 🎯 Next Immediate Action

**Ready to start with Step 1: Data Collection**

Would you like to begin with setting up the ccxt data collection pipeline for BTC historical data?

---

## 🔧 **Critical Implementation Notes**

### **Hyperparameter Tuning Method Selection**

#### **GridSearchCV vs Bayesian Optimization Comparison:**

| Aspect | GridSearchCV | RandomizedSearchCV | Bayesian Optimization |
|--------|--------------|-------------------|---------------------|
| **Speed** | Slowest (exhaustive) | Medium (random sampling) | **Fastest (smart search)** |
| **Efficiency** | Low (explores all combinations) | Medium (random exploration) | **High (learns from previous trials)** |
| **Best Results** | Guaranteed optimal (within grid) | Good (depends on sampling) | **Excellent (often finds better)** |
| **Computational Cost** | High | Medium | **Low** |
| **Implementation** | Simple | Simple | **Requires scikit-optimize** |
| **Use Case** | Small parameter spaces | Large parameter spaces | **Any parameter space** |

#### **Final Implementation Decision for BTC Prediction:**
- **Step 5.1 - Level 0 Models:**
  - **XGBoost:** Use **Bayesian Optimization** (`BayesSearchCV`, n_iter=50)
  - **Random Forest:** Use **Bayesian Optimization** (`BayesSearchCV`, n_iter=50)
  - **Logistic Regression:** Use **GridSearchCV** (few parameters, fast execution)
- **Step 6.1 - Level 1 Meta-Model:**
  - **Logistic Regression Meta-Model:** Use **GridSearchCV** (simple model, deterministic results)
- **Fallback Strategy:** Use **RandomizedSearchCV** if Bayesian optimization fails or takes too long

### **Proper Tuning → Training Workflow**
- **Step 5.1:** Use chosen method to find `best_params` (not `best_estimator`)
- **Step 5.2:** Create NEW models with `best_params` for meta-feature generation
- **Step 5.3:** Create NEW models with `best_params` for final training
- **Step 6.1:** Use chosen method to find meta-model `best_params`
- **Step 6.3:** Create NEW meta-model with `best_params` for final training

### **Model Saving Strategy**
- **GridSearchCV does NOT auto-save models** - only stores `best_params` and `best_estimator` in memory
- **Manual saving required:** `joblib.dump(model, 'final_xgb_model.pkl')`
- **Save after final training:** Use models trained on full dataset, not CV models
- **Modular approach:** Each model type has its own save/load methods

### **Stacking Ensemble Key Points**
- **Never use `best_estimator` directly** - it's trained on CV data
- **Always create fresh models** with `best_params` for each fold
- **Meta-features must be generated** using the same TimeSeriesSplit as tuning
- **Final models trained** on full dataset (2020-2024) for Test Set predictions

### **Bayesian Optimization Implementation Notes**
- **Install:** `pip install scikit-optimize`
- **Import:** `from skopt import BayesSearchCV`
- **Parameter Space:** Use `param_space` (dictionaries with ranges) instead of `param_grid`
- **XGBoost Example:** `param_space = {'max_depth': (3, 10), 'learning_rate': (0.01, 0.3), 'n_estimators': (100, 1000), 'scale_pos_weight': (1, 10)}`
- **Random Forest Example:** `param_space = {'n_estimators': (50, 500), 'max_depth': (3, 20), 'min_samples_split': (2, 20), 'min_samples_leaf': (1, 10)}`
- **Iterations:** Use `n_iter=50` for Level 0 models (XGBoost, Random Forest)
- **Acquisition Function:** Default 'EI' (Expected Improvement) works well for most cases

### **GridSearchCV Implementation Notes**
- **Logistic Regression Example:** `param_grid = {'model__C': [0.01, 0.1, 1, 10, 100], 'model__penalty': ['l1', 'l2'], 'model__solver': ['liblinear', 'saga']}`
- **Meta-Model Example:** Same as Logistic Regression above
- **Use Pipeline:** `Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(...))])`

---

## 🤔 **Need to Discuss Further**

### **Rolling Window Validation as Fallback Strategy**
- **Current Approach:** TimeSeriesSplit(n_splits=5) with expanding window
- **Fallback Consideration:** If results are poor, consider **Rolling Window Validation**
- **Rolling Window Benefits:**
  - **More realistic:** Simulates real-world trading where models are retrained periodically
  - **Market regime adaptation:** Better handles changing market conditions over time
  - **Reduced overfitting:** Prevents models from learning specific temporal patterns
  - **Robust validation:** Tests model performance across different time periods
- **Implementation Strategy:**
  - **If F1 < 0.60 on Test Set:** Implement rolling window validation
  - **Rolling Window Parameters:** 2-year training window, 6-month validation window, 1-month step
  - **Example:** Train on 2020-2022, validate on 2022-2022.5, then train on 2020.5-2022.5, validate on 2022.5-2023, etc.
  - **Meta-feature generation:** Use rolling window for both Level 0 and Level 1 models
- **Decision Criteria:**
  - **Keep TimeSeriesSplit if:** F1 ≥ 0.60, good generalization across folds
  - **Switch to Rolling Window if:** F1 < 0.60, poor performance on recent data, overfitting detected


