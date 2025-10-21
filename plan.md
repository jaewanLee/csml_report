# BTC 'Sell' Signal Prediction - Ablation Study Plan

## 🎯 Project Overview

**Goal:** Systematic Ablation Study to identify optimal Multi-Timeframe (MTF) and Historical Lag feature combinations for BTC 'Sell' signal prediction.

**Research Questions:**
- **RQ1 (Current timepoint):** What is the optimal MTF combination when expanding H4→D1→W1→M1?
- **RQ2 (Historical timepoint):** Do systematic historical lag features improve prediction?

**Architecture:** Stacking Ensemble (XGBoost + Random Forest + Logistic Regression) + Meta-model for fair comparison across all experiments.

---

## 📋 Implementation Steps

### **Step 1: Data Collection**
**Duration:** 1-2 days  
**Objective:** Collect comprehensive BTC historical data for all timeframes

#### Tasks:
- [x] **1.1** Set up ccxt library and configure exchange connections
- [x] **1.2** Collect OHLCV data for multiple timeframes (H4, D1, W1, **M1**)
- [x] **1.3** Data period: 2020-03-01 to 2025-10-19 (fixed end date for reproducibility) with training: 2020-05-12 to 2024-04-20, test: 2024-04-20 to 2025-10-19
- [x] **1.4** Implement data validation and quality checks
- [x] **1.5** Store data in Parquet format (better compression and performance)
- [x] **1.6** Create data collection pipeline with error handling and logging

#### Deliverables:
- [x] Raw BTC OHLCV datasets for all timeframes (Parquet format)
- [x] Data quality report
- [x] Modular data collection pipeline script

---

### **Step 2: Feature Engineering & Experiment Sets**
**Duration:** 3-4 days  
**Objective:** Create systematic Ablation Study feature sets to answer research questions

#### Tasks:
- [x] **2.1** Data exploration and threshold analysis:
  - ✅ Analyzed actual 4H data to determine optimal threshold
  - ✅ Counted Sell labels for different thresholds (-10%, -15%, -20%)
  - ✅ **DECISION: Selected -15% threshold** (see results.md for detailed analysis)
  - ✅ Validated data quality and completeness
- [ ] **2.2** Calculate technical indicators for ALL timeframes (H4, D1, W1, M1):
  - RSI (14-period) - based on CLOSE prices
  - MACD (12,26,9) - based on CLOSE prices
  - **Moving Averages (timeframe-specific):**
    - **H4:** (7,14,20,60,120) - all periods available
    - **D1:** (7,14,20,60,120) - all periods available  
    - **W1:** (7,14,20,60) - 120 MA requires 2.3 years of data
    - **M1:** (7,14,20) - 60/120 MA require 5-10 years of data
  - Ichimoku Cloud components - based on CLOSE prices
  - OHLCV derivatives - all calculations use CLOSE prices
- [ ] **2.3** Create Ablation Study feature sets (RQ1 - MTF contribution):
  - **A0 (Baseline):** H4 indicators (current time t only)
  - **A1 (MTF-1):** A0 + D1 indicators (current time t only)
  - **A2 (MTF-2):** A1 + W1 indicators (current time t only)
  - **A3 (MTF-3):** A2 + M1 indicators (current time t only)
- [ ] **2.4** Create lag features (RQ2 - Historical contribution):
  - H4 Lags: t-1 ~ t-6
  - D1 Lags: t-1 ~ t-7
  - W1 Lags: t-1 ~ t-4
  - M1 Lags: t-1 ~ t-2
- [ ] **2.5** Create A4 feature set:
  - **A4 (Historical Lags):** A3 + all lag features from 2.4
- [ ] **2.6** Target variable creation (Binary: Sell vs Rest):
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
- [ ] **2.7** Data leakage prevention:
  - Use completed candles only (e.g., 2020-01-01 00:00-04:00 candle available at 04:01)
  - No future data usage in feature engineering
  - Proper temporal alignment for multi-timeframe features
- [ ] **2.8** Data split:
  - Train/Validation: 2020-05-12 to 2024-04-20 (split using TimeSeriesSplit)
  - Final Test: 2024-04-20 to 2025-09-19 (30-day buffer for label creation)
  - Label Buffer: 2025-09-20 to 2025-10-19 (for final test labels)
  - Buffer Period: 2020-03-01 to 2020-05-11 (for M1 lag features t-1, t-2)
  - Ensure temporal order: past → future prediction
  - No shuffle: maintain time series data order

#### Deliverables:
- 5 separate feature files: A0.parquet, A1.parquet, A2.parquet, A3.parquet, A4.parquet
- Target variable file: y.parquet
- Data exploration report with threshold analysis
- Feature importance analysis


---

### **Step 3: Architecture & Environment Setup**
**Duration:** 2-3 days  
**Objective:** Build modular codebase for systematic experiment execution

#### Tasks:
- [ ] **3.1** Conda environment setup (Python 3.13, requirements.txt)
- [ ] **3.2** Create modular codebase structure:
  ```
  btc_prediction/
  ├── config/
  │   ├── settings.py               # Data paths, constants
  │   └── model_params.py           # Hyperparameter grids
  ├── data_processing/
  │   ├── 01_data_collector.py      # Data collection wrapper
  │   └── 02_feature_engineer.py  # Create A0-A4 feature sets
  ├── training/
  │   ├── 03_run_experiment.py      # Main experiment runner (takes exp_id)
  │   ├── train_l0.py               # L0 model training utilities
  │   └── train_l1.py               # L1 meta-model training utilities
  ├── evaluation/
  │   └── 04_evaluate_results.py    # Analyze experiment_results.csv
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
  │   ├── cv_utils.py               # TimeSeriesSplit utilities
  │   └── evaluation_utils.py
  ├── features/                     # Step 2 output: Feature sets
  │   ├── A0.parquet
  │   ├── A1.parquet
  │   ├── A2.parquet
  │   ├── A3.parquet
  │   ├── A4.parquet
  │   ├── A4_Pruned.parquet
  │   └── y.parquet
  ├── logs/                         # Training logs and artifacts
  │   ├── experiment_results.csv    # Main results table
  │   └── models/                   # Saved model artifacts
  └── notebooks/                    # Jupyter notebooks for exploration
      ├── 01_data_exploration.ipynb
      ├── 02_feature_engineering.ipynb
      └── 03_results_analysis.ipynb
  ```
- [ ] **3.3** Implement modular model classes (XGB, RF, LR, MetaLR)
- [ ] **3.4** Implement StackingEnsemble wrapper
- [ ] **3.5** Implement TimeSeriesSplit CV framework (n_splits=5 baseline)
- [ ] **3.6** **Create 03_run_experiment.py:**
  - Takes `exp_id` argument (e.g., "A1")
  - Loads corresponding feature set (A1.parquet)
  - Runs full pipeline: L0 tuning → Meta-feature generation → L1 tuning → Final training → Evaluation
  - Saves results to `experiment_results.csv`

#### Deliverables:
- Complete modular codebase structure
- Base model classes and interfaces
- Level 0 and Level 1 model modules
- Experiment runner script (03_run_experiment.py)
- TimeSeriesSplit cross-validation framework

---

### **Step 4: Ablation Study Experiment Loop**
**Duration:** 4-5 days  
**Objective:** Execute systematic experiments to answer research questions

#### Tasks:
- [ ] **4.1** Setup experiment logger:
  - Create `experiment_results.csv` with columns: Experiment_ID, Num_Features, Final_Test_F1, Final_Test_Precision, Final_Test_Recall, etc.
- [ ] **4.2** Main experiment loop - Part 1 (RQ1 - MTF contribution):
  - Run: `python training/03_run_experiment.py --exp_id A0`
  - Run: `python training/03_run_experiment.py --exp_id A1`
  - Run: `python training/03_run_experiment.py --exp_id A2`
  - Run: `python training/03_run_experiment.py --exp_id A3`
- [ ] **4.3** Checkpoint analysis:
  - Analyze A0~A3 results from experiment_results.csv
  - Validate pipeline is working correctly
  - If issues found, return to Step 2 or Step 3
- [ ] **4.4** A4 feature pruning:
  - Load A4.parquet (all features)
  - Run XGBoost-only L0 tuning on A4 (using Bayesian Optimization)
  - Analyze feature_importances_
  - Remove zero/low importance features
  - Create A4_Pruned.parquet with selected features
- [ ] **4.5** Main experiment loop - Part 2 (RQ2 - Historical lags contribution):
  - Run: `python training/03_run_experiment.py --exp_id A4_Pruned`

#### Deliverables:
- experiment_results.csv with all 5 experiments (A0, A1, A2, A3, A4_Pruned)
- Model artifacts (.pkl) for each experiment
- A4_Pruned.parquet (pruned feature set)

---

### **Step 5: Results Analysis & RQ Answers**
**Duration:** 1-2 days  
**Objective:** Analyze results and provide data-driven answers to research questions

#### Tasks:
- [ ] **5.1** Load and visualize experiment_results.csv
- [ ] **5.2** Answer RQ1 (MTF contribution):
  - Compare A0 vs A1 vs A2 vs A3 Final_Test_F1
  - Determine if each timeframe addition helps or adds noise
  - Identify optimal MTF combination
- [ ] **5.3** Answer RQ2 (Historical lags contribution):
  - Compare best from A0~A3 vs A4_Pruned Final_Test_F1
  - Assess if historical features provide meaningful improvement
- [ ] **5.4** Feature importance analysis:
  - Identify top 20 features from A4_Pruned
  - Analyze which timeframes and lags are most important
- [ ] **5.5** Statistical significance testing:
  - Compare performance differences between experiments
  - Validate that improvements are meaningful, not random

#### Deliverables:
- Analysis report answering RQ1 and RQ2
- Performance comparison charts
- Feature importance plots
- Statistical significance test results

---

### **Step 6: Conclusion & Documentation**
**Duration:** 1-2 days  
**Objective:** Finalize results and create comprehensive documentation

#### Tasks:
- [ ] **6.1** Final research results summary
- [ ] **6.2** Compare with baseline models
- [ ] **6.3** Document lessons learned and limitations
- [ ] **6.4** Prepare model deployment guidelines
- [ ] **6.5** Create user documentation
- [ ] **6.6** GitHub repository organization

#### Deliverables:
- Final research results report
- Model deployment package
- Comprehensive documentation
- Lessons learned document

---

## 🔧 **Critical Implementation Notes**

### **Hyperparameter Tuning Strategy**
- **L0 (XGBoost, Random Forest):** BayesSearchCV with n_iter=50
  - Param spaces: max_depth, learning_rate, n_estimators, scale_pos_weight
- **L0 (Logistic Regression):** GridSearchCV
  - Param grid: C, penalty, solver
- **L1 (Meta-LR):** GridSearchCV on meta-features

### **Tuning → Training Workflow**
1. Use BayesSearchCV/GridSearchCV to find `best_params` (NOT `best_estimator`)
2. Create NEW models with `best_params` for meta-feature generation (TimeSeriesSplit loop)
3. Create NEW models with `best_params` for final training (full Train/Val dataset)
4. Manually save: `joblib.dump(model, 'path/to/model.pkl')`

### **TimeSeriesSplit Implementation**
- n_splits=5 baseline (from refine_methods.md)
- Ensure temporal order, no shuffle
- Same split used for tuning and meta-feature generation
- Prevents data leakage

### **Data Leakage Prevention**
- Use completed candles only
- No future data in feature engineering
- Proper temporal alignment for MTF features
- Target calculation: current close → future LOW (not future close)

### **Stacking Ensemble Key Points**
- **Never use `best_estimator` directly** - it's trained on CV data
- **Always create fresh models** with `best_params` for each fold
- **Meta-features must be generated** using the same TimeSeriesSplit as tuning
- **Final models trained** on full dataset (2020-2024) for Test Set predictions

---

## 📊 Success Metrics

### **Model Performance:**
- **Primary:** F1-Score for 'Sell' class
- **Secondary:** Precision, Recall, ROC-AUC
- **Target:** Final model (A4_Pruned) achieves F1 ≥ 0.70 on test set

### **Research:**
- **Clear, data-driven answers to RQ1 and RQ2**
- **Statistical Significance:** Performance differences between experiments are meaningful
- **Feature Insights:** Identify which timeframes and lags contribute most

### **Technical:**
- Model training time < 2 hours per experiment
- Prediction latency < 100ms
- Model size < 100MB

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

### **Experiment Risks:**
- **Checkpoint at 4.3:** Validate pipeline before proceeding to A4
- **Feature Pruning:** Prevents A4 from being too high-dimensional
- **TimeSeriesSplit:** Prevents overfitting through proper validation
- **Fallback:** If all experiments show low performance, revisit target definition (Step 2.6)

---

## 🎯 Next Immediate Action

**Ready to start with Step 1: Data Collection**

The systematic Ablation Study approach will provide clear answers to both research questions through controlled experiments, ensuring fair comparison across all feature combinations.

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