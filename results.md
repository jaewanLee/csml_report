# BTC 'Sell' Signal Prediction - Results Documentation

## 📊 Step 1: Data Collection Results ✅ COMPLETED

### **Data Collection Summary**
- **Collection Period**: 2020-03-01 to 2025-10-19 (fixed end date for reproducibility)
- **Total Records Collected**: 14,785 records across all timeframes
- **Storage Format**: Parquet files with data validation and quality reports

### **Timeframe Breakdown**
- **H4 (4h)**: 12,361 records collected
- **D1 (1d)**: 2,061 records collected  
- **W1 (1w)**: 295 records collected
- **M1 (1M)**: 68 records collected

### **Data Quality Validation**
- ✅ No missing data gaps detected
- ✅ Price anomaly validation passed
- ✅ Duplicate detection completed
- ✅ Temporal consistency verified

---

## 📈 Step 2: Feature Engineering & Threshold Analysis Results ✅ COMPLETED

### **Threshold Analysis Methodology**
- **Analysis Period**: 4H BTC data from 2020-05-12 to 2024-04-19 (training period)
- **Method**: 30-day lookahead window to find lowest LOW price
- **Calculation**: Percentage drop = (lowest_low - current_close) / current_close
- **Thresholds Tested**: -10%, -15%, -20%

### **Threshold Analysis Results**
| Threshold | Sell Signals | Percentage | Assessment |
|-----------|--------------|------------|------------|
| -10% | 3,778 | 43.79% | ❌ Too noisy, too many false positives |
| **-15%** | **2,526** | **29.28%** | ✅ **OPTIMAL BALANCE** |
| -20% | 1,479 | 17.14% | ❌ Too restrictive, misses opportunities |

### **Decision Rationale for -15% Threshold**
1. **Balanced Class Distribution**: 29.28% sell vs 70.72% non-sell (optimal for ML training)
2. **Sufficient Training Data**: 2,526 sell signals provide adequate examples for learning
3. **Meaningful Signals**: -15% drops represent significant price movements worth predicting
4. **Noise Reduction**: Prevents the 43.79% noise from -10% threshold
5. **Actionable Threshold**: -15% drops are substantial enough for trading decisions

### **Data Split Validation**
- **Training Data**: 2020-05-12 to 2024-04-19 (8,630 bars, 2,526 sell signals)
- **Test Data**: 2024-04-20 to 2025-09-19 (30-day buffer for label creation)
- **Label Buffer**: 2025-09-20 to 2025-10-19 (for final test labels)
- **Buffer Period**: 2020-03-01 to 2020-05-11 (for M1 lag features t-1, t-2)
- **No Data Overlap**: Clean separation between training and test sets

### **Key Insights**
- **Class Balance**: The -15% threshold provides an excellent balance between signal quality and quantity
- **Training Sufficiency**: 2,526 sell signals offer robust training data for the stacking ensemble
- **Temporal Integrity**: No data leakage between training and test periods
- **Reproducibility**: Fixed end date ensures consistent datasets across experiments

---

## 🔄 Step 3: Architecture & Environment Setup (Pending)

*Results will be documented as Step 3 is completed.*

---

## 🔄 Step 4: Ablation Study Experiment Loop (Pending)

*Results will be documented as Step 4 is completed.*

---

## 🔄 Step 5: Results Analysis & RQ Answers (Pending)

*Results will be documented as Step 5 is completed.*

---

## 🔄 Step 6: Conclusion & Documentation (Pending)

*Results will be documented as Step 6 is completed.*

---

## 📝 Notes

- **Last Updated**: 2025-01-21
- **Current Status**: Step 2 completed, proceeding to Step 3
- **Next Milestone**: Technical indicators calculation for all timeframes
- **Key Decision**: -15% threshold selected for balanced class distribution

---

*This document will be updated as each step is completed to maintain a comprehensive record of all project results and decisions.*
