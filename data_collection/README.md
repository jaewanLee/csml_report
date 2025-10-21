# Data Collection Module - Step 1 Implementation

This module handles the collection of BTC historical data using ccxt library as specified in the main project plan.

## 📋 Step 1 Tasks (from plan.md)

### **Objective:** Collect comprehensive BTC historical data
**Duration:** 1-2 days

#### **Tasks to Complete:**
- [x] **1.1** Set up ccxt library and configure exchange connections
- [x] **1.2** Create data collection folder structure (`data_collection/`)
- [x] **1.3** Collect OHLCV data for multiple timeframes (H4, D1, W1, M1)
- [x] **1.4** Data period: 2020-03-01 to 2025-10-19 (fixed end date for reproducibility) with training: 2020-05-12 to 2024-04-20, test: 2024-04-20 to 2025-10-19
- [x] **1.5** Implement data validation and quality checks
- [x] **1.6** Store data in Parquet format (better compression and performance)
- [x] **1.7** Create data collection pipeline with error handling and logging

#### **Deliverables:**
- [x] Raw BTC OHLCV datasets for all timeframes (Parquet format)
- [x] Data quality report
- [x] Modular data collection pipeline script
- [x] Configuration files for exchange settings

## 📁 Folder Structure
```
data_collection/
├── scripts/           # Data collection scripts
├── data/             # Raw data storage (Parquet format)
├── logs/             # Collection logs and error reports
└── config/           # Configuration files
```

## 🎯 Data Collection Strategy

### **Timeframes**
- **H4 (4-hour)**: Primary timeframe for technical analysis
- **D1 (Daily)**: Higher timeframe for trend analysis  
- **W1 (Weekly)**: Long-term trend identification
- **M1 (Monthly)**: Longest timeframe for major trend analysis

### **Data Periods (Timeframe-Specific)**
- **H4**: 2020-03-01 to 2025-10-19 (keep current for buffer)
- **D1**: 2020-01-08 to 2025-10-19 (2020-05-12 - 125 days for 120 MA)
- **W1**: 2019-02-11 to 2025-10-19 (2020-05-12 - 65 weeks for 60 MA)
- **M1**: 2018-03-12 to 2025-10-19 (2020-05-12 - 26 months for lagging span)
- **Training Data**: 2020-05-12 to 2024-04-20
- **Test Data**: 2024-04-20 to 2025-10-19

### **Storage Format**
- **Parquet files** for better compression and performance
- **Separate files** for each timeframe
- **Metadata** included for data quality tracking

## 🚀 Implementation Details

### **Task 1.1: Exchange Configuration**
- **Status:** ✅ Complete (exchange_config.py exists)
- **Exchanges:** Binance, Coinbase Pro, Kraken
- **Rate Limits:** Configured for API limits
- **Error Handling:** Connection retry logic

### **Task 1.2: Folder Structure**
- **Status:** ✅ Complete
- **Created:** scripts/, data/, logs/, config/ directories
- **Structure:** Matches plan.md specifications

### **Task 1.3: Data Collection Scripts**
- **Status:** 🔄 In Progress
- **Script:** btc_collector.py (288 lines)
- **Features:** Multi-timeframe collection, error handling
- **Next:** Test and validate collection process

### **Task 1.4: Data Periods**
- **Full Collection:** 2020-03-01 to 2025-10-19 (fixed end date for reproducibility)
- **Training:** 2020-05-12 to 2024-04-20
- **Test:** 2024-04-20 to 2025-09-19 (30-day buffer for label creation)
- **Label Buffer:** 2025-09-20 to 2025-10-19 (for final test labels)
- **Buffer:** 2020-03-01 to 2020-05-11 (for M1 lags t-1, t-2)
- **Status:** 🔄 Ready for execution

### **Task 1.5: Data Validation**
- **Status:** ⏳ Pending
- **Required:** Quality checks, completeness validation
- **Output:** Data quality report

### **Task 1.6: Parquet Storage**
- **Status:** ⏳ Pending
- **Format:** Parquet for compression and performance
- **Structure:** Separate files per timeframe

### **Task 1.7: Pipeline & Logging**
- **Status:** ⏳ Pending
- **Required:** Error handling, logging system
- **Output:** Modular pipeline script

## 🛠️ Setup
```bash
# Create conda environment with Python 3.13 (compatible with all packages)
conda create -n csml python=3.13

# Activate conda environment
conda activate csml

# Install dependencies (from project root)
pip install -r requirements.txt
```

**Note:** If you're using Python 3.14, please create a new environment with Python 3.13 as some packages don't support Python 3.14 yet.

## 📊 Usage
```python
from data_collection.scripts.btc_collector import BTCDataCollector

# Initialize collector
collector = BTCDataCollector()

# Collect all timeframes
collector.collect_all_timeframes()

# Collect specific timeframe
collector.collect_timeframe('1d', '2020-05-12', '2024-04-20')
```

## 📈 Progress Tracking
- **Completed:** 7/7 tasks (100%) ✅
- **In Progress:** 0/7 tasks (0%)
- **Pending:** 0/7 tasks (0%)
- **Status:** Step 1 Data Collection COMPLETED
- **Next Steps:** Proceed to Step 2 Feature Engineering

## 🎯 Threshold Analysis Results (Step 2.1)

**Status**: ✅ COMPLETED - See [results.md](../results.md) for detailed analysis and methodology

**Key Decision**: Selected -15% threshold for balanced class distribution and meaningful signals
