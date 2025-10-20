# Data Collection Module - Step 1 Implementation

This module handles the collection of BTC historical data using ccxt library as specified in the main project plan.

## 📋 Step 1 Tasks (from plan.md)

### **Objective:** Collect comprehensive BTC historical data
**Duration:** 1-2 days

#### **Tasks to Complete:**
- [ ] **1.1** Set up ccxt library and configure exchange connections
- [ ] **1.2** Create data collection folder structure (`data_collection/`)
- [ ] **1.3** Collect OHLCV data for multiple timeframes (H4, D1, W1)
- [ ] **1.4** Data period: 2020-05-12 to 2024-04-20 (training) + 2024-04-20 to present (testing)
- [ ] **1.5** Implement data validation and quality checks
- [ ] **1.6** Store data in Parquet format (better compression and performance)
- [ ] **1.7** Create data collection pipeline with error handling and logging

#### **Deliverables:**
- [ ] Raw BTC OHLCV datasets for all timeframes (Parquet format)
- [ ] Data quality report
- [ ] Modular data collection pipeline script
- [ ] Configuration files for exchange settings

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

### **Data Periods**
- **Training Data**: 2020-05-12 to 2024-04-20
- **Test Data**: 2024-04-20 to present

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
- **Training:** 2020-05-12 to 2024-04-20
- **Test:** 2024-04-20 to present
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
- **Completed:** 2/7 tasks (28.6%)
- **In Progress:** 1/7 tasks (14.3%)
- **Pending:** 4/7 tasks (57.1%)
- **Next Steps:** Execute data collection and validate results
