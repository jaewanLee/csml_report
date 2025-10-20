# Data Collection Module

This module handles the collection of BTC historical data using ccxt library.

## Folder Structure
```
data_collection/
├── scripts/           # Data collection scripts
├── data/             # Raw data storage (Parquet format)
├── logs/             # Collection logs and error reports
└── config/           # Configuration files
```

## Data Collection Strategy

### Timeframes
- **H4 (4-hour)**: Primary timeframe for technical analysis
- **D1 (Daily)**: Higher timeframe for trend analysis  
- **W1 (Weekly)**: Long-term trend identification

### Data Periods
- **Training Data**: 2020-05-12 to 2024-04-20
- **Test Data**: 2024-04-20 to present

### Storage Format
- **Parquet files** for better compression and performance
- **Separate files** for each timeframe
- **Metadata** included for data quality tracking

## Usage
```python
from data_collection.scripts.btc_collector import BTCDataCollector

collector = BTCDataCollector()
collector.collect_all_timeframes()
```
