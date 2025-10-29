# Data Processing Validation & Logging System

## Overview
Comprehensive validation and logging system for the BTC prediction data processing pipeline.

## Validation System

### Feature Validation (`data_processing/validation/feature_validation.py`)
- **Raw Features**: Logs statistics, allows NaNs (expected for indicators/lag windows)
- **Filtered Features**: Strict validation, no NaN/inf allowed
- **Consistency Check**: Validates alignment across feature sets

### Target Validation (`data_processing/validation/target_validation.py`)
- **Raw Target**: Logs distribution statistics
- **Filtered Target**: Strict validation, binary values only
- **Alignment Check**: Validates target-feature alignment

## Logging System (`data_processing/utils/logging_config.py`)

### Log Configuration
- **File Logging**: Rotating files (10MB x 3) in `logs/data_processing/`
- **Console Output**: Configurable console logging
- **Error Logging**: Separate error log file
- **Structured Logging**: Consistent format across all modules

### Log Levels
- **INFO**: General pipeline progress
- **WARNING**: Non-critical issues (NaNs in raw data)
- **ERROR**: Critical issues (NaNs in filtered data)
- **DEBUG**: Detailed debugging information

## Pipeline Stages with Logging

1. **Data Loading**: Load H4, D1, W1 data
2. **Technical Indicators**: Calculate all indicators with validation
3. **Timeframe Alignment**: Align D1, W1 with H4 base
4. **Lag Features**: Create historical lag features
5. **Feature Combination**: Combine A2 + lags → A3, A4, A5
6. **Data Filtering**: Filter to train/test period
7. **Validation**: Strict validation of filtered data
8. **Data Saving**: Save all feature sets

## Key Features

### Validation Strategy
- **Pre-filter**: Log-only validation (allows expected NaNs)
- **Post-filter**: Strict validation (raises errors on issues)
- **Consistency**: Cross-feature-set validation

### Logging Features
- **Stage Tracking**: Start/end logging for each stage
- **Performance Metrics**: Execution time tracking
- **Data Statistics**: Shape, memory usage, NaN counts
- **Error Handling**: Comprehensive error logging with tracebacks

### Error Handling
- **Graceful Degradation**: Raw validation never raises
- **Strict Filtering**: Filtered validation raises on issues
- **Comprehensive Logging**: All errors logged with context

## Usage

```python
# Run the complete pipeline
python -m data_processing.main_pipeline

# Or run directly
python data_processing/main_pipeline.py
```

## Log Files
- `logs/data_processing/pipeline.log`: Main pipeline logs
- `logs/data_processing/errors.log`: Error-only logs
- Console output: Real-time progress monitoring