"""
Main data processing pipeline with comprehensive validation and logging.

This module orchestrates the entire data processing workflow:
1. Load raw data
2. Calculate technical indicators
3. Align timeframes
4. Create lag features
5. Combine all features
6. Filter data
7. Validate results
8. Save processed data
"""

from calendar import week
import sys
import time
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import data processing modules
from data_processing.config.config import TEST_END, TRAIN_START
from data_processing.data.load_data import load_all_data
import pandas as pd
from data_processing.features.indicators import calculate_all_indicators
from data_processing.time.time_align import align_timeframe_data, combine_timeframe_data
from data_processing.data.save_data import save_data_to_parquet
from data_processing.time.lagging import create_lag_features as create_lag_features_func

# Import validation modules
from data_processing.validation.feature_validation import (
    validate_raw_features, validate_filtered_features,
    validate_feature_consistency)
from data_processing.validation.target_validation import (
    validate_raw_target, validate_filtered_target,
    validate_target_feature_alignment)

# Import logging utilities
from data_processing.utils.logging_config import (setup_logging, get_logger,
                                                  log_stage_start,
                                                  log_stage_end, log_data_info,
                                                  log_validation_result)

# Set up logging
logger = setup_logging(log_level="INFO", console_output=True)


def calculate_indicators_on_all_data(df_h4, df_d1, df_w1):
    """Calculate technical indicators on all timeframes."""
    log_stage_start(logger, "Technical Indicators Calculation")

    start_time = time.time()

    # Calculate indicators
    df_h4 = calculate_all_indicators(df_h4)
    df_d1 = calculate_all_indicators(df_d1)
    df_w1 = calculate_all_indicators(df_w1)

    # Validate raw features
    validate_raw_features(df_h4,
                          "H4_indicators",
                          start=pd.to_datetime(TRAIN_START) -
                          pd.Timedelta(hours=4 * 12),
                          end=pd.to_datetime(TEST_END))
    validate_raw_features(df_d1,
                          "D1_indicators",
                          start=pd.to_datetime(TRAIN_START) -
                          pd.Timedelta(days=10),
                          end=pd.to_datetime(TEST_END))
    validate_raw_features(df_w1,
                          "W1_indicators",
                          start=pd.to_datetime(TRAIN_START) -
                          pd.Timedelta(weeks=4),
                          end=pd.to_datetime(TEST_END))

    duration = time.time() - start_time
    log_stage_end(logger, "Technical Indicators Calculation", duration)

    return df_h4, df_d1, df_w1


def align_timeframe(df_h4, df_d1, df_w1):
    """Align different timeframes with H4 base."""
    log_stage_start(logger, "Timeframe Alignment")

    start_time = time.time()

    # Create feature sets
    A0 = df_h4.copy()
    A1 = combine_timeframe_data(A0, df_d1, 'H4', 'D1')
    A2 = combine_timeframe_data(A0, df_w1, 'H4', 'W1')

    # Validate raw features
    validate_raw_features(A0,
                          "A0",
                          start=pd.to_datetime(TRAIN_START) -
                          pd.Timedelta(hours=4 * 12),
                          end=pd.to_datetime(TEST_END))
    validate_raw_features(A1,
                          "A1",
                          start=pd.to_datetime(TRAIN_START) -
                          pd.Timedelta(days=12),
                          end=pd.to_datetime(TEST_END))
    validate_raw_features(A2,
                          "A2",
                          start=pd.to_datetime(TRAIN_START) -
                          pd.Timedelta(weeks=4),
                          end=pd.to_datetime(TEST_END))

    duration = time.time() - start_time
    log_stage_end(logger, "Timeframe Alignment", duration)

    return A0, A1, A2


def create_lag_features(df_h4, df_d1, df_w1):
    """Create historical lag features for all timeframes."""
    log_stage_start(logger, "Lag Features Creation")

    start_time = time.time()

    # Create lag features
    h4_lags = create_lag_features_func(df_h4, 'H4', range(1, 11))
    d1_lags = create_lag_features_func(df_d1, 'D1', range(1, 11))
    w1_lags = create_lag_features_func(df_w1, 'W1', range(1, 5))

    # Validate raw features
    validate_raw_features(h4_lags,
                          "H4_lags",
                          start=pd.to_datetime(TRAIN_START),
                          end=pd.to_datetime(TEST_END),
                          drop_head_rows=10)
    validate_raw_features(d1_lags,
                          "D1_lags",
                          start=pd.to_datetime(TRAIN_START),
                          end=pd.to_datetime(TEST_END),
                          drop_head_rows=10)
    validate_raw_features(w1_lags,
                          "W1_lags",
                          start=pd.to_datetime(TRAIN_START),
                          end=pd.to_datetime(TEST_END),
                          drop_head_rows=3)  # W1 has 3-week lag

    duration = time.time() - start_time
    log_stage_end(logger, "Lag Features Creation", duration)

    return h4_lags, d1_lags, w1_lags


def combine_timeframe_and_lag_features(base_data, lag_data, base_timeframe,
                                       lag_timeframe):
    """Combine base data with lag features."""
    log_stage_start(logger, f"Combining {base_timeframe} + {lag_timeframe}")

    start_time = time.time()

    if lag_timeframe == 'H4_lags':
        lagged_data = pd.concat([base_data, lag_data], axis=1)
    elif lag_timeframe == 'D1_lags':
        lagged_data = align_timeframe_data(base_data, lag_data, 'H4',
                                           'D1_lags')
    elif lag_timeframe == 'W1_lags':
        lagged_data = align_timeframe_data(base_data, lag_data, 'H4',
                                           'W1_lags')
    else:
        raise ValueError(f"Invalid lag timeframe name: {lag_timeframe}")

    # Validate raw features
    # Adjust drop_head_rows based on lag timeframe
    drop_rows = 3 if 'W1' in lag_timeframe else 10
    validate_raw_features(lagged_data,
                          f"{base_timeframe}_{lag_timeframe}",
                          start=pd.to_datetime(TRAIN_START),
                          end=pd.to_datetime(TEST_END),
                          drop_head_rows=drop_rows)

    duration = time.time() - start_time
    log_stage_end(logger, f"Combining {base_timeframe} + {lag_timeframe}",
                  duration)

    return lagged_data


def filter_data(A0, A1, A2, A3, A4, A5, A6, A7, train_start, test_end):
    """Filter data to specified time range."""
    log_stage_start(logger,
                    "Data Filtering",
                    train_start=train_start,
                    test_end=test_end)

    start_time = time.time()

    # Filter data
    A0_filtered = A0[(A0.index >= train_start) & (A0.index <= test_end)]
    A1_filtered = A1[(A1.index >= train_start) & (A1.index <= test_end)]
    A2_filtered = A2[(A2.index >= train_start) & (A2.index <= test_end)]
    A3_filtered = A3[(A3.index >= train_start) & (A3.index <= test_end)]
    A4_filtered = A4[(A4.index >= train_start) & (A4.index <= test_end)]
    A5_filtered = A5[(A5.index >= train_start) & (A5.index <= test_end)]
    A6_filtered = A6[(A6.index >= train_start) & (A6.index <= test_end)]
    A7_filtered = A7[(A7.index >= train_start) & (A7.index <= test_end)]
    # Log data info
    log_data_info(logger, "A0_filtered", A0_filtered.shape,
                  A0_filtered.memory_usage(deep=True).sum() / 1024 / 1024)
    log_data_info(logger, "A1_filtered", A1_filtered.shape,
                  A1_filtered.memory_usage(deep=True).sum() / 1024 / 1024)
    log_data_info(logger, "A2_filtered", A2_filtered.shape,
                  A2_filtered.memory_usage(deep=True).sum() / 1024 / 1024)
    log_data_info(logger, "A3_filtered", A3_filtered.shape,
                  A3_filtered.memory_usage(deep=True).sum() / 1024 / 1024)
    log_data_info(logger, "A4_filtered", A4_filtered.shape,
                  A4_filtered.memory_usage(deep=True).sum() / 1024 / 1024)
    log_data_info(logger, "A5_filtered", A5_filtered.shape,
                  A5_filtered.memory_usage(deep=True).sum() / 1024 / 1024)
    log_data_info(logger, "A6_filtered", A6_filtered.shape,
                  A6_filtered.memory_usage(deep=True).sum() / 1024 / 1024)
    log_data_info(logger, "A7_filtered", A7_filtered.shape,
                  A7_filtered.memory_usage(deep=True).sum() / 1024 / 1024)

    duration = time.time() - start_time
    log_stage_end(logger, "Data Filtering", duration)

    return A0_filtered, A1_filtered, A2_filtered, A3_filtered, A4_filtered, A5_filtered, A6_filtered, A7_filtered


def validate_filtered_data(A0_filtered, A1_filtered, A2_filtered, A3_filtered,
                           A4_filtered, A5_filtered, A6_filtered, A7_filtered):
    """Validate all filtered feature sets."""
    log_stage_start(logger, "Filtered Data Validation")

    start_time = time.time()

    try:
        # Validate each feature set
        validate_filtered_features(A0_filtered,
                                   "A0_filtered",
                                   start=pd.to_datetime(TRAIN_START),
                                   end=pd.to_datetime(TEST_END),
                                   drop_head_rows=12)
        validate_filtered_features(A1_filtered,
                                   "A1_filtered",
                                   start=pd.to_datetime(TRAIN_START),
                                   end=pd.to_datetime(TEST_END),
                                   drop_head_rows=12)
        validate_filtered_features(A2_filtered,
                                   "A2_filtered",
                                   start=pd.to_datetime(TRAIN_START),
                                   end=pd.to_datetime(TEST_END),
                                   drop_head_rows=12)
        validate_filtered_features(A3_filtered,
                                   "A3_filtered",
                                   start=pd.to_datetime(TRAIN_START),
                                   end=pd.to_datetime(TEST_END),
                                   drop_head_rows=12)
        validate_filtered_features(A4_filtered,
                                   "A4_filtered",
                                   start=pd.to_datetime(TRAIN_START),
                                   end=pd.to_datetime(TEST_END),
                                   drop_head_rows=12)
        validate_filtered_features(A5_filtered,
                                   "A5_filtered",
                                   start=pd.to_datetime(TRAIN_START),
                                   end=pd.to_datetime(TEST_END),
                                   drop_head_rows=3)  # A5 has W1 lags
        validate_filtered_features(A6_filtered,
                                   "A6_filtered",
                                   start=pd.to_datetime(TRAIN_START),
                                   end=pd.to_datetime(TEST_END),
                                   drop_head_rows=3)  # A6 has W1 lags
        validate_filtered_features(A7_filtered,
                                   "A7_filtered",
                                   start=pd.to_datetime(TRAIN_START),
                                   end=pd.to_datetime(TEST_END),
                                   drop_head_rows=3)  # A7 has W1 lags
        # Validate consistency across sets
        feature_sets = {
            "A0": A0_filtered,
            "A1": A1_filtered,
            "A2": A2_filtered,
            "A3": A3_filtered,
            "A4": A4_filtered,
            "A5": A5_filtered,
            "A6": A6_filtered,
            "A7": A7_filtered
        }
        validate_feature_consistency(feature_sets)

        log_validation_result(logger, "Filtered Data", True)

    except Exception as e:
        log_validation_result(logger, "Filtered Data", False, str(e))
        raise

    duration = time.time() - start_time
    log_stage_end(logger, "Filtered Data Validation", duration)


def save_data(A0_filtered, A1_filtered, A2_filtered, A3_filtered, A4_filtered,
              A5_filtered, A6_filtered, A7_filtered):
    """Save all filtered feature sets."""
    log_stage_start(logger, "Data Saving")

    start_time = time.time()

    # Save feature sets
    save_data_to_parquet(A0_filtered, 'A0')
    save_data_to_parquet(A1_filtered, 'A1')
    save_data_to_parquet(A2_filtered, 'A2')
    save_data_to_parquet(A3_filtered, 'A3')
    save_data_to_parquet(A4_filtered, 'A4')
    save_data_to_parquet(A5_filtered, 'A5')
    save_data_to_parquet(A6_filtered, 'A6')
    save_data_to_parquet(A7_filtered, 'A7')

    duration = time.time() - start_time
    log_stage_end(logger, "Data Saving", duration)

    return A0_filtered, A1_filtered, A2_filtered, A3_filtered, A4_filtered, A5_filtered, A6_filtered, A7_filtered


def __main__():
    """Main pipeline execution function.
    python -m data_processing.main_pipeline
    """
    logger.info("🚀 Starting BTC Data Processing Pipeline")
    logger.info("=" * 60)

    pipeline_start_time = time.time()

    try:
        # Stage 1: Load data
        log_stage_start(logger, "Data Loading")
        load_start = time.time()
        df_h4, df_d1, df_w1 = load_all_data()
        log_stage_end(logger, "Data Loading", time.time() - load_start)

        # Stage 2: Calculate indicators
        df_h4, df_d1, df_w1 = calculate_indicators_on_all_data(
            df_h4, df_d1, df_w1)

        # Stage 3: Align timeframes
        A0, A1, A2 = align_timeframe(df_h4, df_d1, df_w1)

        # Stage 4: Create lag features
        h4_lags, d1_lags, w1_lags = create_lag_features(df_h4, df_d1, df_w1)

        # Stage 5: Combine features
        A3 = combine_timeframe_and_lag_features(A1, h4_lags, 'H4', 'H4_lags')
        A4 = combine_timeframe_and_lag_features(A3, d1_lags, 'H4', 'D1_lags')

        A5 = combine_timeframe_and_lag_features(A2, h4_lags, 'H4', 'H4_lags')
        A6 = combine_timeframe_and_lag_features(A5, d1_lags, 'H4', 'D1_lags')
        A7 = combine_timeframe_and_lag_features(A6, w1_lags, 'H4', 'W1_lags')

        # Stage 6: Filter data
        A0_filtered, A1_filtered, A2_filtered, A3_filtered, A4_filtered, A5_filtered, A6_filtered, A7_filtered = filter_data(
            A0, A1, A2, A3, A4, A5, A6, A7, TRAIN_START, TEST_END)

        # Stage 7: Validate filtered data
        validate_filtered_data(A0_filtered, A1_filtered, A2_filtered,
                               A3_filtered, A4_filtered, A5_filtered,
                               A6_filtered, A7_filtered)

        # Stage 8: Save data
        save_data(A0_filtered, A1_filtered, A2_filtered, A3_filtered,
                  A4_filtered, A5_filtered, A6_filtered, A7_filtered)

        # Pipeline completion
        total_duration = time.time() - pipeline_start_time
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"⏱️  Total execution time: {total_duration:.2f} seconds")
        logger.info("=" * 60)

        return "success"

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    __main__()
