import pandas as pd

from data_processing.time.time_align import align_timeframe_data


def create_lag_features(indicators_full, timeframe_name, lag_periods):
    """Create historical lag features for a timeframe using full data (including buffer)"""

    # Create all lag features at once to avoid fragmentation warning
    lag_dataframes = []

    for lag in lag_periods:
        lag_df = indicators_full.shift(lag)
        lag_df.columns = [
            f"{col}_lag_{timeframe_name}_{lag}" for col in lag_df.columns
        ]
        lag_dataframes.append(lag_df)

    lag_features = pd.concat(lag_dataframes, axis=1)

    print(
        f"✅ {timeframe_name} lags: {len(lag_features.columns)} features created"
    )
    return lag_features
