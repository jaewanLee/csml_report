import pandas as pd
import numpy as np


def align_timeframe_data(base_data, target_data, base_timeframe,
                         target_timeframe):
    """
    Align target timeframe data with base timeframe data using proper temporal alignment
    """
    timeframe_offsets = {
        'D1': pd.Timedelta(days=1),
        'W1': pd.Timedelta(weeks=1),
        # Add lag support
        'D1_lags': pd.Timedelta(days=1),
        'W1_lags': pd.Timedelta(weeks=1)
    }

    aligned_data = pd.DataFrame(index=base_data.index,
                                columns=target_data.columns)

    for base_timestamp in base_data.index:
        # Use regular timedelta for other timeframes
        offset = timeframe_offsets[target_timeframe]
        cutoff_time = base_timestamp - offset

        # Find target data that is <= cutoff_time (previous completed data)
        available_target_data = target_data[target_data.index <= cutoff_time]

        if len(available_target_data) > 0:
            # Use the most recent available data (previous completed)
            latest_target_data = available_target_data.iloc[-1]
            aligned_data.loc[base_timestamp] = latest_target_data
        else:
            # If no data available, fill with NaN
            aligned_data.loc[base_timestamp] = np.nan

    print(
        f"✅ {target_timeframe} data aligned: {len(aligned_data.columns)} features, {len(aligned_data)} records"
    )
    return aligned_data


def combine_timeframe_data(base_data, target_data, base_timeframe,
                           target_timeframe):
    """
    Combine base timeframe data with target timeframe data
    """
    aligned_data = align_timeframe_data(base_data, target_data, base_timeframe,
                                        target_timeframe)
    aligned_data.columns = [
        f"{target_timeframe}_{col}" for col in aligned_data.columns
    ]
    combined_data = pd.concat([base_data, aligned_data], axis=1)
    print(
        f"✅ {base_timeframe} + {target_timeframe} data combined: {len(combined_data.columns)} features, {len(combined_data)} records"
    )
    return combined_data
