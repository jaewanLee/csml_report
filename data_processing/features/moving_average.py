from itertools import combinations
import pandas as pd
import talib
from typing import List
import numpy as np


def calculate_moving_averages(
        data: pd.DataFrame,
        periods: List[int] = [7, 14, 20, 60, 120]) -> pd.DataFrame:
    new_features = {}
    for period in periods:
        new_features[f"MA_{period}"] = talib.SMA(data['close'],
                                                 timeperiod=period)
    for col_name, values in new_features.items():
        data[col_name] = values
    return data


def normalize_moving_averages(
        data: pd.DataFrame,
        periods: List[int] = [7, 14, 20, 60, 120]) -> pd.DataFrame:
    """Normalize moving averages by dividing by close price"""
    new_features = {}

    for period in periods:
        ma_col = f'MA_{period}'
        new_feature_name = f'{ma_col}_norm'
        if ma_col in data.columns:
            normalized_val = (data['close'] / data[ma_col]) - 1
            new_features[new_feature_name] = normalized_val.replace(
                [np.inf, -np.inf], 0)
        else:
            print(f"Warning: {ma_col} not found. Skipping {new_feature_name}.")

    for (p_short, p_long) in combinations(periods, 2):
        ma_col_short = f'MA_{p_short}'
        ma_col_long = f'MA_{p_long}'
        if ma_col_short not in data.columns or ma_col_long not in data.columns:
            print(
                f"Warning: {ma_col_short} or {ma_col_long} not found. Skipping {new_feature_name}."
            )
            continue
        new_feature_name = f'{ma_col_short}_{ma_col_long}_norm'
        if ma_col_short in data.columns and ma_col_long in data.columns:
            normalized_val = (data[ma_col_short] / data[ma_col_long]) - 1
            new_features[new_feature_name] = normalized_val.replace(
                [np.inf, -np.inf], 0)
        else:
            print(
                f"Warning: {ma_col_short} or {ma_col_long} not found. Skipping {new_feature_name}."
            )

    for col_name, values in new_features.items():
        data[col_name] = values

    return data


def remove_absolute_value_features(
        data: pd.DataFrame,
        periods: List[int] = [7, 14, 20, 60, 120]) -> pd.DataFrame:
    cols_to_remove = []
    cols_to_remove.extend([f'MA_{period}' for period in periods])
    data = data.drop(columns=cols_to_remove)
    return data


def transform_moving_average_features(
        data: pd.DataFrame,
        periods: List[int] = [7, 14, 20, 60, 120]) -> pd.DataFrame:
    data = calculate_moving_averages(data, periods)
    data = normalize_moving_averages(data, periods)
    data = remove_absolute_value_features(data, periods)
    return data
