import pandas as pd
import talib
import numpy as np


def calculate_bollinger_bands(data: pd.DataFrame,
                              period: int = 20,
                              std: int = 2) -> pd.DataFrame:
    new_features = {}
    upper_band, middle_band, lower_band = talib.BBANDS(data['close'],
                                                       timeperiod=period,
                                                       nbdevup=std,
                                                       nbdevdn=std,
                                                       matype=0)
    new_features['upper_band'] = upper_band
    new_features['middle_band'] = middle_band
    new_features['lower_band'] = lower_band
    for col_name, values in new_features.items():
        data[col_name] = values
    return data


def normalize_bollinger_bands(data: pd.DataFrame) -> pd.DataFrame:
    """

    """
    upper_band = data['upper_band']
    middle_band = data['middle_band']
    lower_band = data['lower_band']
    close = data['close']
    new_features = {}

    new_features['band_width'] = np.where(
        middle_band != 0, (upper_band - lower_band) / middle_band, 0)
    band_range = upper_band - lower_band
    new_features['band_width_pct'] = np.where(
        band_range != 0, (close - lower_band) / band_range, 0)

    for col_name, values in new_features.items():
        data[col_name] = values
    return data


def remove_absolute_value_features(data: pd.DataFrame) -> pd.DataFrame:
    cols_to_remove = []
    cols_to_remove.extend(['upper_band', 'middle_band', 'lower_band'])
    data = data.drop(columns=cols_to_remove)
    return data


def transform_bollinger_bands_features(data: pd.DataFrame) -> pd.DataFrame:
    data = calculate_bollinger_bands(data)
    data = normalize_bollinger_bands(data)
    data = remove_absolute_value_features(data)
    return data
