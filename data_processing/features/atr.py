import pandas as pd
import talib
import numpy as np


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high = data['high']
    low = data['low']
    close = data['close']

    data['ATR'] = talib.ATR(high, low, close, timeperiod=period)

    return data


def normalize_atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    new_features = {}
    atr = data['ATR']
    close = data['close']

    new_features['ATR_norm'] = np.where(close != 0, atr / close, 0)

    atr_ma = talib.SMA(atr, timeperiod=period)
    atr_norm_ma = np.where(atr_ma != 0, (atr / atr_ma) - 1, 0)
    atr_norm_ma = np.where(np.isinf(atr_norm_ma), 0, atr_norm_ma)
    new_features['ATR_norm_ma'] = atr_norm_ma

    for col_name, values in new_features.items():
        data[col_name] = values
    return data


def remove_absolute_value_features(data: pd.DataFrame) -> pd.DataFrame:
    cols_to_remove = []
    cols_to_remove.extend(['ATR'])
    data = data.drop(columns=cols_to_remove)
    return data


def transform_atr_features(data: pd.DataFrame) -> pd.DataFrame:
    data = calculate_atr(data)
    data = normalize_atr(data)
    data = remove_absolute_value_features(data)
    return data
