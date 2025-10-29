from typing import List
import pandas as pd
import talib

from data_processing.features.ohlcv import transform_ohlcv_features
from data_processing.features.moving_average import transform_moving_average_features
from data_processing.features.rsi import calculate_rsi
from data_processing.features.macd import calculate_macd
from data_processing.features.ichimoku import transform_ichimoku_features
from data_processing.features.atr import transform_atr_features
from data_processing.features.bollinger_bands import transform_bollinger_bands_features


def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = transform_moving_average_features(data)
    data = calculate_rsi(data)
    data = calculate_macd(data)
    data = transform_ichimoku_features(data)
    data = transform_atr_features(data)
    data = transform_bollinger_bands_features(data)
    data = transform_ohlcv_features(data)
    return data
