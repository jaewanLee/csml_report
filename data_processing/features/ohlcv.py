from typing import List
import pandas as pd
import numpy as np
import talib


def extract_candle_features(data: pd.DataFrame) -> pd.DataFrame:
    new_features = {}
    open_price = data['open']
    high_price = data['high']
    low_price = data['low']
    close_price = data['close']

    new_features['pattern_hammer'] = talib.CDLHAMMER(open_price, high_price,
                                                     low_price, close_price)

    # 예시: 교수형 (Hanging Man) - 약세 반전 가능성
    new_features['pattern_hangingman'] = talib.CDLHANGINGMAN(
        open_price, high_price, low_price, close_price)

    # 예시: 장악형 (Engulfing)
    new_features['pattern_engulfing'] = talib.CDLENGULFING(
        open_price, high_price, low_price, close_price)

    # 예시: 십자형 (Doji)
    new_features['pattern_doji'] = talib.CDLDOJI(open_price, high_price,
                                                 low_price, close_price)

    # 예시: 저녁별형 (Evening Star) - 약세 반전
    new_features['pattern_eveningstar'] = talib.CDLEVENINGSTAR(
        open_price, high_price, low_price, close_price)

    for col_name, values in new_features.items():
        data[col_name] = values

    return data


def extract_volume_features(data: pd.DataFrame) -> pd.DataFrame:
    data['volume_MA_20'] = talib.SMA(data['volume'], timeperiod=20)
    return data


def normalize_candle_features(data: pd.DataFrame) -> pd.DataFrame:
    new_features = {}

    new_features['is_hammer'] = (data['pattern_hammer'] != 0).astype(int)
    new_features['is_hangingman'] = (data['pattern_hangingman']
                                     != 0).astype(int)
    new_features['is_engulfing'] = (data['pattern_engulfing'] != 0).astype(int)
    new_features['is_doji'] = (data['pattern_doji'] != 0).astype(int)
    new_features['is_eveningstar'] = (data['pattern_eveningstar']
                                      != 0).astype(int)

    for col_name, values in new_features.items():
        data[col_name] = values

    return data


def normalize_ohlcv_features(data: pd.DataFrame) -> pd.DataFrame:
    new_features = {}

    # 1. 캔들 몸통 비율 (종가 변화율)
    open_price = data['open']
    close_price = data['close']
    new_features['candle_body_pct'] = np.where(
        open_price != 0, (close_price - open_price) / open_price, 0)

    # 2. 윗꼬리 비율
    # (high - max(open, close)) / close
    high_price = data['high']
    body_top = np.maximum(open_price,
                          close_price)  # 몸통 상단 (양봉이면 close, 음봉이면 open)
    new_features['high_wick_pct'] = np.where(
        close_price != 0, (high_price - body_top) / close_price, 0)

    # 3. 아랫꼬리 비율
    # (min(open, close) - low) / close
    low_price = data['low']
    body_bottom = np.minimum(open_price,
                             close_price)  # 몸통 하단 (양봉이면 open, 음봉이면 close)
    new_features['low_wick_pct'] = np.where(
        close_price != 0, (body_bottom - low_price) / close_price, 0)

    # 4. 캔들 전체 범위 비율
    # (high - low) / low
    new_features['range_pct'] = np.where(low_price != 0,
                                         (high_price - low_price) / low_price,
                                         0)

    # 계산된 모든 피처를 원본 DataFrame에 한 번에 추가
    for col_name, values in new_features.items():
        data[col_name] = values

    return data


def normalize_volume_features(data: pd.DataFrame,
                              ma_periods: List[int] = [20]) -> pd.DataFrame:
    new_features = {}

    # --- 1. (Volume vs. Volume MA) 이격도 계산 ---
    if 'volume' in data.columns:
        for period in ma_periods:
            ma_col = f'volume_MA_{period}'
            new_feature_name = f'volume_vs_{ma_col}_pct'

            if ma_col in data.columns:
                # (volume / volume_MA) - 1
                normalized_val = (data['volume'] / data[ma_col]) - 1
                new_features[new_feature_name] = normalized_val.replace(
                    [np.inf, -np.inf, np.nan], 0)
            else:
                print(
                    f"Warning: {ma_col} not found for normalization. Skipping {new_feature_name}."
                )

    # --- 2. (이전 캔들 대비 Volume 변화율) 계산 ---
    if 'volume' in data.columns:
        new_feature_name = 'volume_change_pct'
        # .pct_change()는 이전 값 대비 변화율을 계산합니다.
        # 첫 번째 값은 NaN이 되므로 fillna(0)으로 처리합니다.
        new_features[new_feature_name] = data['volume'].pct_change().fillna(
            0).replace([np.inf, -np.inf], 0)
    else:
        print(
            "Warning: 'volume' column not found. Skipping volume_change_pct.")

    # --- 3. 계산된 모든 피처를 원본 DataFrame에 한 번에 추가 ---
    for col_name, values in new_features.items():
        data[col_name] = values

    return data


def remove_absolute_value_features(data: pd.DataFrame) -> pd.DataFrame:
    cols_to_remove = []
    cols_to_remove.extend(['open', 'high', 'low', 'close'])
    cols_to_remove.extend(['volume', 'volume_MA_20'])
    cols_to_remove.extend([
        'pattern_hammer', 'pattern_hangingman', 'pattern_engulfing',
        'pattern_doji', 'pattern_eveningstar'
    ])
    data = data.drop(columns=cols_to_remove)
    return data


def transform_ohlcv_features(data: pd.DataFrame) -> pd.DataFrame:
    data = extract_candle_features(data)
    data = extract_volume_features(data)
    data = normalize_candle_features(data)
    data = normalize_volume_features(data)
    data = normalize_ohlcv_features(data)
    data = remove_absolute_value_features(data)
    return data
