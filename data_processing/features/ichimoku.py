import pandas as pd
import talib
import numpy as np


def calculate_ichimoku(data: pd.DataFrame) -> pd.DataFrame:

    new_features = {}

    # Tenkan-sen (Conversion Line)
    high_9 = data['high'].rolling(window=9).max()
    low_9 = data['low'].rolling(window=9).min()
    new_features['conversion_line'] = (high_9 + low_9) / 2

    # Kijun-sen (Baseline)
    high_26 = data['high'].rolling(window=26).max()
    low_26 = data['low'].rolling(window=26).min()
    new_features['baseline'] = (high_26 + low_26) / 2

    # Senkou Span A (Leading Span A)
    span_a_value = (new_features['conversion_line'] +
                    new_features['baseline']) / 2
    new_features['leading_span_A'] = span_a_value.shift(26)

    # Senkou Span B (Leading Span B)
    high_52 = talib.MAX(data['high'], timeperiod=52)
    low_52 = talib.MIN(data['low'], timeperiod=52)
    span_b_value = (high_52 + low_52) / 2
    new_features['leading_span_B'] = span_b_value.shift(26)

    new_features['lagging_span'] = data['close'].shift(
        26
    )  # 정확히 lagging span은 아니지만, 실제 lagging span 활용은 26일전 캔들과 현재 캔들을 비교하기때문에 과거값을 활용

    for col_name, values in new_features.items():
        data[col_name] = values
    return data


def normalize_ichimoku(data: pd.DataFrame) -> pd.DataFrame:
    new_features = {}

    # 원본 Ichimoku 라인 이름 목록
    ichimoku_lines = [
        'conversion_line', 'baseline', 'leading_span_A', 'leading_span_B',
        'lagging_span'
    ]

    # --- 1. (Close vs. 원본 Ichimoku Line) 계산 ---
    if 'close' not in data.columns:
        print(
            "Warning: 'close' column not found. Skipping close-based normalizations."
        )
    else:
        for line_col in ichimoku_lines:
            if line_col not in data.columns:
                print(f"Warning: {line_col} not found. Skipping.")
                continue

            new_feature_name = f'close_vs_{line_col}_pct'
            # (close / line) - 1
            normalized_val = (data['close'] / data[line_col]) - 1
            new_features[new_feature_name] = normalized_val.replace(
                [np.inf, -np.inf, np.nan], 0)  # NaN도 0으로 처리

    # --- 2. (원본 Line vs. 원본 Line) 계산 (크로스) ---

    # 전환선 vs 기준선
    if 'conversion_line' in data.columns and 'baseline' in data.columns:
        new_feature_name = 'conversion_vs_baseline_pct'
        # (conversion / baseline) - 1
        normalized_val = (data['conversion_line'] / data['baseline']) - 1
        new_features[new_feature_name] = normalized_val.replace(
            [np.inf, -np.inf, np.nan], 0)

    # 선행스팬 A vs 선행스팬 B (구름 관계)
    if 'leading_span_A' in data.columns and 'leading_span_B' in data.columns:
        new_feature_name = 'span_A_vs_span_B_pct'
        # (Span A / Span B) - 1
        normalized_val = (data['leading_span_A'] / data['leading_span_B']) - 1
        new_features[new_feature_name] = normalized_val.replace(
            [np.inf, -np.inf, np.nan], 0)

    # --- 3. 계산된 모든 피처를 원본 DataFrame에 한 번에 추가 ---
    for col_name, values in new_features.items():
        data[col_name] = values

    return data


def remove_absolute_value_features(data: pd.DataFrame) -> pd.DataFrame:
    cols_to_remove = []
    cols_to_remove.extend([
        f'conversion_line', 'baseline', 'leading_span_A', 'leading_span_B',
        'lagging_span'
    ])
    data = data.drop(columns=cols_to_remove)
    return data


def transform_ichimoku_features(data: pd.DataFrame) -> pd.DataFrame:
    data = calculate_ichimoku(data)
    data = normalize_ichimoku(data)
    data = remove_absolute_value_features(data)
    return data
