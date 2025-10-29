import pandas as pd
import talib


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    close = data['close']
    data['RSI'] = talib.RSI(close, timeperiod=period)
    return data
