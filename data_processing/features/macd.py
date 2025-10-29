import pandas as pd
import talib


def calculate_macd(data: pd.DataFrame,
                   fast: int = 12,
                   slow: int = 26,
                   signal: int = 9) -> pd.DataFrame:
    macd_line, macd_signal, macd_histogram = talib.MACD(data['close'],
                                                        fastperiod=fast,
                                                        slowperiod=slow,
                                                        signalperiod=signal)
    data['MACD_line'] = macd_line
    data['MACD_signal'] = macd_signal
    data['MACD_histogram'] = macd_histogram
    return data
