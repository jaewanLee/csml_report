from typing import Tuple
import pandas as pd
from pathlib import Path


def load_data(file_name: str):
    # Use absolute path from project root
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / "data_collection" / "data" / f"btc_{file_name}_20251029.parquet"
    df = pd.read_parquet(file_path)
    return df


def load_4h_data():
    df = load_data('4h')
    return df


def load_1d_data():
    df = load_data('1d')
    return df


def load_1w_data():
    df = load_data('1w')
    return df


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_h4 = load_4h_data()
    df_d1 = load_1d_data()
    df_w1 = load_1w_data()
    return df_h4, df_d1, df_w1
