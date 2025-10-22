"""
BTC Prediction Project - Cross-Validation Utilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Generator, Tuple
from config.settings import N_SPLITS


def create_time_series_cv(features: pd.DataFrame,
                          target: pd.DataFrame,
                          n_splits: int = N_SPLITS) -> Generator:
    """
    Create TimeSeriesSplit for temporal cross-validation
    
    Args:
        features: Feature DataFrame
        target: Target DataFrame
        n_splits: Number of splits
    
    Yields:
        Tuple of (train_idx, val_idx) for each fold
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Convert to numpy arrays for sklearn
    X = features.values
    y = target['target'].values

    for train_idx, val_idx in tscv.split(X):
        yield train_idx, val_idx


def get_cv_splits_info(features: pd.DataFrame,
                       target: pd.DataFrame,
                       n_splits: int = N_SPLITS) -> pd.DataFrame:
    """
    Get information about cross-validation splits
    
    Args:
        features: Feature DataFrame
        target: Target DataFrame
        n_splits: Number of splits
    
    Returns:
        DataFrame with split information
    """
    splits_info = []

    for fold, (train_idx, val_idx) in enumerate(
            create_time_series_cv(features, target, n_splits)):
        train_start = features.index[train_idx[0]]
        train_end = features.index[train_idx[-1]]
        val_start = features.index[val_idx[0]]
        val_end = features.index[val_idx[-1]]

        train_target_dist = target.iloc[train_idx]['target'].value_counts(
        ).to_dict()
        val_target_dist = target.iloc[val_idx]['target'].value_counts(
        ).to_dict()

        splits_info.append({
            'fold': fold + 1,
            'train_start': train_start,
            'train_end': train_end,
            'train_samples': len(train_idx),
            'train_sell': train_target_dist.get(1, 0),
            'train_rest': train_target_dist.get(0, 0),
            'val_start': val_start,
            'val_end': val_end,
            'val_samples': len(val_idx),
            'val_sell': val_target_dist.get(1, 0),
            'val_rest': val_target_dist.get(0, 0)
        })

    return pd.DataFrame(splits_info)


def print_cv_splits_info(features: pd.DataFrame,
                         target: pd.DataFrame,
                         n_splits: int = N_SPLITS):
    """
    Print cross-validation splits information
    
    Args:
        features: Feature DataFrame
        target: Target DataFrame
        n_splits: Number of splits
    """
    splits_df = get_cv_splits_info(features, target, n_splits)

    print(f"🔄 TimeSeriesSplit ({n_splits} folds):")
    print("=" * 80)

    for _, row in splits_df.iterrows():
        print(f"Fold {row['fold']}:")
        print(
            f"  Training: {row['train_start']} to {row['train_end']} ({row['train_samples']} samples)"
        )
        print(f"    - SELL: {row['train_sell']}, REST: {row['train_rest']}")
        print(
            f"  Validation: {row['val_start']} to {row['val_end']} ({row['val_samples']} samples)"
        )
        print(f"    - SELL: {row['val_sell']}, REST: {row['val_rest']}")
        print()
