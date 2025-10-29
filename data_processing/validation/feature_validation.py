"""
Feature validation utilities for data processing pipeline.

This module provides validation functions for feature data at different stages:
- Raw features (before filtering): Logs shape, NaN counts, inf counts
- Filtered features (after filtering): Strict validation, no NaN/inf allowed
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def validate_raw_features(df: pd.DataFrame, name: str, *, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None, drop_head_rows: int = 0) -> Dict[str, any]:
    """
    Validate raw features before filtering. Logs statistics but never raises.
    
    Args:
        df: DataFrame to validate
        name: Name of the feature set (e.g., 'A0', 'A1', etc.)
    
    Returns:
        Dict with validation statistics
    """
    # Restrict to date window if provided
    df_view = df
    if start is not None or end is not None:
        start_dt = start if start is not None else df.index.min()
        end_dt = end if end is not None else df.index.max()
        df_view = df_view[(df_view.index >= start_dt) & (df_view.index <= end_dt)]
    if drop_head_rows > 0 and len(df_view) > drop_head_rows:
        df_view = df_view.iloc[drop_head_rows:]

    stats = {
        'name': name,
        'shape': df_view.shape,
        'total_cells': df_view.shape[0] * df_view.shape[1],
        'nan_count': df_view.isnull().sum().sum(),
        'inf_count': np.isinf(df_view.select_dtypes(include=[np.number])).sum().sum(),
        'duplicate_indices': df_view.index.duplicated().sum(),
        'memory_usage_mb': df_view.memory_usage(deep=True).sum() / 1024 / 1024
    }

    # Log basic statistics
    logger.info(
        f"📊 {name} raw features: {stats['shape'][0]} rows × {stats['shape'][1]} cols"
    )
    logger.info(f"   Memory: {stats['memory_usage_mb']:.2f} MB")

    # Log NaN statistics (top 20 columns with most NaNs)
    nan_counts = df_view.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

    if len(nan_counts) > 0:
        logger.warning(
            f"⚠️  {name} has {stats['nan_count']} NaNs ({stats['nan_count']/stats['total_cells']*100:.1f}%)"
        )
        logger.warning(f"   Top NaN columns: {dict(nan_counts.head(10))}")
    else:
        logger.info(f"✅ {name} has no NaN values")

    # Log inf statistics
    if stats['inf_count'] > 0:
        logger.warning(f"⚠️  {name} has {stats['inf_count']} infinite values")
    else:
        logger.info(f"✅ {name} has no infinite values")

    # Log duplicate indices
    if stats['duplicate_indices'] > 0:
        logger.warning(
            f"⚠️  {name} has {stats['duplicate_indices']} duplicate indices")
    else:
        logger.info(f"✅ {name} has unique indices")

    return stats


def validate_filtered_features(df: pd.DataFrame, name: str, *, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None, drop_head_rows: int = 0) -> None:
    """
    Strict validation for filtered features. Raises ValueError on any issues.
    
    Args:
        df: DataFrame to validate
        name: Name of the feature set (e.g., 'A0_filtered', 'A1_filtered', etc.)
    
    Raises:
        ValueError: If validation fails
    """
    logger.info(f"🔍 Validating {name} filtered features...")

    # Restrict to date window if provided
    df_view = df
    if start is not None or end is not None:
        start_dt = start if start is not None else df.index.min()
        end_dt = end if end is not None else df.index.max()
        df_view = df_view[(df_view.index >= start_dt) & (df_view.index <= end_dt)]
    if drop_head_rows > 0 and len(df_view) > drop_head_rows:
        df_view = df_view.iloc[drop_head_rows:]

    # Check for NaN values
    nan_count = df_view.isnull().sum().sum()
    if nan_count > 0:
        nan_cols = df.isnull().sum()
        nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
        error_msg = f"{name} contains {nan_count} NaN values in columns: {dict(nan_cols.head(10))}"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

    # Check for infinite values
    inf_count = np.isinf(df_view.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        inf_cols = np.isinf(df.select_dtypes(include=[np.number])).sum()
        inf_cols = inf_cols[inf_cols > 0].sort_values(ascending=False)
        error_msg = f"{name} contains {inf_count} infinite values in columns: {dict(inf_cols.head(10))}"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

    # Check for duplicate indices
    duplicate_count = df_view.index.duplicated().sum()
    if duplicate_count > 0:
        error_msg = f"{name} has {duplicate_count} duplicate indices"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

    # Check for empty DataFrame
    if df_view.empty:
        error_msg = f"{name} is empty"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

    logger.info(
        f"✅ {name} validation passed: {df_view.shape[0]} rows × {df_view.shape[1]} cols")


def validate_feature_consistency(dfs: Dict[str, pd.DataFrame]) -> None:
    """
    Validate consistency across multiple feature sets.
    
    Args:
        dfs: Dictionary of feature sets {name: DataFrame}
    """
    logger.info("🔍 Validating feature consistency across sets...")

    if not dfs:
        logger.warning("⚠️  No feature sets provided for consistency check")
        return

    # Check index alignment
    base_index = None
    for name, df in dfs.items():
        if base_index is None:
            base_index = df.index
            logger.info(
                f"   Using {name} as base index: {len(base_index)} records")
        else:
            if not df.index.equals(base_index):
                logger.warning(f"⚠️  {name} index doesn't match base index")
            else:
                logger.info(f"✅ {name} index matches base")

    # Check for overlapping columns
    all_columns = {}
    for name, df in dfs.items():
        for col in df.columns:
            if col in all_columns:
                logger.warning(
                    f"⚠️  Column '{col}' appears in both {all_columns[col]} and {name}"
                )
            else:
                all_columns[col] = name

    logger.info("✅ Feature consistency validation completed")
