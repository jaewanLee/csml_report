"""
Target variable validation utilities for data processing pipeline.

This module provides validation functions for target variables at different stages:
- Raw target (before filtering): Logs distribution statistics
- Filtered target (after filtering): Strict validation, binary values only
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def validate_raw_target(y: pd.DataFrame,
                        name: str = "target") -> Dict[str, any]:
    """
    Validate raw target variable before filtering. Logs distribution but never raises.
    
    Args:
        y: Target DataFrame to validate
        name: Name of the target variable
    
    Returns:
        Dict with validation statistics
    """
    if 'target' not in y.columns:
        logger.error(f"❌ {name} missing 'target' column")
        return {}

    target_series = y['target']

    stats = {
        'name': name,
        'total_samples': len(target_series),
        'unique_values': target_series.unique(),
        'value_counts': target_series.value_counts().to_dict(),
        'nan_count': target_series.isnull().sum(),
        'inf_count': np.isinf(target_series).sum(),
        'duplicate_indices': y.index.duplicated().sum(),
        'memory_usage_mb': y.memory_usage(deep=True).sum() / 1024 / 1024
    }

    # Log basic statistics
    logger.info(f"🎯 {name} raw target: {stats['total_samples']} samples")
    logger.info(f"   Memory: {stats['memory_usage_mb']:.2f} MB")

    # Log value distribution
    logger.info(f"   Value distribution: {stats['value_counts']}")

    # Log unique values
    logger.info(f"   Unique values: {stats['unique_values']}")

    # Log NaN statistics
    if stats['nan_count'] > 0:
        logger.warning(f"⚠️  {name} has {stats['nan_count']} NaN values")
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


def validate_filtered_target(y: pd.DataFrame,
                             name: str = "target_filtered") -> None:
    """
    Strict validation for filtered target variable. Raises ValueError on any issues.
    
    Args:
        y: Target DataFrame to validate
        name: Name of the target variable
    
    Raises:
        ValueError: If validation fails
    """
    logger.info(f"🔍 Validating {name} filtered target...")

    # Check for 'target' column
    if 'target' not in y.columns:
        error_msg = f"{name} missing 'target' column"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

    target_series = y['target']

    # Check for NaN values
    nan_count = target_series.isnull().sum()
    if nan_count > 0:
        error_msg = f"{name} contains {nan_count} NaN values"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

    # Check for infinite values
    inf_count = np.isinf(target_series).sum()
    if inf_count > 0:
        error_msg = f"{name} contains {inf_count} infinite values"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

    # Check for binary values (0 and 1 only)
    unique_values = target_series.unique()
    if not set(unique_values).issubset({0, 1}):
        error_msg = f"{name} contains non-binary values: {unique_values}"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

    # Check for duplicate indices
    duplicate_count = y.index.duplicated().sum()
    if duplicate_count > 0:
        error_msg = f"{name} has {duplicate_count} duplicate indices"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

    # Check for empty DataFrame
    if y.empty:
        error_msg = f"{name} is empty"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

    # Log target distribution
    value_counts = target_series.value_counts().sort_index()
    logger.info(f"✅ {name} validation passed: {len(y)} samples")
    logger.info(f"   Target distribution: {value_counts.to_dict()}")
    logger.info(f"   Sell percentage: {target_series.mean()*100:.2f}%")


def validate_target_distribution_with_folds(y: pd.DataFrame,
                                            n_folds: int = 5,
                                            name: str = "target") -> Dict:
    """
    Validate target distribution with fold analysis.
    
    Args:
        y: Target DataFrame with 'target' column
        n_folds: Number of folds for analysis
        name: Name for logging
    
    Returns:
        Dict with distribution statistics
    """
    logger.info(
        f"🔍 Validating {name} distribution with {n_folds}-fold analysis...")

    if 'target' not in y.columns:
        error_msg = f"{name} missing 'target' column"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

    target_series = y['target']
    total_samples = len(target_series)

    # 전체 분포 계산
    value_counts = target_series.value_counts().sort_index()
    positive_count = value_counts.get(1, 0)
    negative_count = value_counts.get(0, 0)
    positive_ratio = positive_count / total_samples if total_samples > 0 else 0

    logger.info(f"📊 Overall distribution:")
    logger.info(f"   Total samples: {total_samples}")
    logger.info(f"   Positive (1): {positive_count} ({positive_ratio:.3f})")
    logger.info(f"   Negative (0): {negative_count} ({1-positive_ratio:.3f})")

    # 5-fold 분석
    fold_size = total_samples // n_folds
    fold_stats = []

    for fold in range(n_folds):
        start_idx = fold * fold_size
        if fold == n_folds - 1:  # 마지막 fold는 나머지 모든 데이터 포함
            end_idx = total_samples
        else:
            end_idx = (fold + 1) * fold_size

        fold_data = target_series.iloc[start_idx:end_idx]
        fold_positive = fold_data.sum()
        fold_total = len(fold_data)
        fold_positive_ratio = fold_positive / fold_total if fold_total > 0 else 0

        fold_stats.append({
            'fold': fold + 1,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'total_samples': fold_total,
            'positive_samples': fold_positive,
            'positive_ratio': fold_positive_ratio,
            'negative_samples': fold_total - fold_positive,
            'negative_ratio': 1 - fold_positive_ratio
        })

        logger.info(
            f"   Fold {fold+1}: {fold_positive}/{fold_total} ({fold_positive_ratio:.3f})"
        )

    # Fold별 비율 균형 검사
    fold_ratios = [stat['positive_ratio'] for stat in fold_stats]
    ratio_std = np.std(fold_ratios)
    ratio_range = max(fold_ratios) - min(fold_ratios)

    logger.info(f"📈 Fold balance analysis:")
    logger.info(f"   Ratio std: {ratio_std:.3f}")
    logger.info(f"   Ratio range: {ratio_range:.3f}")

    # 균형 검사 (std < 0.05, range < 0.1)
    if ratio_std > 0.05:
        logger.warning(
            f"⚠️  High fold ratio variance (std={ratio_std:.3f} > 0.05)")
    if ratio_range > 0.1:
        logger.warning(f"⚠️  High fold ratio range ({ratio_range:.3f} > 0.1)")

    if ratio_std <= 0.05 and ratio_range <= 0.1:
        logger.info("✅ Fold distribution is well balanced")
    else:
        logger.warning("⚠️  Fold distribution shows imbalance")

    return {
        'overall': {
            'total_samples': total_samples,
            'positive_samples': positive_count,
            'positive_ratio': positive_ratio,
            'negative_samples': negative_count,
            'negative_ratio': 1 - positive_ratio
        },
        'folds': fold_stats,
        'balance_metrics': {
            'ratio_std': ratio_std,
            'ratio_range': ratio_range,
            'is_balanced': ratio_std <= 0.05 and ratio_range <= 0.1
        }
    }


def validate_target_feature_alignment(y: pd.DataFrame,
                                      X: pd.DataFrame,
                                      target_name: str = "target",
                                      feature_name: str = "features") -> None:
    """
    Validate alignment between target and feature data.
    
    Args:
        y: Target DataFrame
        X: Feature DataFrame
        target_name: Name of target for logging
        feature_name: Name of features for logging
    
    Raises:
        ValueError: If alignment fails
    """
    logger.info(f"🔍 Validating {target_name} and {feature_name} alignment...")

    # Check index alignment
    if not y.index.equals(X.index):
        error_msg = f"{target_name} and {feature_name} indices don't match"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

    # Check length alignment
    if len(y) != len(X):
        error_msg = f"{target_name} length ({len(y)}) doesn't match {feature_name} length ({len(X)})"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

    logger.info(
        f"✅ {target_name} and {feature_name} are properly aligned: {len(y)} samples"
    )
