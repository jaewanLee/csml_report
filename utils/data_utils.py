"""
BTC Prediction Project - Data Utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from config.settings import FEATURE_SETS, TARGET_FILE, TRAIN_START, TRAIN_END, TEST_START, TEST_END


def load_experiment_data(exp_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load feature set and target variable for a specific experiment
    
    Args:
        exp_id: Experiment ID (A0, A1, A2, A3, A4, A4_Pruned)
    
    Returns:
        Tuple of (features, target) DataFrames
    """
    if exp_id not in FEATURE_SETS:
        raise ValueError(
            f"Invalid experiment ID: {exp_id}. Must be one of {list(FEATURE_SETS.keys())}"
        )

    # Load features
    features_path = FEATURE_SETS[exp_id]
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    features = pd.read_parquet(features_path)

    # Load target
    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Target file not found: {TARGET_FILE}")

    target = pd.read_parquet(TARGET_FILE)

    # Ensure same index
    common_index = features.index.intersection(target.index)
    features = features.loc[common_index]
    target = target.loc[common_index]

    return features, target


def split_data_temporal(
    features: pd.DataFrame, target: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets using temporal split
    
    Args:
        features: Feature DataFrame
        target: Target DataFrame
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Training data
    train_mask = (features.index >= TRAIN_START) & (features.index
                                                    <= TRAIN_END)
    X_train = features[train_mask]
    y_train = target[train_mask]

    # Test data
    test_mask = (features.index >= TEST_START) & (features.index <= TEST_END)
    X_test = features[test_mask]
    y_test = target[test_mask]

    # Safety: ensure no overlap between train and test indices
    if len(X_train) > 0 and len(X_test) > 0:
        assert X_train.index.max() < X_test.index.min(
        ), "Train/Test overlap detected: adjust TRAIN_END and TEST_START"

    print(f"Data split:")
    print(
        f"  Training: {len(X_train)} samples ({X_train.index[0]} to {X_train.index[-1]})"
    )
    print(
        f"  Test: {len(X_test)} samples ({X_test.index[0]} to {X_test.index[-1]})"
    )

    return X_train, X_test, y_train, y_test


def validate_data_quality(features: pd.DataFrame,
                          target: pd.DataFrame) -> bool:
    """
    Validate data quality for training
    
    Args:
        features: Feature DataFrame
        target: Target DataFrame
    
    Returns:
        True if data is valid, False otherwise
    """
    print("🔍 Data Quality Validation:")

    # Check for missing values
    missing_features = features.isnull().sum().sum()
    missing_target = target.isnull().sum().sum()

    if missing_features > 0:
        print(f"  ❌ Missing values in features: {missing_features}")
        return False
    else:
        print(f"  ✅ No missing values in features")

    if missing_target > 0:
        print(f"  ❌ Missing values in target: {missing_target}")
        return False
    else:
        print(f"  ✅ No missing values in target")

    # Check for infinite values
    inf_features = np.isinf(
        features.select_dtypes(include=[np.number])).sum().sum()
    inf_target = np.isinf(
        target.select_dtypes(include=[np.number])).sum().sum()

    if inf_features > 0:
        print(f"  ❌ Infinite values in features: {inf_features}")
        return False
    else:
        print(f"  ✅ No infinite values in features")

    if inf_target > 0:
        print(f"  ❌ Infinite values in target: {inf_target}")
        return False
    else:
        print(f"  ✅ No infinite values in target")

    # Check data types
    print(f"  📊 Features shape: {features.shape}")
    print(f"  📊 Target shape: {target.shape}")
    print(
        f"  📊 Target distribution: {target['target'].value_counts().to_dict()}"
    )

    return True


def get_class_weights(y: pd.DataFrame) -> dict:
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        y: Target DataFrame
    
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y['target'])
    weights = compute_class_weight('balanced', classes=classes, y=y['target'])

    class_weights = dict(zip(classes, weights))
    print(f"📊 Class weights: {class_weights}")

    return class_weights
