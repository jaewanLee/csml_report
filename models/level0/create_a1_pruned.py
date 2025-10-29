#!/usr/bin/env python3
"""
Create A1_pruned.parquet with top 20 features selected from A1
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.settings import *
from utils.data_utils import load_experiment_data, split_data_temporal
from sklearn.feature_selection import SelectKBest, f_classif


def create_a1_pruned():
    """Create A1_pruned.parquet with top 20 features"""
    print("🔧 Creating A1_pruned.parquet with top 20 features")
    print("=" * 60)

    # Load A1 data
    print("📁 Loading A1 feature set...")
    try:
        X, y = load_experiment_data('A1')
        print(f"✅ Loaded A1 data: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Split data temporally (to prevent data leakage)
    print("\n📊 Splitting data temporally...")
    X_train_full, X_test_full, y_train, y_test = split_data_temporal(X, y)

    # Select top 20 features using only training data
    print("\n🔍 Feature Selection (using Training Data only)...")
    selector = SelectKBest(score_func=f_classif, k=20)
    X_train_selected = selector.fit_transform(X_train_full, y_train['target'])
    X_test_selected = selector.transform(X_test_full)

    # Get selected feature names
    selected_features = X_train_full.columns[selector.get_support()].tolist()
    feature_scores = selector.scores_[selector.get_support()]

    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'score': feature_scores
    }).sort_values('score', ascending=False)

    print(f"✅ Selected {len(selected_features)} features")
    print("🔍 Top 20 selected features:")
    for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:<40} {row['score']:.4f}")

    # Convert back to DataFrame
    X_train = pd.DataFrame(X_train_selected,
                           index=y_train.index,
                           columns=selected_features)
    X_test = pd.DataFrame(X_test_selected,
                          index=y_test.index,
                          columns=selected_features)

    # Combine train and test data
    X_combined = pd.concat([X_train, X_test])
    y_combined = pd.concat([y_train, y_test])

    # Ensure same index order
    common_index = X_combined.index.intersection(y_combined.index)
    X_combined = X_combined.loc[common_index]
    y_combined = y_combined.loc[common_index]

    # Save A1_pruned features
    output_path = FEATURES_DIR / 'A1_pruned.parquet'
    X_combined.to_parquet(output_path)
    print(f"\n💾 A1_pruned features saved to: {output_path}")
    print(f"📊 Final shape: {X_combined.shape}")

    # Save feature importance
    importance_path = FEATURES_DIR / 'A1_pruned_feature_importance.csv'
    feature_importance.to_csv(importance_path, index=False)
    print(f"💾 Feature importance saved to: {importance_path}")

    # Update FEATURE_SETS in settings if needed
    print(f"\n📝 Add this to config/settings.py FEATURE_SETS:")
    print(f"    'A1_pruned': FEATURES_DIR / 'A1_pruned.parquet',")

    return X_combined, y_combined, selected_features


if __name__ == "__main__":
    create_a1_pruned()
