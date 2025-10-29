#!/usr/bin/env python3
"""
Test other models (Random Forest, Logistic Regression) on A1_pruned
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.settings import *
from utils.data_utils import load_experiment_data, split_data_temporal
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_score, recall_score,
                             f1_score, average_precision_score)


def find_optimal_threshold(
        y_true,
        y_proba,
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]):
    """Find optimal threshold for F1-score"""
    print("\n🔍 Finding optimal threshold...")
    print("Threshold  Precision  Recall     F1-Score  ")
    print("---------------------------------------------")

    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if len(np.unique(y_pred)) < 2:
            print(f"{threshold:<9.1f} N/A        N/A        N/A       ")
            continue

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"{threshold:<9.1f} {precision:<9.4f} {recall:<9.4f} {f1:<9.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\n✅ Best threshold: {best_threshold} (F1: {best_f1:.4f})")
    return best_threshold


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model with both default and optimal thresholds"""
    print(f"\n🧪 Evaluating {model_name}...")

    # Get predictions
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_test['target'], y_test_proba)

    # Evaluate with default threshold (0.5)
    y_train_pred_default = (y_train_proba >= 0.5).astype(int)
    y_test_pred_default = (y_test_proba >= 0.5).astype(int)

    # Evaluate with optimal threshold
    y_train_pred_optimal = (y_train_proba >= optimal_threshold).astype(int)
    y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)

    # Calculate metrics for default threshold
    train_metrics_default = {
        'f1_score': f1_score(y_train['target'], y_train_pred_default),
        'precision': precision_score(y_train['target'], y_train_pred_default),
        'recall': recall_score(y_train['target'], y_train_pred_default),
        'roc_auc': roc_auc_score(y_train['target'], y_train_proba)
    }

    test_metrics_default = {
        'f1_score': f1_score(y_test['target'], y_test_pred_default),
        'precision': precision_score(y_test['target'], y_test_pred_default),
        'recall': recall_score(y_test['target'], y_test_pred_default),
        'roc_auc': roc_auc_score(y_test['target'], y_test_proba)
    }

    # Calculate metrics for optimal threshold
    train_metrics_optimal = {
        'f1_score': f1_score(y_train['target'], y_train_pred_optimal),
        'precision': precision_score(y_train['target'], y_train_pred_optimal),
        'recall': recall_score(y_train['target'], y_train_pred_optimal)
    }

    test_metrics_optimal = {
        'f1_score': f1_score(y_test['target'], y_test_pred_optimal),
        'precision': precision_score(y_test['target'], y_test_pred_optimal),
        'recall': recall_score(y_test['target'], y_test_pred_optimal)
    }

    # Print results
    print(f"\n📊 {model_name} Results:")
    print("=" * 60)
    print("Training Set (Default Threshold 0.5):")
    print(f"  F1-Score: {train_metrics_default['f1_score']:.4f}")
    print(f"  Precision: {train_metrics_default['precision']:.4f}")
    print(f"  Recall: {train_metrics_default['recall']:.4f}")
    print(f"  ROC-AUC: {train_metrics_default['roc_auc']:.4f}")

    print(f"\nTraining Set (Optimal Threshold {optimal_threshold:.1f}):")
    print(f"  F1-Score: {train_metrics_optimal['f1_score']:.4f}")
    print(f"  Precision: {train_metrics_optimal['precision']:.4f}")
    print(f"  Recall: {train_metrics_optimal['recall']:.4f}")

    print(f"\nTest Set (Default Threshold 0.5):")
    print(f"  F1-Score: {test_metrics_default['f1_score']:.4f}")
    print(f"  Precision: {test_metrics_default['precision']:.4f}")
    print(f"  Recall: {test_metrics_default['recall']:.4f}")
    print(f"  ROC-AUC: {test_metrics_default['roc_auc']:.4f}")

    print(f"\nTest Set (Optimal Threshold {optimal_threshold:.1f}):")
    print(f"  F1-Score: {test_metrics_optimal['f1_score']:.4f}")
    print(f"  Precision: {test_metrics_optimal['precision']:.4f}")
    print(f"  Recall: {test_metrics_optimal['recall']:.4f}")

    return {
        'train_metrics_default': train_metrics_default,
        'test_metrics_default': test_metrics_default,
        'train_metrics_optimal': train_metrics_optimal,
        'test_metrics_optimal': test_metrics_optimal,
        'optimal_threshold': optimal_threshold
    }


def test_random_forest(X_train, X_test, y_train, y_test):
    """Test Random Forest with hyperparameter tuning"""
    print("\n🌲 Testing Random Forest...")

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    # Create Random Forest
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)

    # Grid search with TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(rf,
                               param_grid,
                               cv=tscv,
                               scoring='f1',
                               n_jobs=1,
                               verbose=1)

    print("🔍 Tuning Random Forest hyperparameters...")
    grid_search.fit(X_train, y_train['target'])

    print(f"✅ Best CV F1-Score: {grid_search.best_score_:.4f}")
    print("🎯 Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    # Evaluate best model
    results = evaluate_model(grid_search.best_estimator_, X_train, X_test,
                             y_train, y_test, "Random Forest")

    # Save model
    model_path = LOGS_DIR / 'tuning' / 'random_forest_A1_pruned.pkl'
    os.makedirs(LOGS_DIR / 'tuning', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    print(f"💾 Model saved to: {model_path}")

    return results


def test_logistic_regression(X_train, X_test, y_train, y_test):
    """Test Logistic Regression with hyperparameter tuning"""
    print("\n📈 Testing Logistic Regression...")

    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced', None]
    }

    # Create Logistic Regression
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)

    # Grid search with TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(lr,
                               param_grid,
                               cv=tscv,
                               scoring='f1',
                               n_jobs=1,
                               verbose=1)

    print("🔍 Tuning Logistic Regression hyperparameters...")
    grid_search.fit(X_train, y_train['target'])

    print(f"✅ Best CV F1-Score: {grid_search.best_score_:.4f}")
    print("🎯 Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    # Evaluate best model
    results = evaluate_model(grid_search.best_estimator_, X_train, X_test,
                             y_train, y_test, "Logistic Regression")

    # Save model
    model_path = LOGS_DIR / 'tuning' / 'logistic_regression_A1_pruned.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    print(f"💾 Model saved to: {model_path}")

    return results


def main():
    """Main function to test other models"""
    print("🔧 BTC Prediction - Testing Other Models on A1_pruned")
    print("=" * 70)

    # Load A1_pruned data
    print("📁 Loading A1_pruned feature set...")
    try:
        X, y = load_experiment_data('A1_pruned')
        print(
            f"✅ Loaded A1_pruned data: {X.shape[0]} samples, {X.shape[1]} features"
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Split data temporally
    print("\n📊 Splitting data temporally...")
    X_train, X_test, y_train, y_test = split_data_temporal(X, y)

    print("Data split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Features: {X_train.shape[1]}")

    # Test Random Forest
    rf_results = test_random_forest(X_train, X_test, y_train, y_test)

    # Test Logistic Regression
    lr_results = test_logistic_regression(X_train, X_test, y_train, y_test)

    # Compare results
    print("\n📋 Model Comparison:")
    print("=" * 50)
    print("Model                F1-Score  Precision  Recall    ROC-AUC")
    print("------------------------------------------------------------")
    print(
        f"Random Forest        {rf_results['test_metrics_optimal']['f1_score']:.4f}     {rf_results['test_metrics_optimal']['precision']:.4f}     {rf_results['test_metrics_optimal']['recall']:.4f}     {rf_results['test_metrics_default']['roc_auc']:.4f}"
    )
    print(
        f"Logistic Regression  {lr_results['test_metrics_optimal']['f1_score']:.4f}     {lr_results['test_metrics_optimal']['precision']:.4f}     {lr_results['test_metrics_optimal']['recall']:.4f}     {lr_results['test_metrics_default']['roc_auc']:.4f}"
    )

    # Find best model
    rf_f1 = rf_results['test_metrics_optimal']['f1_score']
    lr_f1 = lr_results['test_metrics_optimal']['f1_score']

    if rf_f1 > lr_f1:
        print(f"\n🏆 Best Model: Random Forest (F1: {rf_f1:.4f})")
    else:
        print(f"\n🏆 Best Model: Logistic Regression (F1: {lr_f1:.4f})")


if __name__ == "__main__":
    main()
