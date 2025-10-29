#!/usr/bin/env python3
"""
BTC Prediction - XGBoost Hyperparameter Tuning with Feature Selection
A1 피처 세트에서 상위 15-20개 피처만 선택하여 precision에 집중
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings

# Suppress XGBoost warnings
os.environ['XGBOOST_VERBOSE'] = '0'
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.settings import *
from utils.data_utils import load_experiment_data, split_data_temporal, validate_data_quality as du_validate_data_quality
from skopt import BayesSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_score, recall_score,
                             f1_score, average_precision_score)
from skopt.space import Real, Integer, Categorical
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import TimeSeriesSplit


def validate_data_quality(X, y):
    """Validate data quality before training"""
    print("🔍 Data Quality Validation:")

    # Check for missing values
    if X.isnull().any().any():
        print("  ❌ Missing values in features")
        return False
    else:
        print("  ✅ No missing values in features")

    if y.isnull().any().any():
        print("  ❌ Missing values in target")
        return False
    else:
        print("  ✅ No missing values in target")

    # Check for infinite values
    if np.isinf(X).any().any():
        print("  ❌ Infinite values in features")
        return False
    else:
        print("  ✅ No infinite values in features")

    if np.isinf(y).any().any():
        print("  ❌ Infinite values in target")
        return False
    else:
        print("  ✅ No infinite values in target")

    print(f"  📊 Features shape: {X.shape}")
    print(f"  📊 Target shape: {y.shape}")
    print(f"  📊 Target distribution: {y['target'].value_counts().to_dict()}")

    return True


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


def evaluate_tuned_model(model, X_train, X_test, y_train, y_test,
                         selected_features):
    """Evaluate the tuned model with both default and optimal thresholds"""
    print("\n🧪 Evaluating tuned model...")

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

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Print results
    print("\n📊 Tuned Model Results:")
    print("=" * 60)
    print("Training Set (Default Threshold 0.5):")
    print(f"  F1-Score: {train_metrics_default['f1_score']:.4f}")
    print(f"  Precision: {train_metrics_default['precision']:.4f}")
    print(f"  Recall: {train_metrics_default['recall']:.4f}")
    print(f"  ROC-AUC: {train_metrics_default['roc_auc']:.4f}")

    print(
        "\nTraining Set (Optimal Threshold {:.1f}):".format(optimal_threshold))
    print(f"  F1-Score: {train_metrics_optimal['f1_score']:.4f}")
    print(f"  Precision: {train_metrics_optimal['precision']:.4f}")
    print(f"  Recall: {train_metrics_optimal['recall']:.4f}")

    print("\nTest Set (Default Threshold 0.5):")
    print(f"  F1-Score: {test_metrics_default['f1_score']:.4f}")
    print(f"  Precision: {test_metrics_default['precision']:.4f}")
    print(f"  Recall: {test_metrics_default['recall']:.4f}")
    print(f"  ROC-AUC: {test_metrics_default['roc_auc']:.4f}")

    print("\nTest Set (Optimal Threshold {:.1f}):".format(optimal_threshold))
    print(f"  F1-Score: {test_metrics_optimal['f1_score']:.4f}")
    print(f"  Precision: {test_metrics_optimal['precision']:.4f}")
    print(f"  Recall: {test_metrics_optimal['recall']:.4f}")

    # Print top feature importances
    print("\n🔍 Top 15 Feature Importances:")
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:<40} {row['importance']:.4f}")

    return {
        'train_metrics_default': train_metrics_default,
        'test_metrics_default': test_metrics_default,
        'train_metrics_optimal': train_metrics_optimal,
        'test_metrics_optimal': test_metrics_optimal,
        'feature_importance': feature_importance,
        'optimal_threshold': optimal_threshold
    }


def save_tuning_results(results, experiment_name):
    """Save tuning results to files"""
    # Create directories
    os.makedirs(LOGS_DIR / 'tuning', exist_ok=True)

    # Save model
    model_path = LOGS_DIR / 'tuning' / f'xgboost_tuned_{experiment_name}_pruned.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(results['model'], f)
    print(f"💾 Model saved to: {model_path}")

    # Save optimization results
    opt_path = LOGS_DIR / 'tuning' / f'xgboost_tuning_results_{experiment_name}_pruned.pkl'
    with open(opt_path, 'wb') as f:
        pickle.dump(results['optimization_results'], f)
    print(f"💾 Optimization results saved to: {opt_path}")

    # Save best parameters
    params_path = LOGS_DIR / 'tuning' / f'xgboost_best_params_{experiment_name}_pruned.pkl'
    with open(params_path, 'wb') as f:
        pickle.dump(results['best_params'], f)
    print(f"💾 Best parameters saved to: {params_path}")

    # Save feature importance
    importance_path = LOGS_DIR / 'tuning' / f'xgboost_feature_importance_{experiment_name}_pruned.pkl'
    with open(importance_path, 'wb') as f:
        pickle.dump(results['feature_importance'], f)
    print(f"💾 Feature importance saved to: {importance_path}")


def main():
    """
    Main function to run XGBoost hyperparameter tuning with feature selection
    """
    print("🔧 BTC Prediction - XGBoost Hyperparameter Tuning (Pruned)")
    print("=" * 70)

    # Load A1 data (H4 + D1)
    print("📁 Loading A1 feature set...")
    try:
        X, y = load_experiment_data('A1')
        print(f"✅ Loaded A1 data: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Validate data quality
    print("\n🔍 Validating data quality...")
    if not validate_data_quality(X, y):
        print("❌ Data quality validation failed")
        return

    # Split data temporally FIRST (to prevent data leakage)
    print("\n📊 Splitting data temporally...")
    X_train_full, X_test_full, y_train, y_test = split_data_temporal(X, y)

    # Select top features USING ONLY TRAINING DATA (to prevent data leakage)
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
    print("🔍 Top 10 selected features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:<40} {row['score']:.4f}")

    # Convert back to DataFrame (important!)
    X_train = pd.DataFrame(X_train_selected,
                           index=y_train.index,
                           columns=selected_features)
    X_test = pd.DataFrame(X_test_selected,
                          index=y_test.index,
                          columns=selected_features)

    print("Data split:")
    print(f"  Training: {len(X_train)} samples ({TRAIN_START} to {TRAIN_END})")
    print(f"  Test: {len(X_test)} samples ({TEST_START} to {TEST_END})")
    print("⚠️  CRITICAL: Test set will ONLY be used for final evaluation")

    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(y_train['target']),
                                         y=y_train['target'])
    class_weight_dict = dict(zip(np.unique(y_train['target']), class_weights))
    print(f"📊 Class weights: {class_weight_dict}")

    # Calculate scale_pos_weight for XGBoost
    y_series = y_train['target'] if isinstance(y_train,
                                               pd.DataFrame) else y_train
    num_negative = int((y_series == 0).sum())
    num_positive = int((y_series == 1).sum())
    scale_pos_weight = num_negative / max(num_positive, 1)

    print(f"📊 Class counts -> neg: {num_negative}, pos: {num_positive}")
    print(f"⚖️  Scale pos weight (neg/pos): {scale_pos_weight:.4f}")

    # Start hyperparameter tuning
    print("\n🔧 Starting XGBoost hyperparameter tuning...")
    print(
        f"📊 Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features"
    )
    print(
        f"🎯 Target distribution: {y_train['target'].value_counts().to_dict()}")
    print(f"🔄 Using {N_SPLITS}-fold TimeSeriesSplit for tuning")
    print(f"🔍 Running 50 optimization trials")
    print("=" * 70)

    # Create TimeSeriesSplit iterator for cross-validation
    tscv_iterator = TimeSeriesSplit(n_splits=N_SPLITS)

    # Define parameter search space - Focus on precision with strong regularization
    param_space = {
        'max_depth': Integer(2, 6),  # Conservative depth
        'learning_rate': Real(1e-4, 0.1,
                              prior='log-uniform'),  # Low learning rate
        'n_estimators': Integer(200, 1500),  # Wide range
        'subsample': Real(0.5, 0.9),  # More aggressive subsampling
        'colsample_bytree': Real(0.5, 0.9),  # More aggressive column sampling
        'reg_alpha': Real(1.0, 100.0,
                          prior='log-uniform'),  # Strong L1 regularization
        'reg_lambda': Real(1.0, 100.0,
                           prior='log-uniform'),  # Strong L2 regularization
        'min_child_weight': Integer(1, 20),  # Higher min_child_weight
        'gamma': Real(0.1, 10.0)  # Strong pruning
    }

    # Create XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight,
                                       random_state=RANDOM_STATE,
                                       eval_metric='logloss',
                                       verbosity=0)

    # Create Bayesian search
    print("🚀 Starting Bayesian optimization...")
    bayes_search = BayesSearchCV(
        xgb_classifier,
        param_space,
        n_iter=50,
        cv=tscv_iterator,  # Use the correct iterator
        scoring='f1',  # Focus on F1-score for balanced precision-recall
        random_state=RANDOM_STATE,
        n_jobs=1,  # Avoid parallel processing conflicts
        verbose=0)

    # Fit the model
    bayes_search.fit(X_train, y_train['target'])

    print("✅ Tuning completed!")
    print(f"🏆 Best CV F1-Score: {bayes_search.best_score_:.4f}")
    print("🎯 Best parameters:")
    for param, value in bayes_search.best_params_.items():
        print(f"  {param}: {value}")

    # Evaluate the tuned model
    eval_results = evaluate_tuned_model(bayes_search.best_estimator_, X_train,
                                        X_test, y_train, y_test,
                                        selected_features)

    # Prepare final results
    final_results = {
        'model': bayes_search.best_estimator_,
        'optimization_results': bayes_search,
        'best_params': bayes_search.best_params_,
        'best_score': bayes_search.best_score_,
        'selected_features': selected_features,
        'feature_importance': feature_importance,
        'train_metrics_default': eval_results['train_metrics_default'],
        'test_metrics_default': eval_results['test_metrics_default'],
        'train_metrics_optimal': eval_results['train_metrics_optimal'],
        'test_metrics_optimal': eval_results['test_metrics_optimal']
    }

    # Save results
    save_tuning_results(final_results, 'A1')

    # Performance assessment
    test_f1_default = eval_results['test_metrics_default']['f1_score']
    test_f1_optimal = eval_results['test_metrics_optimal']['f1_score']
    test_precision_optimal = eval_results['test_metrics_optimal']['precision']

    print(f"\n📋 Final Assessment:")
    print("=" * 30)
    print(f"Best CV F1-Score: {bayes_search.best_score_:.4f}")
    print(f"Test F1-Score (Default): {test_f1_default:.4f}")
    print(f"Test F1-Score (Optimal): {test_f1_optimal:.4f}")
    print(f"Test Precision (Optimal): {test_precision_optimal:.4f}")
    print(f"Optimal Threshold: {eval_results['optimal_threshold']:.1f}")

    if test_f1_optimal > 0.2:
        print("🎉 Good performance achieved!")
    elif test_f1_optimal > 0.15:
        print(
            "🔄 Performance improved but still low. Consider more tuning or feature engineering."
        )
    else:
        print("❌ Performance still very low. Consider different approach.")


if __name__ == "__main__":
    main()
