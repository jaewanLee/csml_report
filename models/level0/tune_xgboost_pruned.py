"""
BTC Prediction Project - XGBoost Hyperparameter Tuning (Pruned Features)
Bayesian optimization for XGBoost hyperparameter tuning with selected features
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import warnings
import joblib
from pathlib import Path
import sys
from typing import List

warnings.filterwarnings('ignore')

# Import project utilities
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.data_utils import load_experiment_data, split_data_temporal, validate_data_quality, get_class_weights
from config.settings import N_SPLITS, RANDOM_STATE, LOGS_DIR

# Global list of top features to keep. This will be assigned right before calling main().
TOP_FEATURES: List[str] = []


def tune_xgboost_hyperparameters(X_train, y_train, n_trials=50, cv_folds=3):
    """
    Tune XGBoost hyperparameters using Bayesian optimization
    """
    print(f"🔧 Starting XGBoost hyperparameter tuning...")
    print(
        f"📊 Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features"
    )
    print(
        f"🎯 Target distribution: {y_train['target'].value_counts().to_dict()}")
    print(f"🔄 Using {cv_folds}-fold TimeSeriesSplit for tuning")
    print(f"🔍 Running {n_trials} optimization trials")
    print("=" * 60)

    # Calculate scale_pos_weight using counts (negatives/positives)
    y_series = y_train['target'] if isinstance(y_train,
                                               pd.DataFrame) else y_train
    num_negative = int((y_series == 0).sum())
    num_positive = int((y_series == 1).sum())
    scale_pos_weight = num_negative / max(num_positive, 1)

    class_weights = get_class_weights(y_train)
    print(f"📊 Class weights: {class_weights}")
    print(f"📊 Class counts \u2192 neg: {num_negative}, pos: {num_positive}")
    print(f"⚖️  Scale pos weight (neg/pos): {scale_pos_weight:.4f}")

    # Define parameter search space
    param_space = {
        'max_depth': Integer(2, 7),
        'learning_rate': Real(1e-4, 0.05, prior='log-uniform'),
        'n_estimators': Integer(500, 2000),
        'subsample': Real(0.5, 0.8),
        'colsample_bytree': Real(0.5, 0.8),
        'reg_alpha': Real(1.0, 200.0, prior='log-uniform'),
        'reg_lambda': Real(1.0, 200.0, prior='log-uniform'),
        'min_child_weight': Integer(5, 50),
        'gamma': Real(1.0, 50.0),
    }

    # Fixed parameters
    fixed_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE,
        'scale_pos_weight': scale_pos_weight,
        'n_jobs': -1,
        'verbosity': 0,
    }

    base_estimator = xgb.XGBClassifier(**fixed_params)

    tscv = TimeSeriesSplit(n_splits=cv_folds)

    bayes_search = BayesSearchCV(
        estimator=base_estimator,
        search_spaces=param_space,
        n_iter=n_trials,
        cv=tscv,
        scoring='average_precision',
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    print(f"\n🚀 Starting Bayesian optimization...")
    bayes_search.fit(X_train, y_train['target'])

    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_

    print(f"\n✅ Tuning completed!")
    print(f"🏆 Best CV Average Precision: {best_score:.4f}")
    print(f"🎯 Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    optimization_results = pd.DataFrame(bayes_search.cv_results_)

    return {
        'best_params': best_params,
        'best_score': best_score,
        'optimization_results': optimization_results,
        'search_object': bayes_search,
    }


def find_optimal_threshold(
    y_true,
    y_proba,
    thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
):
    """
    Find optimal threshold for precision-recall balance
    """
    best_threshold = 0.5
    best_f1 = 0

    print(f"\n🔍 Finding optimal threshold...")
    print(
        f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 45)

    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        if len(np.unique(y_pred)) > 1:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            print(
                f"{threshold:<10.1f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        else:
            print(f"{threshold:<10.1f} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

    print(f"\n✅ Best threshold: {best_threshold:.1f} (F1: {best_f1:.4f})")
    return best_threshold


def evaluate_tuned_model(X_train, y_train, X_test, y_test, best_params,
                         class_weights):
    """
    Evaluate the tuned model on both training and test sets
    """
    print(f"\n🧪 Evaluating tuned model...")

    y_series = y_train['target'] if isinstance(y_train,
                                               pd.DataFrame) else y_train
    num_negative = int((y_series == 0).sum())
    num_positive = int((y_series == 1).sum())
    scale_pos_weight = num_negative / max(num_positive, 1)

    model_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE,
        'scale_pos_weight': scale_pos_weight,
        'n_jobs': -1,
        'verbosity': 0,
        **best_params,
    }

    model = xgb.XGBClassifier(**model_params)

    model.fit(X_train, y_train['target'])

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    optimal_threshold = find_optimal_threshold(y_test['target'], y_test_proba)

    y_train_pred_default = model.predict(X_train)
    y_test_pred_default = model.predict(X_test)

    train_f1_default = f1_score(y_train['target'], y_train_pred_default)
    train_precision_default = precision_score(y_train['target'],
                                              y_train_pred_default)
    train_recall_default = recall_score(y_train['target'],
                                        y_train_pred_default)
    train_auc = roc_auc_score(y_train['target'], y_train_proba)

    test_f1_default = f1_score(y_test['target'], y_test_pred_default)
    test_precision_default = precision_score(y_test['target'],
                                             y_test_pred_default)
    test_recall_default = recall_score(y_test['target'], y_test_pred_default)
    test_auc = roc_auc_score(y_test['target'], y_test_proba)

    y_train_pred_opt = (y_train_proba > optimal_threshold).astype(int)
    y_test_pred_opt = (y_test_proba > optimal_threshold).astype(int)

    train_f1_opt = f1_score(y_train['target'], y_train_pred_opt)
    train_precision_opt = precision_score(y_train['target'], y_train_pred_opt)
    train_recall_opt = recall_score(y_train['target'], y_train_pred_opt)

    test_f1_opt = f1_score(y_test['target'], y_test_pred_opt)
    test_precision_opt = precision_score(y_test['target'], y_test_pred_opt)
    test_recall_opt = recall_score(y_test['target'], y_test_pred_opt)

    print(f"\n📊 Tuned Model Results:")
    print("=" * 60)
    print(f"Training Set (Default Threshold 0.5):")
    print(f"  F1-Score: {train_f1_default:.4f}")
    print(f"  Precision: {train_precision_default:.4f}")
    print(f"  Recall: {train_recall_default:.4f}")
    print(f"  ROC-AUC: {train_auc:.4f}")

    print(f"\nTraining Set (Optimal Threshold {optimal_threshold:.1f}):")
    print(f"  F1-Score: {train_f1_opt:.4f}")
    print(f"  Precision: {train_precision_opt:.4f}")
    print(f"  Recall: {train_recall_opt:.4f}")

    print(f"\nTest Set (Default Threshold 0.5):")
    print(f"  F1-Score: {test_f1_default:.4f}")
    print(f"  Precision: {test_precision_default:.4f}")
    print(f"  Recall: {test_recall_default:.4f}")
    print(f"  ROC-AUC: {test_auc:.4f}")

    print(f"\nTest Set (Optimal Threshold {optimal_threshold:.1f}):")
    print(f"  F1-Score: {test_f1_opt:.4f}")
    print(f"  Precision: {test_precision_opt:.4f}")
    print(f"  Recall: {test_recall_opt:.4f}")

    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    print(f"\n🔍 Top 15 Feature Importances:")
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:<35} {row['importance']:.4f}")

    return {
        'model': model,
        'optimal_threshold': optimal_threshold,
        'train_metrics_default': {
            'f1_score': train_f1_default,
            'precision': train_precision_default,
            'recall': train_recall_default,
            'roc_auc': train_auc,
        },
        'train_metrics_optimal': {
            'f1_score': train_f1_opt,
            'precision': train_precision_opt,
            'recall': train_recall_opt,
        },
        'test_metrics_default': {
            'f1_score': test_f1_default,
            'precision': test_precision_default,
            'recall': test_recall_default,
            'roc_auc': test_auc,
        },
        'test_metrics_optimal': {
            'f1_score': test_f1_opt,
            'precision': test_precision_opt,
            'recall': test_recall_opt,
        },
        'feature_importance': feature_importance,
    }


def save_tuning_results(results, exp_id='A0'):
    """
    Save tuning results to files
    """
    results_dir = LOGS_DIR / 'tuning'
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = results_dir / f'xgboost_tuned_{exp_id}.pkl'
    joblib.dump(results['model'], model_path)
    print(f"💾 Model saved to: {model_path}")

    opt_results_path = results_dir / f'xgboost_tuning_results_{exp_id}.pkl'
    joblib.dump(results['optimization_results'], opt_results_path)
    print(f"💾 Optimization results saved to: {opt_results_path}")

    best_params_path = results_dir / f'xgboost_best_params_{exp_id}.pkl'
    joblib.dump(results['best_params'], best_params_path)
    print(f"💾 Best parameters saved to: {best_params_path}")

    feature_importance_path = results_dir / f'xgboost_feature_importance_{exp_id}.pkl'
    joblib.dump(results['feature_importance'], feature_importance_path)
    print(f"💾 Feature importance saved to: {feature_importance_path}")


def save_pruned_dataset(X_pruned, exp_id='A0'):
    """
    Save pruned feature matrix to features directory
    """
    features_dir = Path(__file__).parent.parent.parent / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)

    X_pruned_path = features_dir / f'{exp_id}_pruned.parquet'
    X_pruned.to_parquet(X_pruned_path)
    print(f"💾 Pruned features saved to: {X_pruned_path}")

    return X_pruned_path


def main():
    """
    Main function to run XGBoost hyperparameter tuning with pruned features
    """
    print("🔧 BTC Prediction - XGBoost Hyperparameter Tuning (Pruned Features)")
    print("=" * 70)

    # Load A1 data
    print("📁 Loading A1 feature set...")
    try:
        X, y = load_experiment_data('A1')
        print(f"✅ Loaded A1 data: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Resolve top features to use
    top_features = TOP_FEATURES
    if not top_features:
        print(
            "⚠️  TOP_FEATURES not provided. Using all features (no pruning).")
        X_pruned = X.copy()
        available_features = list(X_pruned.columns)
    else:
        print(f"\n✂️  Pruning features to top {len(top_features)} features...")
        print(f"📊 Original features: {X.shape[1]}")

        available_features = [f for f in top_features if f in X.columns]
        missing_features = [f for f in top_features if f not in X.columns]

        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")

        print(
            f"✅ Available features: {len(available_features)}/{len(top_features)}"
        )
        X_pruned = X[available_features].copy()
        print(f"📊 Pruned features: {X_pruned.shape[1]}")

    # Save pruned dataset
    pruned_path = save_pruned_dataset(X_pruned, 'A1')

    # Validate data quality
    print("\n🔍 Validating pruned data quality...")
    if not validate_data_quality(X_pruned, y):
        print("❌ Pruned data quality validation failed")
        return

    # Split data temporally
    print("\n📊 Splitting pruned data temporally...")
    X_train, X_test, y_train, y_test = split_data_temporal(X_pruned, y)
    print("⚠️  CRITICAL: Test set will ONLY be used for final evaluation")

    # Get class weights
    class_weights = get_class_weights(y_train)

    # Run hyperparameter tuning
    tuning_results = tune_xgboost_hyperparameters(
        X_train,
        y_train,
        n_trials=50,
        cv_folds=5,
    )

    # Evaluate tuned model
    eval_results = evaluate_tuned_model(
        X_train,
        y_train,
        X_test,
        y_test,
        tuning_results['best_params'],
        class_weights,
    )

    # Combine results
    final_results = {
        'best_params': tuning_results['best_params'],
        'best_cv_score': tuning_results['best_score'],
        'optimization_results': tuning_results['optimization_results'],
        'model': eval_results['model'],
        'optimal_threshold': eval_results['optimal_threshold'],
        'train_metrics_default': eval_results['train_metrics_default'],
        'train_metrics_optimal': eval_results['train_metrics_optimal'],
        'test_metrics_default': eval_results['test_metrics_default'],
        'test_metrics_optimal': eval_results['test_metrics_optimal'],
        'feature_importance': eval_results['feature_importance'],
        'pruned_features': available_features,
        'pruned_dataset_path': pruned_path,
    }

    # Save results
    save_tuning_results(final_results, 'A1_pruned')

    # Performance assessment
    test_f1_default = eval_results['test_metrics_default']['f1_score']
    test_f1_optimal = eval_results['test_metrics_optimal']['f1_score']
    test_precision_optimal = eval_results['test_metrics_optimal']['precision']

    print(f"\n📋 Final Assessment:")
    print("=" * 30)
    print(f"Best CV Average Precision: {tuning_results['best_score']:.4f}")
    print(f"Test F1-Score (Default): {test_f1_default:.4f}")
    print(f"Test F1-Score (Optimal): {test_f1_optimal:.4f}")
    print(f"Test Precision (Optimal): {test_precision_optimal:.4f}")
    print(f"Optimal Threshold: {eval_results['optimal_threshold']:.1f}")
    print(
        f"Pruned Features: {len(available_features)}/{len(top_features) if top_features else X.shape[1]}"
    )

    return final_results


if __name__ == "__main__":
    # Define top features right before calling main so it's easy to edit later
    TOP_FEATURES = [
        'ATR_norm', 'D1_ATR_norm', 'D1_close_vs_lagging_span_pct',
        'D1_MA_7_MA_60_norm', 'D1_MACD_line', 'D1_close_vs_leading_span_A_pct',
        'D1_band_width', 'MA_20_MA_120_norm', 'D1_MACD_signal', 'range_pct',
        'D1_MA_14_MA_60_norm', 'D1_MA_14_MA_20_norm', 'D1_MA_7_MA_20_norm',
        'D1_RSI', 'MA_14_MA_120_norm', 'D1_span_A_vs_span_B_pct',
        'D1_MA_60_norm', 'D1_MA_20_MA_120_norm', 'D1_MA_20_MA_60_norm',
        'MACD_line', 'D1_close_vs_baseline_pct', 'ATR_norm_ma',
        'MA_14_MA_60_norm', 'MACD_signal'
    ]

    results = main()
