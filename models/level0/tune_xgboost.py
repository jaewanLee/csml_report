"""
BTC Prediction Project - XGBoost Hyperparameter Tuning
Bayesian optimization for XGBoost hyperparameter tuning
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

warnings.filterwarnings('ignore')

# Import project utilities
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.data_utils import load_experiment_data, split_data_temporal, validate_data_quality, get_class_weights
from utils.cv_utils import create_time_series_cv, print_cv_splits_info
from config.settings import N_SPLITS, RANDOM_STATE, LOGS_DIR


def create_time_series_cv_for_tuning(X, y, n_splits=N_SPLITS):
    """
    Create TimeSeriesSplit for hyperparameter tuning
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv.split(X, y)


def tune_xgboost_hyperparameters(X_train, y_train, n_trials=50, cv_folds=3):
    """
    Tune XGBoost hyperparameters using Bayesian optimization
    
    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials
        cv_folds: Number of CV folds for tuning
    
    Returns:
        Best parameters and optimization results
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
    print(f"📊 Class counts -> neg: {num_negative}, pos: {num_positive}")
    print(f"⚖️  Scale pos weight (neg/pos): {scale_pos_weight:.4f}")

    # Define parameter search space - Slightly relaxed regularization for precision
    param_space = {
        'max_depth': Integer(3, 8),  # ↑ 여유 확대
        'learning_rate': Real(1e-4, 0.05, prior='log-uniform'),
        'n_estimators': Integer(600, 1800),  # 약간 좁힘
        'subsample': Real(0.6, 0.9),  # 살짝 상향
        'colsample_bytree': Real(0.6, 0.9),  # 살짝 상향
        'reg_alpha': Real(0.5, 50.0, prior='log-uniform'),  # 완화
        'reg_lambda': Real(0.5, 50.0, prior='log-uniform'),  # 완화
        'min_child_weight': Integer(3, 20),  # 완화
        'gamma': Real(0.5, 30.0),  # 완화
    }

    # Fixed parameters
    fixed_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE,
        'scale_pos_weight': scale_pos_weight,
        'n_jobs': -1,
        'verbosity': 0
    }

    # Create base estimator
    base_estimator = xgb.XGBClassifier(**fixed_params)

    # Create TimeSeriesSplit for tuning
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    # Create BayesSearchCV - Focus on Average Precision for better precision-recall balance
    bayes_search = BayesSearchCV(
        estimator=base_estimator,
        search_spaces=param_space,
        n_iter=n_trials,
        cv=tscv,
        scoring=
        'average_precision',  # Focus on precision-recall area under curve
        n_jobs=1,  # Use 1 job to avoid conflicts with XGBoost n_jobs
        random_state=RANDOM_STATE,
        verbose=1)

    # Fit the search
    print(f"\n🚀 Starting Bayesian optimization...")
    bayes_search.fit(X_train, y_train['target'])

    # Get results
    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_

    print(f"\n✅ Tuning completed!")
    print(f"🏆 Best CV F1-Score: {best_score:.4f}")
    print(f"🎯 Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Get optimization history
    optimization_results = pd.DataFrame(bayes_search.cv_results_)

    return {
        'best_params': best_params,
        'best_score': best_score,
        'optimization_results': optimization_results,
        'search_object': bayes_search
    }


def find_threshold_with_precision_constraint(y_true,
                                             y_proba,
                                             min_precision: float = 0.20,
                                             thresholds: list = [
                                                 0.10, 0.12, 0.14, 0.16, 0.18,
                                                 0.20, 0.22, 0.24, 0.26, 0.28,
                                                 0.30, 0.32, 0.34, 0.36, 0.38,
                                                 0.40
                                             ]):
    """
    Choose threshold maximizing recall while meeting a minimum precision constraint.
    Searches primarily in low-threshold region [0.10, 0.40].
    """
    best_threshold = 0.5
    best_recall = -1.0

    print("\n🔍 Finding threshold with precision constraint...")
    print(f"Target min precision: {min_precision:.2f}")
    print(f"{'Thr':<6} {'Prec':<8} {'Rec':<8} {'F1':<8}")

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        # Need both classes to compute stable metrics
        if len(np.unique(y_pred)) < 2:
            print(f"{thr:<6.2f} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
            continue

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"{thr:<6.2f} {prec:<8.4f} {rec:<8.4f} {f1:<8.4f}")

        if prec >= min_precision and rec > best_recall:
            best_recall = rec
            best_threshold = thr

    if best_recall < 0:
        # Fallback: choose threshold with highest recall even if precision < min_precision
        print(
            "⚠️ No threshold met the precision constraint. Falling back to max recall overall."
        )
        fallback_thr = None
        fallback_recall = -1.0
        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)
            if len(np.unique(y_pred)) < 2:
                continue
            rec = recall_score(y_true, y_pred, zero_division=0)
            if rec > fallback_recall:
                fallback_recall = rec
                fallback_thr = thr
        if fallback_thr is not None:
            best_threshold = fallback_thr

    print(
        f"\n✅ Chosen threshold: {best_threshold:.2f} (min_precision={min_precision:.2f})"
    )
    return best_threshold


def evaluate_tuned_model(X_train, y_train, X_test, y_test, best_params,
                         class_weights, optimal_threshold):
    """
    Evaluate the tuned model on both training and test sets
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        best_params: Best hyperparameters from tuning
        class_weights: Class weights
        optimal_threshold: Pre-determined optimal threshold (found using training data only)
    
    Returns:
        Evaluation results
    """
    print(f"\n🧪 Evaluating tuned model...")

    # Calculate scale_pos_weight using counts (negatives/positives)
    y_series = y_train['target'] if isinstance(y_train,
                                               pd.DataFrame) else y_train
    num_negative = int((y_series == 0).sum())
    num_positive = int((y_series == 1).sum())
    scale_pos_weight = num_negative / max(num_positive, 1)

    # Create model with best parameters
    model_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE,
        'scale_pos_weight': scale_pos_weight,
        'n_jobs': -1,
        'verbosity': 0,
        **best_params
    }

    model = xgb.XGBClassifier(**model_params)

    # Train on full training set
    model.fit(X_train, y_train['target'])

    # Get probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Use pre-determined optimal threshold (found using training data only)
    print(f"\nUsing pre-determined optimal threshold: {optimal_threshold:.2f}")

    # Evaluate with default threshold (0.5)
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

    # Evaluate with optimal threshold
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

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
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
            'roc_auc': train_auc
        },
        'train_metrics_optimal': {
            'f1_score': train_f1_opt,
            'precision': train_precision_opt,
            'recall': train_recall_opt
        },
        'test_metrics_default': {
            'f1_score': test_f1_default,
            'precision': test_precision_default,
            'recall': test_recall_default,
            'roc_auc': test_auc
        },
        'test_metrics_optimal': {
            'f1_score': test_f1_opt,
            'precision': test_precision_opt,
            'recall': test_recall_opt
        },
        'feature_importance': feature_importance
    }


def save_tuning_results(results, exp_id='A0'):
    """
    Save tuning results to files
    
    Args:
        results: Tuning results dictionary
        exp_id: Experiment ID
    """
    # Create results directory
    results_dir = LOGS_DIR / 'tuning'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = results_dir / f'xgboost_tuned_{exp_id}.pkl'
    joblib.dump(results['model'], model_path)
    print(f"💾 Model saved to: {model_path}")

    # Save optimization results
    opt_results_path = results_dir / f'xgboost_tuning_results_{exp_id}.pkl'
    joblib.dump(results['optimization_results'], opt_results_path)
    print(f"💾 Optimization results saved to: {opt_results_path}")

    # Save best parameters
    best_params_path = results_dir / f'xgboost_best_params_{exp_id}.pkl'
    joblib.dump(results['best_params'], best_params_path)
    print(f"💾 Best parameters saved to: {best_params_path}")

    # Save feature importance
    feature_importance_path = results_dir / f'xgboost_feature_importance_{exp_id}.pkl'
    joblib.dump(results['feature_importance'], feature_importance_path)
    print(f"💾 Feature importance saved to: {feature_importance_path}")


def main():
    """
    Main function to run XGBoost hyperparameter tuning
    """
    # ===== CONFIGURATION =====
    FEATURE_SET = 'A0'  # Change this to A0, A1, A2, A3, A4, A5, A6, A7, etc.
    # =========================

    print("🔧 BTC Prediction - XGBoost Hyperparameter Tuning")
    print("=" * 60)

    # Load specified feature set
    print(f"📁 Loading {FEATURE_SET} feature set...")
    try:
        X, y = load_experiment_data(FEATURE_SET)
        print(
            f"✅ Loaded {FEATURE_SET} data: {X.shape[0]} samples, {X.shape[1]} features"
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Validate data quality
    print("\n🔍 Validating data quality...")
    if not validate_data_quality(X, y):
        print("❌ Data quality validation failed")
        return

    # Split data temporally
    print("\n📊 Splitting data temporally...")
    X_train, X_test, y_train, y_test = split_data_temporal(X, y)
    print("⚠️  CRITICAL: Test set will ONLY be used for final evaluation")

    # Get class weights
    class_weights = get_class_weights(y_train)

    # Run hyperparameter tuning
    tuning_results = tune_xgboost_hyperparameters(
        X_train,
        y_train,
        n_trials=50,  # Adjust based on available time
        cv_folds=5  # Use 5 folds for consistent evaluation
    )

    # Find optimal threshold using CV predictions on Training Data (NO DATA LEAKAGE)
    print(
        "\n🔍 Finding optimal threshold using CV predictions on Training Data..."
    )
    from sklearn.model_selection import cross_val_predict

    # Use the same CV strategy as tuning - convert TimeSeriesSplit to list of splits
    tscv_iterator = TimeSeriesSplit(n_splits=5)  # Match cv_folds from tuning
    cv_splits = list(tscv_iterator.split(X_train, y_train['target']))

    y_train_cv_proba = cross_val_predict(
        tuning_results['search_object'].best_estimator_,
        X_train,
        y_train['target'],
        cv=cv_splits,  # Use list of splits instead of iterator
        method='predict_proba',
        n_jobs=1  # Match BayesSearch n_jobs
    )[:, 1]

    optimal_threshold = find_threshold_with_precision_constraint(
        y_train['target'],  # Use TRAINING target
        y_train_cv_proba,  # Use CV probabilities
        min_precision=0.20,
        thresholds=[
            0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30,
            0.32, 0.34, 0.36, 0.38, 0.40
        ])

    # Evaluate tuned model with pre-determined threshold
    eval_results = evaluate_tuned_model(X_train, y_train, X_test, y_test,
                                        tuning_results['best_params'],
                                        class_weights, optimal_threshold)

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
        'feature_importance': eval_results['feature_importance']
    }

    # Save results
    save_tuning_results(final_results, FEATURE_SET)

    # Performance assessment
    test_f1_default = eval_results['test_metrics_default']['f1_score']
    test_f1_optimal = eval_results['test_metrics_optimal']['f1_score']
    test_precision_optimal = eval_results['test_metrics_optimal']['precision']

    print(f"\n📋 Final Assessment:")
    print("=" * 30)
    print(f"Best CV Precision: {tuning_results['best_score']:.4f}")
    print(f"Test F1-Score (Default): {test_f1_default:.4f}")
    print(f"Test F1-Score (Optimal): {test_f1_optimal:.4f}")
    print(f"Test Precision (Optimal): {test_precision_optimal:.4f}")
    print(f"Optimal Threshold: {eval_results['optimal_threshold']:.1f}")

    if test_f1_optimal > 0.3:
        print(
            "✅ Performance improved! Ready to proceed with full experiments.")
    elif test_f1_optimal > 0.1:
        print(
            "🔄 Performance improved but still low. Consider more tuning or feature engineering."
        )
    else:
        print(
            "⚠️  Performance still very low. Consider fundamental changes to approach."
        )

    return final_results


if __name__ == "__main__":
    results = main()
