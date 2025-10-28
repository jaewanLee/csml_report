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

    # Define parameter search space
    param_space = {
        'max_depth': Integer(3, 10),
        # log-uniform requires lower bound > 0
        'learning_rate': Real(1e-3, 0.3, prior='log-uniform'),
        'n_estimators': Integer(50, 1000),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'reg_alpha': Real(1e-8, 10.0, prior='log-uniform'),
        'reg_lambda': Real(1e-8, 10.0, prior='log-uniform'),
        'min_child_weight': Integer(1, 10),
        # keep gamma uniform but avoid exact 0 lower bound for numerical stability
        'gamma': Real(1e-8, 5.0),
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

    # Create BayesSearchCV
    bayes_search = BayesSearchCV(
        estimator=base_estimator,
        search_spaces=param_space,
        n_iter=n_trials,
        cv=tscv,
        scoring='f1',
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


def evaluate_tuned_model(X_train, y_train, X_test, y_test, best_params,
                         class_weights):
    """
    Evaluate the tuned model on both training and test sets
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        best_params: Best hyperparameters from tuning
        class_weights: Class weights
    
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

    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    train_f1 = f1_score(y_train['target'], y_train_pred)
    train_precision = precision_score(y_train['target'], y_train_pred)
    train_recall = recall_score(y_train['target'], y_train_pred)
    train_auc = roc_auc_score(y_train['target'], y_train_proba)

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    test_f1 = f1_score(y_test['target'], y_test_pred)
    test_precision = precision_score(y_test['target'], y_test_pred)
    test_recall = recall_score(y_test['target'], y_test_pred)
    test_auc = roc_auc_score(y_test['target'], y_test_proba)

    print(f"\n📊 Tuned Model Results:")
    print("=" * 40)
    print(f"Training Set:")
    print(f"  F1-Score: {train_f1:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall: {train_recall:.4f}")
    print(f"  ROC-AUC: {train_auc:.4f}")
    print(f"\nTest Set:")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  ROC-AUC: {test_auc:.4f}")

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
        'train_metrics': {
            'f1_score': train_f1,
            'precision': train_precision,
            'recall': train_recall,
            'roc_auc': train_auc
        },
        'test_metrics': {
            'f1_score': test_f1,
            'precision': test_precision,
            'recall': test_recall,
            'roc_auc': test_auc
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
    print("🔧 BTC Prediction - XGBoost Hyperparameter Tuning")
    print("=" * 60)

    # Load A0 data
    print("📁 Loading A0 feature set...")
    try:
        X, y = load_experiment_data('A0')
        print(f"✅ Loaded A0 data: {X.shape[0]} samples, {X.shape[1]} features")
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
        cv_folds=3  # Use 3 folds for faster tuning
    )

    # Evaluate tuned model
    eval_results = evaluate_tuned_model(X_train, y_train, X_test, y_test,
                                        tuning_results['best_params'],
                                        class_weights)

    # Combine results
    final_results = {
        'best_params': tuning_results['best_params'],
        'best_cv_score': tuning_results['best_score'],
        'optimization_results': tuning_results['optimization_results'],
        'model': eval_results['model'],
        'train_metrics': eval_results['train_metrics'],
        'test_metrics': eval_results['test_metrics'],
        'feature_importance': eval_results['feature_importance']
    }

    # Save results
    save_tuning_results(final_results, 'A0')

    # Performance assessment
    test_f1 = eval_results['test_metrics']['f1_score']
    print(f"\n📋 Final Assessment:")
    print("=" * 20)
    print(f"Best CV F1-Score: {tuning_results['best_score']:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")

    if test_f1 > 0.3:
        print(
            "✅ Performance improved! Ready to proceed with full experiments.")
    elif test_f1 > 0.1:
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
