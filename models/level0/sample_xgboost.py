"""
BTC Prediction Project - Sample XGBoost Model
Simple XGBoost implementation for A0 feature testing
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Import project utilities
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.data_utils import load_experiment_data, split_data_temporal, validate_data_quality, get_class_weights
from utils.cv_utils import create_time_series_cv, print_cv_splits_info
from config.settings import N_SPLITS, RANDOM_STATE


def train_xgboost_model(X_train, y_train, class_weights=None):
    """
    Train XGBoost model with basic parameters (no validation set to prevent data leakage)
    
    Args:
        X_train: Training features
        y_train: Training target
        class_weights: Dictionary of class weights
    
    Returns:
        Trained XGBoost model
    """
    # Calculate scale_pos_weight for class imbalance
    if class_weights is not None:
        scale_pos_weight = class_weights[1] / class_weights[
            0]  # SELL / REST ratio
    else:
        scale_pos_weight = 1.0

    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'scale_pos_weight': scale_pos_weight,
        'n_jobs': -1
    }

    print(f"🚀 Training XGBoost with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Create and train model (no early stopping to avoid validation set usage)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, model_name="XGBoost"):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name for logging
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\n📊 {model_name} Test Results:")
    print("=" * 50)
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['REST', 'SELL']))

    return {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def run_cross_validation(X, y, n_splits=N_SPLITS):
    """
    Run cross-validation on the dataset using TimeSeriesSplit to prevent data leakage
    
    Args:
        X: Features DataFrame (TRAINING SET ONLY)
        y: Target DataFrame (TRAINING SET ONLY)
        n_splits: Number of CV splits
    
    Returns:
        Dictionary of CV results
    """
    print(f"\n🔄 Running {n_splits}-fold TimeSeriesSplit Cross-Validation")
    print(
        "⚠️  CRITICAL: Using only training data to prevent future information leakage"
    )
    print("=" * 60)

    cv_scores = []
    fold_results = []

    for fold, (train_idx,
               val_idx) in enumerate(create_time_series_cv(X, y, n_splits)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        # Split data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        print(
            f"Train: {len(X_train_fold)} samples, Val: {len(X_val_fold)} samples"
        )

        # Get class weights for this fold
        class_weights = get_class_weights(y_train_fold)

        # Train model (no validation set to prevent data leakage)
        model = train_xgboost_model(X_train_fold, y_train_fold, class_weights)

        # Evaluate on validation set
        val_results = evaluate_model(model, X_val_fold, y_val_fold,
                                     f"XGBoost Fold {fold + 1}")

        # Store results
        cv_scores.append(val_results['f1_score'])
        fold_results.append({
            'fold': fold + 1,
            'f1_score': val_results['f1_score'],
            'precision': val_results['precision'],
            'recall': val_results['recall']
        })

    # Calculate CV statistics
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    print(f"\n📈 Cross-Validation Results:")
    print("=" * 30)
    print(f"Mean F1-Score: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Min F1-Score: {np.min(cv_scores):.4f}")
    print(f"Max F1-Score: {np.max(cv_scores):.4f}")

    return {
        'cv_scores': cv_scores,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'fold_results': fold_results
    }


def main():
    """
    Main function to run XGBoost test on A0 features
    """
    print("🚀 BTC Prediction - XGBoost Sample Test")
    print("=" * 50)

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

    # Print CV splits info
    print_cv_splits_info(X, y)

    # Split data temporally
    print("\n📊 Splitting data temporally...")
    X_train, X_test, y_train, y_test = split_data_temporal(X, y)
    print(
        "⚠️  CRITICAL: Test set will ONLY be used for final evaluation, never for training/validation"
    )

    # Get class weights
    class_weights = get_class_weights(y_train)

    # Run cross-validation on training data ONLY (prevents data leakage)
    cv_results = run_cross_validation(X_train, y_train)

    # Train final model on full training set (no test set usage to prevent data leakage)
    print(f"\n🎯 Training final model on full training set...")
    print("⚠️  CRITICAL: Final model trained ONLY on training data")
    final_model = train_xgboost_model(X_train, y_train, class_weights)

    # Evaluate on test set (FIRST TIME test set is used)
    print(f"\n🧪 Evaluating on test set (first time test set is accessed)...")
    test_results = evaluate_model(final_model, X_test, y_test, "Final XGBoost")

    # Feature importance
    print(f"\n🔍 Top 10 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature':
        X.columns,
        'importance':
        final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")

    # Summary
    print(f"\n📋 Summary:")
    print("=" * 20)
    print(
        f"CV F1-Score: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}"
    )
    print(f"Test F1-Score: {test_results['f1_score']:.4f}")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test Recall: {test_results['recall']:.4f}")

    # Performance assessment
    if test_results['f1_score'] > 0.5:
        print(
            "✅ Performance looks promising! Ready to proceed with full experiments."
        )
    else:
        print(
            "⚠️  Performance is low. Consider feature engineering improvements."
        )

    return {
        'cv_results': cv_results,
        'test_results': test_results,
        'model': final_model,
        'feature_importance': feature_importance
    }


if __name__ == "__main__":
    results = main()
