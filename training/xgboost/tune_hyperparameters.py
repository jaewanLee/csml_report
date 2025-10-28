#!/usr/bin/env python3
"""
XGBoost Hyperparameter Tuning Script
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.level0.xgboost_gemini_model import XGBoostGeminiModel
from utils.data_utils import load_experiment_data, split_data_temporal, validate_data_quality
from utils.cv_utils import print_cv_splits_info
from config.settings import LOGS_DIR, RANDOM_STATE
import joblib
import pandas as pd


def setup_logging(exp_id: str):
    """Setup logging for hyperparameter tuning"""
    log_dir = LOGS_DIR / "tuning"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"xgboost_tuning_{exp_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file),
                  logging.StreamHandler()])

    return logging.getLogger(f"xgboost_tuning_{exp_id}")


def tune_hyperparameters(exp_id: str, n_splits: int = 5, n_iter: int = 50):
    """
    Tune XGBoost hyperparameters for given experiment
    
    Args:
        exp_id: Experiment ID (A0, A1, A2, A3, A4_Pruned)
        n_splits: Number of TimeSeriesSplit folds
        n_iter: Number of Bayesian optimization iterations
    """
    logger = setup_logging(exp_id)

    logger.info(f"🚀 Starting XGBoost hyperparameter tuning for {exp_id}")
    logger.info(f"Parameters: n_splits={n_splits}, n_iter={n_iter}")

    try:
        # 1. Load data
        logger.info("1. Loading data...")
        X, y = load_experiment_data(exp_id)
        logger.info(f"   Features shape: {X.shape}")
        logger.info(
            f"   Target distribution: {y['target'].value_counts().to_dict()}")

        # 2. Validate data quality
        logger.info("2. Validating data quality...")
        if not validate_data_quality(X, y):
            logger.error("Data quality validation failed!")
            return False
        logger.info("   ✅ Data quality validation passed!")

        # 3. Split data
        logger.info("3. Splitting data...")
        X_train, X_test, y_train, y_test = split_data_temporal(X, y)

        # 4. Print CV splits info
        logger.info("4. Cross-validation splits info:")
        print_cv_splits_info(X_train, y_train, n_splits)

        # 5. Create model and tune hyperparameters
        logger.info("5. Starting hyperparameter tuning...")
        model = XGBoostGeminiModel(f"xgboost_{exp_id}")

        # Update n_iter for this specific tuning
        from config.model_params import BAYESIAN_OPTIMIZATION
        original_n_iter = BAYESIAN_OPTIMIZATION['n_iter']
        BAYESIAN_OPTIMIZATION['n_iter'] = n_iter

        try:
            best_params = model.tune_hyperparameters(X_train, y_train,
                                                     n_splits)
            logger.info(f"   ✅ Tuning completed!")
            logger.info(f"   Best parameters: {best_params}")

            # 6. Save tuning results
            logger.info("6. Saving tuning results...")
            tuning_results = {
                'exp_id': exp_id,
                'best_params': best_params,
                'cv_results': model.cv_results_,
                'n_splits': n_splits,
                'n_iter': n_iter,
                'data_shape': X.shape,
                'train_shape': X_train.shape,
                'test_shape': X_test.shape
            }

            results_file = LOGS_DIR / f"xgboost_tuning_results_{exp_id}.pkl"
            joblib.dump(tuning_results, results_file)
            logger.info(f"   Results saved to: {results_file}")

            # 7. Quick evaluation with best parameters
            logger.info("7. Quick evaluation with best parameters...")
            model.train(X_train, y_train)

            # Evaluate on training set
            train_metrics = model.evaluate(X_train, y_train)
            logger.info(f"   Training metrics: {train_metrics}")

            # Evaluate on test set
            test_metrics = model.evaluate(X_test, y_test)
            logger.info(f"   Test metrics: {test_metrics}")

            # 8. Save model
            logger.info("8. Saving tuned model...")
            model_path = model.save_model(exp_id, "l0")
            logger.info(f"   Model saved to: {model_path}")

            logger.info("🎉 Hyperparameter tuning completed successfully!")
            return True

        finally:
            # Restore original n_iter
            BAYESIAN_OPTIMIZATION['n_iter'] = original_n_iter

    except Exception as e:
        logger.error(f"❌ Hyperparameter tuning failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='XGBoost Hyperparameter Tuning')
    parser.add_argument('--exp_id',
                        type=str,
                        required=True,
                        help='Experiment ID (A0, A1, A2, A3, A4_Pruned)')
    parser.add_argument('--n_splits',
                        type=int,
                        default=5,
                        help='Number of TimeSeriesSplit folds (default: 5)')
    parser.add_argument(
        '--n_iter',
        type=int,
        default=50,
        help='Number of Bayesian optimization iterations (default: 50)')

    args = parser.parse_args()

    success = tune_hyperparameters(exp_id=args.exp_id,
                                   n_splits=args.n_splits,
                                   n_iter=args.n_iter)

    if success:
        print(
            f"\n✅ XGBoost hyperparameter tuning for {args.exp_id} completed successfully!"
        )
        sys.exit(0)
    else:
        print(f"\n❌ XGBoost hyperparameter tuning for {args.exp_id} failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
