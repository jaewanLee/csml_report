#!/usr/bin/env python3
"""
XGBoost Training and Evaluation Script
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
from config.settings import LOGS_DIR, MODELS_DIR
import joblib
import pandas as pd
import numpy as np


def setup_logging(exp_id: str):
    """Setup logging for training and evaluation"""
    log_dir = LOGS_DIR / "training"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"xgboost_training_{exp_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file),
                  logging.StreamHandler()])

    return logging.getLogger(f"xgboost_training_{exp_id}")


def load_tuning_results(exp_id: str):
    """Load hyperparameter tuning results"""
    results_file = LOGS_DIR / f"xgboost_tuning_results_{exp_id}.pkl"

    if not results_file.exists():
        raise FileNotFoundError(
            f"Tuning results not found: {results_file}. Run tune_hyperparameters.py first."
        )

    return joblib.load(results_file)


def train_and_evaluate(exp_id: str, use_tuned_params: bool = True):
    """
    Train XGBoost model and evaluate performance
    
    Args:
        exp_id: Experiment ID (A0, A1, A2, A3, A4_Pruned)
        use_tuned_params: Whether to use tuned parameters or default ones
    """
    logger = setup_logging(exp_id)

    logger.info(f"🚀 Starting XGBoost training and evaluation for {exp_id}")
    logger.info(f"Use tuned parameters: {use_tuned_params}")

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

        # 4. Load tuning results if requested
        if use_tuned_params:
            logger.info("4. Loading tuning results...")
            tuning_results = load_tuning_results(exp_id)
            best_params = tuning_results['best_params']
            logger.info(f"   Best parameters: {best_params}")
        else:
            logger.info("4. Using default parameters...")
            best_params = None

        # 5. Create and train model
        logger.info("5. Training model...")
        model = XGBoostGeminiModel(f"xgboost_{exp_id}")

        if best_params:
            model.best_params = best_params
            model.train(X_train, y_train)
        else:
            # Use default parameters
            model.train(X_train, y_train)

        logger.info("   ✅ Model training completed!")

        # 6. Evaluate on training set
        logger.info("6. Evaluating on training set...")
        train_metrics = model.evaluate(X_train, y_train)
        logger.info(f"   Training metrics: {train_metrics}")

        # 7. Evaluate on test set
        logger.info("7. Evaluating on test set...")
        test_metrics = model.evaluate(X_test, y_test)
        logger.info(f"   Test metrics: {test_metrics}")

        # 8. Generate meta-features for stacking
        logger.info("8. Generating meta-features for stacking...")
        meta_features = model.generate_meta_features(X_train,
                                                     y_train,
                                                     n_splits=5)
        logger.info(f"   Meta-features shape: {meta_features.shape}")

        # 9. Save model and results
        logger.info("9. Saving model and results...")
        model_path = model.save_model(exp_id, "l0")
        logger.info(f"   Model saved to: {model_path}")

        # 10. Save evaluation results
        evaluation_results = {
            'exp_id': exp_id,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'meta_features_shape': meta_features.shape,
            'data_shape': X.shape,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'best_params': best_params
        }

        results_file = LOGS_DIR / f"xgboost_evaluation_results_{exp_id}.pkl"
        joblib.dump(evaluation_results, results_file)
        logger.info(f"   Evaluation results saved to: {results_file}")

        # 11. Print summary
        logger.info("10. Results Summary:")
        logger.info("=" * 50)
        logger.info(f"Experiment ID: {exp_id}")
        logger.info(f"Features: {X.shape[1]}")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Test samples: {X_test.shape[0]}")
        logger.info("")
        logger.info("Training Metrics:")
        for metric, value in train_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        logger.info("")
        logger.info("Test Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        logger.info("=" * 50)

        logger.info("🎉 Training and evaluation completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Training and evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='XGBoost Training and Evaluation')
    parser.add_argument('--exp_id',
                        type=str,
                        required=True,
                        help='Experiment ID (A0, A1, A2, A3, A4_Pruned)')
    parser.add_argument('--use_tuned_params',
                        action='store_true',
                        default=True,
                        help='Use tuned parameters (default: True)')
    parser.add_argument('--use_default_params',
                        action='store_true',
                        help='Use default parameters instead of tuned ones')

    args = parser.parse_args()

    # Handle conflicting arguments
    if args.use_default_params:
        args.use_tuned_params = False

    success = train_and_evaluate(exp_id=args.exp_id,
                                 use_tuned_params=args.use_tuned_params)

    if success:
        print(
            f"\n✅ XGBoost training and evaluation for {args.exp_id} completed successfully!"
        )
        sys.exit(0)
    else:
        print(f"\n❌ XGBoost training and evaluation for {args.exp_id} failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
