#!/usr/bin/env python3
"""
XGBoost Complete Experiment Pipeline
"""

import sys
import argparse
import logging
from pathlib import Path
import subprocess
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.settings import LOGS_DIR


def setup_logging(exp_id: str):
    """Setup logging for experiment pipeline"""
    log_dir = LOGS_DIR / "experiments"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"xgboost_experiment_{exp_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file),
                  logging.StreamHandler()])

    return logging.getLogger(f"xgboost_experiment_{exp_id}")


def run_command(command: list, logger: logging.Logger, step_name: str):
    """Run a command and log the results"""
    logger.info(f"Running: {' '.join(command)}")

    start_time = time.time()
    try:
        result = subprocess.run(command,
                                capture_output=True,
                                text=True,
                                check=True,
                                cwd=project_root)

        duration = time.time() - start_time
        logger.info(
            f"✅ {step_name} completed successfully in {duration:.2f} seconds")

        if result.stdout:
            logger.info(f"STDOUT: {result.stdout}")

        return True

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        logger.error(f"❌ {step_name} failed after {duration:.2f} seconds")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"STDERR: {e.stderr}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        return False


def run_experiment(exp_id: str,
                   n_splits: int = 5,
                   n_iter: int = 50,
                   skip_tuning: bool = False):
    """
    Run complete XGBoost experiment pipeline
    
    Args:
        exp_id: Experiment ID (A0, A1, A2, A3, A4_Pruned)
        n_splits: Number of TimeSeriesSplit folds
        n_iter: Number of Bayesian optimization iterations
        skip_tuning: Skip hyperparameter tuning step
    """
    logger = setup_logging(exp_id)

    logger.info(f"🚀 Starting complete XGBoost experiment for {exp_id}")
    logger.info(
        f"Parameters: n_splits={n_splits}, n_iter={n_iter}, skip_tuning={skip_tuning}"
    )

    start_time = time.time()

    try:
        # Step 1: Hyperparameter Tuning
        if not skip_tuning:
            logger.info("=" * 60)
            logger.info("STEP 1: Hyperparameter Tuning")
            logger.info("=" * 60)

            tuning_command = [
                "python", "training/xgboost/tune_hyperparameters.py",
                "--exp_id", exp_id, "--n_splits",
                str(n_splits), "--n_iter",
                str(n_iter)
            ]

            if not run_command(tuning_command, logger,
                               "Hyperparameter tuning"):
                logger.error("❌ Hyperparameter tuning failed!")
                return False
        else:
            logger.info(
                "⏭️  Skipping hyperparameter tuning (using existing results)")

        # Step 2: Training and Evaluation
        logger.info("=" * 60)
        logger.info("STEP 2: Training and Evaluation")
        logger.info("=" * 60)

        training_command = [
            "python", "training/xgboost/train_and_evaluate.py", "--exp_id",
            exp_id, "--use_tuned_params"
        ]

        if not run_command(training_command, logger,
                           "Training and evaluation"):
            logger.error("❌ Training and evaluation failed!")
            return False

        # Step 3: Results Summary
        logger.info("=" * 60)
        logger.info("STEP 3: Results Summary")
        logger.info("=" * 60)

        total_duration = time.time() - start_time
        logger.info(
            f"🎉 Complete experiment for {exp_id} finished successfully!")
        logger.info(
            f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)"
        )

        # Print results file locations
        logger.info("Results saved to:")
        logger.info(
            f"  - Tuning results: logs/xgboost_tuning_results_{exp_id}.pkl")
        logger.info(
            f"  - Evaluation results: logs/xgboost_evaluation_results_{exp_id}.pkl"
        )
        logger.info(f"  - Model: logs/models/{exp_id}/l0/xgboost_model.pkl")
        logger.info(
            f"  - Logs: logs/experiments/xgboost_experiment_{exp_id}.log")

        return True

    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(
            f"❌ Experiment failed after {total_duration:.2f} seconds: {str(e)}"
        )
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='XGBoost Complete Experiment Pipeline')
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
    parser.add_argument('--skip_tuning',
                        action='store_true',
                        help='Skip hyperparameter tuning step')
    parser.add_argument('--quick',
                        action='store_true',
                        help='Quick test with reduced iterations (n_iter=10)')

    args = parser.parse_args()

    # Handle quick mode
    if args.quick:
        args.n_iter = 10
        print("🚀 Quick mode: Using n_iter=10 for faster testing")

    success = run_experiment(exp_id=args.exp_id,
                             n_splits=args.n_splits,
                             n_iter=args.n_iter,
                             skip_tuning=args.skip_tuning)

    if success:
        print(
            f"\n✅ Complete XGBoost experiment for {args.exp_id} completed successfully!"
        )
        sys.exit(0)
    else:
        print(f"\n❌ Complete XGBoost experiment for {args.exp_id} failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
