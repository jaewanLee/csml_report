from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 콘솔 출력
        logging.FileHandler('rolling_window_experiments.log')  # 파일 출력
    ])

logger = logging.getLogger(__name__)


def create_rolling_window_target(window: int, threshold: float,
                                 train_start: str,
                                 test_end: str) -> pd.DataFrame:
    """Create rolling window target variable."""
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / "data_collection" / "data" / "btc_4h_20251029.parquet"
    df_raw = pd.read_parquet(file_path)

    y_full = pd.DataFrame(index=df_raw.index)
    y_full['target'] = 0

    total_len = len(df_raw)
    for i in range(total_len):
        if i + window >= total_len:
            break
        current_close = df_raw.iloc[i]['close']
        future_data = df_raw.iloc[i + 1:i + 1 + window]
        future_min = future_data['low'].min()
        if future_min <= current_close * (1 - threshold):
            y_full.iloc[i, 0] = 1

    y_full = y_full[(y_full.index >= train_start) & (y_full.index <= test_end)]
    return y_full


def experiment_rolling_window_targets(
        train_start: str = '2014-02-01',
        test_end: str = '2025-09-19') -> List[Dict]:
    """실험용 함수 - 다양한 window/threshold 조합 테스트"""
    logger.info("🚀 Starting rolling window target experiments...")

    windows = range(30, 60)  # 40~60
    thresholds = [i / 100 for i in range(8, 12)]  #(10% ~ 15%)

    results = []
    total_combinations = len(windows) * len(thresholds)
    current = 0

    for window in windows:
        for threshold in thresholds:
            current += 1
            logger.info(
                f"📊 Testing window={window}, threshold={threshold:.2f} ({current}/{total_combinations})"
            )

            try:
                y_target = create_rolling_window_target(
                    window, threshold, train_start, test_end)

                # 통계 계산
                target_counts = y_target['target'].value_counts()
                total_samples = len(y_target)
                positive_samples = target_counts.get(1, 0)
                positive_ratio = positive_samples / total_samples if total_samples > 0 else 0

                # Fold별 비율 계산
                from data_processing.validation.target_validation import validate_target_distribution_with_folds
                fold_stats = validate_target_distribution_with_folds(
                    y_target,
                    n_folds=5,
                    name=f"Window{window}_Thresh{threshold:.2f}")

                result = {
                    'window':
                    window,
                    'threshold':
                    threshold,
                    'total_samples':
                    total_samples,
                    'positive_samples':
                    positive_samples,
                    'positive_ratio':
                    positive_ratio,
                    'negative_samples':
                    target_counts.get(0, 0),
                    'negative_ratio':
                    1 - positive_ratio,
                    'fold_positive_ratios':
                    [stat['positive_ratio'] for stat in fold_stats['folds']],
                    'fold_balance_score':
                    fold_stats['balance_metrics']['ratio_std'],
                    'fold_std':
                    fold_stats['balance_metrics']['ratio_std']
                }

                results.append(result)
                fold_ratios = [
                    stat['positive_ratio'] for stat in fold_stats['folds']
                ]
                logger.info(
                    f"   ✅ Results: {positive_samples}/{total_samples} ({positive_ratio:.3f}) | Fold balance: {fold_stats['balance_metrics']['ratio_std']:.3f} | Fold ratios: {[f'{r:.3f}' for r in fold_ratios]}"
                )

            except Exception as e:
                logger.error(f"   ❌ Error: {e}")
                logger.error(f"   ❌ Full error: {type(e).__name__}: {str(e)}")
                results.append({
                    'window': window,
                    'threshold': threshold,
                    'error': str(e)
                })

    logger.info(f"🎯 Completed {len(results)} experiments")
    return results


def analyze_experiment_results(results: List[Dict]) -> None:
    """실험 결과 분석 및 최적 조합 찾기"""
    logger.info("📈 Analyzing experiment results...")

    # 에러가 없는 결과만 필터링
    valid_results = [r for r in results if 'error' not in r]

    if not valid_results:
        logger.error("❌ No valid results found")
        return

    # DataFrame으로 변환
    df_results = pd.DataFrame(valid_results)

    # 통계 출력
    logger.info(f"📊 Total valid experiments: {len(df_results)}")
    logger.info(
        f"📊 Positive ratio range: {df_results['positive_ratio'].min():.3f} ~ {df_results['positive_ratio'].max():.3f}"
    )
    logger.info(
        f"📊 Average positive ratio: {df_results['positive_ratio'].mean():.3f}")

    # 최적 조합 찾기 (positive_ratio가 0.1~0.3 범위 내에서)
    balanced_results = df_results[(df_results['positive_ratio'] >= 0.1)
                                  & (df_results['positive_ratio'] <= 0.3)]

    if len(balanced_results) > 0:
        # 가장 균형잡힌 조합 (0.2에 가까운 것)
        balanced_results['balance_score'] = abs(
            balanced_results['positive_ratio'] - 0.2)
        best_result = balanced_results.loc[
            balanced_results['balance_score'].idxmin()]

        logger.info("🎯 Best balanced combination:")
        logger.info(f"   Window: {best_result['window']}")
        logger.info(f"   Threshold: {best_result['threshold']:.2f}")
        logger.info(f"   Positive ratio: {best_result['positive_ratio']:.3f}")
        logger.info(f"   Total samples: {best_result['total_samples']}")
    else:
        logger.warning(
            "⚠️  No balanced results found (positive_ratio 0.1~0.3)")
        # 가장 가까운 것 찾기
        df_results['balance_score'] = abs(df_results['positive_ratio'] - 0.2)
        best_result = df_results.loc[df_results['balance_score'].idxmin()]

        logger.info("🎯 Closest to balanced combination:")
        logger.info(f"   Window: {best_result['window']}")
        logger.info(f"   Threshold: {best_result['threshold']:.2f}")
        logger.info(f"   Positive ratio: {best_result['positive_ratio']:.3f}")
        logger.info(f"   Total samples: {best_result['total_samples']}")

    # Fold 균형도 순으로 정렬 (낮을수록 좋음)
    logger.info(
        "📊 All experiment results (sorted by fold balance score - lower is better):"
    )
    df_results_sorted = df_results.sort_values('fold_balance_score')
    for i, (_, row) in enumerate(df_results_sorted.iterrows(), 1):
        fold_ratios_str = [f'{r:.3f}' for r in row['fold_positive_ratios']]
        logger.info(
            f"   {i:3d}. Window={int(row['window']):2d}, Threshold={row['threshold']:5.2f}, "
            f"Ratio={row['positive_ratio']:.3f}, Fold_balance={row['fold_balance_score']:.3f}, "
            f"Fold_ratios={fold_ratios_str}, Samples={int(row['total_samples'])}"
        )


#python -m data_processing.main_pipeline
if __name__ == '__main__':
    # 실험 실행
    # results = experiment_rolling_window_targets(train_start='2014-02-01',
    #                                             test_end='2025-09-19')

    # # 결과 분석
    # analyze_experiment_results(results)

    try:
        y_target = create_rolling_window_target(window=30,
                                                threshold=0.11,
                                                train_start='2014-02-01',
                                                test_end='2025-09-19')

        # 프로젝트 루트 기준으로 features 폴더 경로 설정
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / 'features'
        data_dir.mkdir(exist_ok=True)

        y_target.to_parquet(data_dir / 'y.parquet')
        logger.info(f"✅ Target saved to: {data_dir / 'y.parquet'}")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.error(f"❌ Full error: {type(e).__name__}: {str(e)}")
