네, 지금까지 논의한 BTC 'Sell' 신호 예측 모델 개발 전략을 Markdown 형식으로 정리해 드릴게요.

-----

# BTC 'Sell' 예측 앙상블 모델 개발 전략

## 1\. 문제 정의: 리스크 관리 집중

  * **목표:** 30일 뒤 가격이 -10% 이하로 하락하는 'Sell' 신호를 정확히 예측한다.
  * **유형:** 2진 분류 (Binary Classification)
      * **Target 1 (Positive):** `Sell` (30일 뒤 -10% 이하 하락)
      * **Target 0 (Negative):** `Buy` + `Wait` (그 외 모든 경우)
  * **이유:** 'Wait'만 예측하는 모델을 피하고, 리스크 관리에 가장 중요한 'Sell' 신호 탐지에 집중한다.

-----

## 2\. 핵심 과제: 데이터 불균형 (Class Imbalance)

  * **문제:** 'Sell' (Target 1) 데이터가 'Rest' (Target 0)에 비해 현저히 적다.
  * **해결책:** **샘플링 (X) ➡️ 클래스 가중치 (O)**
      * 데이터를 임의로 삭제(언더샘플링)하거나 생성(오버샘플링, SMOTE)하는 방식은 시계열 데이터의 중요 정보(패턴)를 왜곡/손실시킬 수 있어 **사용하지 않는다.**
      * 대신, 모델이 학습 시 소수 클래스('Sell')의 오류에 더 큰 페널티를 부여하도록 **`class_weight`** 파라미터를 사용한다.
          * **XGBoost:** `scale_pos_weight`
          * **Random Forest:** `class_weight='balanced'`
          * **Logistic Regression:** `class_weight='balanced'`

-----

## 3\. 모델 아키텍처: 스태킹 앙상블 (Stacking Ensemble)

"다르게 틀리는" 모델들을 조합하여 예측 성능을 극대화한다.

### Level 0: 기본 모델 (Base Models)

  * **Model 1: XGBoost** (트리, 부스팅 계열)
      * **역할:** 복잡하고 비선형적인 패턴 탐지.
  * **Model 2: Random Forest** (트리, 배깅 계열)
      * **역할:** XGBoost와 다른 방식(독립적 트리)으로 비선형 패턴 탐지, 과적합 방지에 유리.
  * **Model 3: Logistic Regression** (선형 계열)
      * **역할:** "지표가 오르면 Sell 확률이 선형적으로 증가"와 같은 단순한 선형 관계를 탐지하여 트리 모델들을 보완.

### Level 1: 메타 모델 (Meta-Model)

  * **Model: Logistic Regression**
      * **입력 (Meta-Features):** Level 0 모델들이 예측한 'Sell' 확률값 (예: [XGB\_prob, RF\_prob, LR\_prob])
      * **역할:** Level 0 모델들의 예측값을 보고, 각 모델의 신뢰도를 학습하여 "최적의 가중 평균"으로 최종 'Sell' 여부를 결정한다.
      * **이유:** 단순하고 빨라 과적합 위험이 낮다.

-----

## 4\. 데이터 분할: 시계열 원칙 준수

시계열 데이터는 절대 섞으면 안 되며(No Shuffle), 항상 과거로 미래를 예측해야 한다.

  * **Final Test Set (최종 시험지):** `2024.04.20 ~ 현재`
      * 모델 튜닝, 학습 과정에서 **절대 사용하지 않는다.**
      * 모든 개발이 완료된 최종 모델의 성능을 **단 한 번** 평가하는 용도.
  * **Train / Validation Set (학습/검증용):** `2020.05.12 ~ 2024.04.20`
      * 모든 모델 학습, 하이퍼파라미터 튜닝, 스태킹 학습에 이 데이터만 사용한다.

-----

## 5\. 핵심 검증 전략: `TimeSeriesSplit` (확장 윈도우)

`Train / Validation Set (2020-2024)`를 고정된 비율로 나누지 않고, `TimeSeriesSplit`을 사용해 교차 검증 및 메타 피처를 생성한다.

### 목적 1: 메타 모델(Level 1) 학습 데이터 생성

`TimeSeriesSplit(n_splits=5)`을 예시로 한 스태킹 학습 과정:

```python
# (의사 코드)
# X_full, y_full = 2020-2024년치 데이터

tscv = TimeSeriesSplit(n_splits=5)
meta_features_list = []
meta_targets_list = []

# 1. for 루프가 5번 돌면서 (Train, Val) 조합을 만듦
for train_index, val_index in tscv.split(X_full):
    
    # 2. 현재 Fold의 (Train, Val) 데이터 분리
    # FOLD 1: Train (2020-2021), Val (2021)
    # FOLD 2: Train (2020-2022), Val (2022) ...
    X_train, X_val = X_full[train_index], X_full[val_index]
    y_train, y_val = y_full[train_index], y_full[val_index]

    # 3. Level 0 모델들을 'X_train'으로 학습 (class_weight 적용)
    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # 4. 학습된 모델로 'X_val'의 확률 예측 (데이터 누수 방지)
    xgb_pred = xgb.predict_proba(X_val)[:, 1]
    rf_pred = rf.predict_proba(X_val)[:, 1]
    lr_pred = lr.predict_proba(X_val)[:, 1]
    
    # 5. 예측값(메타 피처)과 실제 정답(메타 타겟)을 저장
    meta_features_list.append(np.column_stack([xgb_pred, rf_pred, lr_pred]))
    meta_targets_list.append(y_val)

# 6. 모든 Fold에서 모인 예측값과 정답을 하나로 합침
X_meta_train = np.concatenate(meta_features_list)
y_meta_train = np.concatenate(meta_targets_list)

# 7. 이 데이터로 메타 모델(Level 1)을 학습
meta_model.fit(X_meta_train, y_meta_train)
```

### 목적 2: 하이퍼파라미터 튜닝

  * 위 `for` 루프의 4번 단계에서 `X_val` 예측 성능(예: F1-Score)을 각 Fold마다 측정한다.
  * 5개 Fold의 **평균 성능**이 가장 좋은 하이퍼파라미터 조합(예: `max_depth` 등)을 찾는다.

-----

## 6\. 최종 모델 학습 및 평가

1.  **최종 학습:**
      * `TimeSeriesSplit`으로 찾은 최적의 하이퍼파라미터를 적용한다.
      * 위 스태킹 프로세스(1\~7단계)를 **`Train / Validation Set (2020-2024)` 전체**에 대해 수행하여 `meta_model`까지 학습시킨다.
      * Level 0 모델들은 \*\*`Train / Validation Set` 전체(2020-2024)\*\*로 다시 학습시킨다. (`fit(X_full, y_full)`)
2.  **최종 평가:**
      * 이렇게 완성된 "스태킹 파이프라인"을, 절대 사용한 적 없는 \*\*`Final Test Set (2024-현재)`\*\*에 적용하여 최종 성능을 평가한다.