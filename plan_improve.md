
# BTC 'Sell' 신호 예측을 위한 Ablation Study 계획

## 🎯 프로젝트 개요 (Project Overview)

**목표:** 체계적인 'Ablation Study' (절제 연구)를 수행하여, BTC 'Sell' 신호 예측을 위한 최적의 다중 타임프레임(MTF) 및 시차(Lag) 조합을 식별한다.

**연구 질문 (RQs):**
* **RQ1 (현재 시점):** H4 $\rightarrow$ D1 $\rightarrow$ W1 $\rightarrow$ M1로 확장 시, '현재 시점(t)' 정보만을 사용한 모델의 최적 타임프레임 조합은 무엇이며, 각 타임프레임 추가는 성능 향상에 기여하는가, 아니면 노이즈를 유발하는가?
* **RQ2 (과거 시점):** RQ1의 최적 조합에 체계적인 '과거 시차 특성(Lags)'을 추가했을 때, 예측 성능이 유의미하게 향상되는가?

**아키텍처:** 각 실험(`A0`~`A4_Pruned`)은 공정한 비교를 위해 동일한 스태킹 앙상블(XGBoost + RF + LR) + 메타 모델 아키텍처를 사용하여 평가된다.

---

## 📋 실행 단계 (Implementation Steps)

### **Step 1: 데이터 수집 (Data Collection)**
**기간:** 1-2일
**목표:** 모든 필수 타임프레임의 BTC OHLCV 원본 데이터 수집

#### Tasks:
- [ ] **1.1** ccxt 라이브러리 설정
- [ ] **1.2** 데이터 수집 폴더 구조 생성 (`data_collection/`)
- [ ] **1.3** 다중 타임프레임 OHLCV 데이터 수집: **H4, D1, W1, M1**
- [ ] **1.4** 데이터 기간: 2020-05-12 ~ 2024-04-20 (학습/검증) + 2024-04-20 ~ 현재 (최종 테스트)
- [ ] **1.5** 데이터 품질 검증 및 Parquet 형식으로 저장

#### Deliverables:
- 모든 타임프레임(H4, D1, W1, M1)의 원본 OHLCV 데이터셋 (Parquet)

---

### **Step 2: 피처 엔지니어링 및 실험 세트 구성**
**기간:** 3-4일
**목표:** 모든 기술 지표를 계산하고, Ablation Study를 위한 5개의 독립적인 피처 세트(`A0`~`A4`)를 생성함

#### Tasks:
- [ ] **2.1** 데이터 탐색 (임계값 -10% vs -15%, Fold 수 분석 등)
- [ ] **2.2** 모든 타임프레임에 대한 기술 지표 계산 (RSI, MACD, MA, Ichimoku 등)
- [ ] **2.3** **Ablation Study 피처 세트 생성 (RQ1용):**
    - **A0 (Baseline):** H4 지표 (현재 시점 `t`만)
    - **A1 (MTF-1):** `A0` + D1 지표 (현재 시점 `t`만)
    - **A2 (MTF-2):** `A1` + W1 지표 (현재 시점 `t`만)
    - **A3 (MTF-3):** `A2` + M1 지표 (현재 시점 `t`만)
- [ ] **2.4** **시차(Lag) 특성 생성 (RQ2용):**
    - H4 Lags (t-1 ~ t-6)
    - D1 Lags (t-1 ~ t-7)
    - W1 Lags (t-1 ~ t-4)
    - M1 Lags (t-1 ~ t-2)
- [ ] **2.5** **A4 피처 세트 생성:**
    - **A4 (Historical Lags):** `A3` + `2.4`에서 생성한 모든 시차 특성 (수백 개의 피처)
- [ ] **2.6** 타겟 변수 생성 (Binary: Sell vs Rest) 및 별도 저장
- [ ] **2.7** 데이터 분할 (시간순):
    - Train/Validation: 2020-2024
    - Final Test: 2024-현재

#### Deliverables:
- 5개의 분리된 피처 세트 파일 (A0.parquet, A1.parquet, A2.parquet, A3.parquet, A4.parquet)
- 타겟 변수 파일 (y.parquet)

---

### **Step 3: 아키텍처 및 환경 설정**
**기간:** 2-3일
**목표:** 실험 루프를 효율적으로 실행하기 위한 모듈형 코드베이스 및 환경 구축

#### Tasks:
- [ ] **3.1** Conda 환경(`btc_model`) 및 `requirements.txt` 완성 (Python 3.13 기반)
- [ ] **3.2** Git 저장소 및 프로젝트 폴더 구조 확립 (모듈형 설계)
    - `config/`, `data_processing/`, `training/`, `evaluation/`, `models/level0/`, `models/level1/` 등
- [ ] **3.3** 모듈형 모델 클래스 구현 (XGB, RF, LR, MetaLR)
- [ ] **3.4** 스태킹 앙상블 래퍼(`StackingEnsemble`) 구현
- [ ] **3.5** TimeSeriesSplit 교차검증 프레임워크 유틸리티(`utils/`) 구현
- [ ] **3.6** 로깅 및 모델 저장/로드 유틸리티 구현
- [ ] **3.7** **실험 스크립트 작성 (`training/03_run_experiment.py`):**
    - `exp_id` (예: "A1")를 인자로 받아, 해당 피처 세트로 전체 훈련/평가/기록(L0 튜닝 ~ L1 평가)을 수행하는 단일 스크립트 구현

#### Deliverables:
- 모듈형 코드베이스 (`.py` 파일들)
- 모든 실험에서 공통으로 사용할 `03_run_experiment.py` 스크립트

---

### **Step 4: Ablation Study 실험 루프 실행**
**기간:** 4-5일
**목표:** `A0`~`A3`를 먼저 실행하여 파이프라인을 검증하고 RQ1의 답을 찾은 뒤, `A4` 작업을 수행하여 RQ2의 답을 찾는다.

#### Tasks:
- [ ] **4.1** **실험 결과 로거(Logger) 설정:**
    - `experiment_results.csv` 파일 생성 (컬럼: `Experiment_ID`, `Num_Features`, `Final_Test_F1`, `Final_Test_Precision`, `Final_Test_Recall` 등)
- [ ] **4.2** **메인 실험 루프 - Part 1 (RQ1 검증):**
    - **`for exp_id in [A0, A1, A2, A3]:`**
        - `python training/03_run_experiment.py --exp_id ${exp_id}` 실행
        - (내부 동작: `exp_id` 피처 로드 $\rightarrow$ L0 튜닝 $\rightarrow$ Meta-Feature 생성 $\rightarrow$ L1 튜닝 $\rightarrow$ 최종 모델 훈련 $\rightarrow$ 최종 평가 $\rightarrow$ `experiment_results.csv`에 1행 기록)
- [ ] **4.3** **중간 분석 및 검증 (Checkpoint):**
    - `experiment_results.csv`에 기록된 **`A0`~`A3`의 결과**를 분석
    - **(검증):** F1 점수가 비정상적이지 않은지, 파이프라인이 정상 작동하는지 확인
    - **(결정):** 파이프라인이 정상 작동함을 확인한 후에만 `4.4` 단계로 진행 (문제 시 `Step 2` 또는 `Step 3` 복귀)
- [ ] **4.4** **A4 피처 가지치기(Pruning) 작업:**
    - **4.4.1 (Load A4):** `A4.parquet` (모든 피처) 데이터 로드
    - **4.4.2 (XGBoost Tuning on A4):** `A4` 데이터로 **XGBoost 모델만** L0 튜닝 프레임워크(Optuna) 실행
    - **4.4.3 (Analyze Importance):** 튜닝된 XGBoost의 `feature_importances_` 분석
    - **4.4.4 (Prune Features):** 중요도가 0 또는 매우 낮은 피처를 제거하고, 살아남은 "알짜 피처" 목록(`A4_pruned_features`) 확정
    - **4.4.5 (Create Pruned Dataset):** `A4_pruned_features`만으로 구성된 새로운 피처 세트(`A4_Pruned.parquet`)를 디스크에 저장
- [ ] **4.5** **메인 실험 루프 - Part 2 (RQ2 검증):**
    - `exp_id = "A4_Pruned"`로 설정
    - `python training/03_run_experiment.py --exp_id A4_Pruned` 실행
    - (`experiment_results.csv`에 `A4_Pruned` 행이 추가됨)

#### Deliverables:
- **`experiment_results.csv`**: 모든 실험(A0, A1, A2, A3, A4_Pruned)의 성능이 요약된 핵심 결과 테이블
- 각 실험별로 저장된 모델 아티팩트 (`.pkl`) 및 상세 로그
- `A4_Pruned.parquet` (최종 "알짜 피처" 데이터셋)

---

### **Step 5: 결과 분석 및 연구 질문(RQ) 답변**
**기간:** 1-2일
**목표:** `experiment_results.csv`를 분석하여 님의 연구 질문에 대한 명확한 답을 도출함

#### Tasks:
- [ ] **5.1** `experiment_results.csv` 로드 및 시각화 (예: `A0`~`A4_Pruned` F1 점수 막대그래프)
- [ ] **5.2** **RQ1 답변 (현재 시점):**
    - `A0`, `A1`, `A2`, `A3`의 `Final_Test_F1` 점수를 비교 분석
    - "D1 추가는 성능을 X% 향상시켰으나, W1과 M1 추가는 유의미한 변화를 주지 못했다" 등 결론 도출
- [ ] **5.3** **RQ2 답변 (과거 시점):**
    - `5.2`에서 찾은 최적 조합(예: `A3`)의 `Final_Test_F1`과 `A4_Pruned`의 `Final_Test_F1`을 비교
    - "체계적인 과거 시차 특성을 추가하자 F1 점수가 X에서 Y로 크게 향상되었다" 등 결론 도출
- [ ] **5.4** 피처 중요도 분석:
    - `A4_Pruned` 실험의 `feature_importances_`를 분석하여 "Sell" 신호에 가장 중요했던 상위 20개 피처(타임프레임, 시차) 식별

#### Deliverables:
- **RQ1, RQ2에 대한 명확한 답변이 포함된 분석 보고서**
- `A0`~`A4_Pruned` 성능 비교 차트
- 최종 모델(`A4_Pruned`)의 피처 중요도 플롯

---

### **Step 6: 결론 및 최종 문서화**
**기간:** 1-2일
**목표:** 전체 연구 결과를 요약하고 최종 보고서 작성

#### Tasks:
- [ ] **6.1** 최종 연구 결과(5단계) 요약
- [ ] **6.2** 한계점 및 향후 연구 방향 제시 (예: 롤링 윈도우 검증, LSTM 앙상블 추가 등)
- [ ] **6.3** 프로젝트 코드 및 문서를 GitHub에 정리

#### Deliverables:
- 최종 연구 보고서 (PDF 제안서의 완성본)
- 정리된 GitHub 저장소

---

## 🛠️ 기술 스택 (Technical Stack)

-   **Data & ML:** `ccxt`, `pandas`, `numpy`, `scikit-learn`, `xgboost`
-   **Tuning:** **`scikit-optimize`** (Bayesian Optimization용)
-   **Interpretability:** `shap`
-   **Development:** **`python 3.13`** (3.14 호환성 이슈 회피), `jupyter`, `black`

---

## 📊 성공 지표 (Success Metrics)

-   **Model Performance:**
    -   **최종 모델(`A4_Pruned`)**의 `Final_Test_F1` $\ge$ 0.70
-   **Research:**
    -   `A0`~`A4_Pruned` 간의 성능 차이가 통계적으로 유의미하게 관찰됨
    -   RQ1, RQ2에 대한 명확하고 데이터에 기반한 결론 도출

---

## 🚨 리스크 및 대응 (Risk Mitigation)

-   **Data Risks:** 누락 데이터 처리 전략, 데이터 품질 검증, 시계열 일관성 검증
-   **Model Risks:** `TimeSeriesSplit`을 통한 과적합 방지, `class_weight`를 통한 불균형 처리
-   **Research Risks:** 만약 `A0`~`A3` 성능이 모두 낮다면, `Step 4.3`에서 중단하고 타겟 변수 정의(`Step 2.6`) 또는 기본 지표(`Step 2.2`) 재검토

---

## 🔧 핵심 구현 노트 (Critical Implementation Notes)

-   **튜닝 전략 (Hybrid):**
    -   **L0 (XGB, RF):** `Bayesian Optimization (scikit-optimize)` 사용 (넓은 탐색 공간)
    -   **L1 (LR):** `GridSearchCV` 사용 (단순한 탐색 공간)
-   **튜닝 워크플로우:**
    -   **Step 1:** `GridSearchCV`/`BayesSearchCV` + `cv=TimeSeriesSplit`으로 `best_params_` (설정값)만 찾는다.
    -   **Step 2:** 찾은 `best_params_`를 사용하여 **새 모델**을 만들고, `TimeSeriesSplit` 루프를 돌며 `Meta-Features`를 생성한다.
    -   **Step 3:** 찾은 `best_params_`를 사용하여 **새 모델**을 만들고, **전체 Train/Val 데이터**로 `Final Model`을 학습시킨다.
-   **피처 가지치기 (Pruning):**
    -   `A4` 데이터셋은 L0 앙상블 훈련에 직접 사용되지 않음.
    -   오직 `A4_Pruned` 데이터셋을 생성하기 위한 "재료"로만 사용됨.

---

## 🤔 추가 논의 사항 (Need to Discuss Further)

-   **롤링 윈도우 (Rolling Window) 검증:**
    -   현재 계획은 `TimeSeriesSplit` (확장 윈도우)을 사용함.
    -   만약 `A4_Pruned` 모델이 `Final Test Set` (2024년)에서 성능이 급격히 하락한다면(과거에 과적합), 모델 검증 방식을 "롤링 윈도우"로 변경하여 "최신 시장 상황 적응력"을 테스트하는 2차 실험을 고려할 수 있음.