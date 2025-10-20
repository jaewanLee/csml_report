[cite_start]`proposal_english.md` [cite: 1, 31]
[cite_start]`2025-10-06` [cite: 2, 32]

# [cite_start]Systematic Multi-Timeframe Ablation Study for Bitcoin Price Prediction [cite: 3]
[cite_start]**Student Name: Jeawn Lee** [cite: 4]

## [cite_start]1. Background & Motivation [cite: 5]
[cite_start]Bitcoin experiences repeated significant drawdowns due to extreme volatility and fat-tail distributions, degrading risk-adjusted performance while increasing the practical value of decline avoidance[cite: 6]. [cite_start]While recent research has demonstrated the effectiveness of multi-timeframe analysis in Bitcoin prediction, existing studies often focus on short-term timeframes (5min-4h) and assume that adding more timeframes continues to improve performance[cite: 7]. [cite_start]However, the fundamental question remains: does adding more timeframes actually continue to improve prediction performance, or does it introduce noise and overfitting? [cite: 8] [cite_start]Additionally, does utilizing historical information (24h) also contribute meaningfully to risk prediction? [cite: 9]

[cite_start]This study employs a controlled Ablation Study $(AO\rightarrow A1\rightarrow A2\rightarrow A3)$ to rigorously test whether each timeframe addition provides meaningful performance gains or introduces diminishing returns[cite: 10].

[cite_start]**Research Questions:** [cite: 11]
* [cite_start]RQ1: What is the optimal MTF combination for long-term risk prediction when expanding $H4\rightarrow D1\rightarrow W1$? [cite: 12]
* [cite_start]RQ2: Does utilizing historical information (24h) contribute meaningfully to risk prediction? [cite: 13]

## [cite_start]2. Methods & Hypotheses [cite: 14]

### 2.1. [cite_start]Methods [cite: 15]

[cite_start]**Features & Ablation Study:** [cite: 16]
* [cite_start]Core Technical Indicators: RSI, MACD, MA (7/14/20/60/120), Ichimoku, OHLCV derivatives [cite: 17]
* [cite_start]Multi-Timeframe Hierarchy: $W1\rightarrow D1\rightarrow H4$ [cite: 18]
* [cite_start]Ablation Study Design: [cite: 19]
    * [cite_start]AO (Baseline): H4 timeframe technical indicators only [cite: 20]
    * [cite_start]A1 (MTF-1): $A0+D1$ timeframe indicators [cite: 21]
    * [cite_start]A2 (MTF-2): $A1+W1$ timeframe indicators [cite: 22]
    * [cite_start]A3 (MTF-3): $A2+24h$ historical features [cite: 23]

[cite_start]**Model & Validation:** [cite: 24]
* [cite_start]Model: LightGBM multiclass (class_weight=1/prevalence) [cite: 25]
* [cite_start]Cross-Validation: Purged & Embargoed Time-Series CV ( $(H=30$, $E=30$) [cite: 26]
* [cite_start]Evaluation: Macro-F1 (primary), Balanced Accuracy, OvR ROC-AUC, Brier Score [cite: 27]

### 2.2. [cite_start]Hypotheses [cite: 28]

[cite_start]H1 (Incremental Contribution of MTF): Multi-timeframe addition will show either meaningful improvement $(AO<A1<A2)$ or diminishing returns $(AO\ge A1\ge A2)$, indicating whether additional timeframes provide value or introduce noise[cite: 29].

[cite_start]H2 (Historical Features Contribution): Historical features will either enhance performance ( $(A2<A3)$ or introduce noise $(A2\ge A3)$, testing the value of 24h lookback information[cite: 33].

[cite_start]H3 (Feature Sensitivity for Sell Signal): SHAP analysis will identify the most important features for 'sell' signal prediction, revealing which features are most critical for decline risk prediction[cite: 34].

## [cite_start]3. Materials & Data Sources [cite: 35]

### 3.1. [cite_start]Data Source and Period [cite: 36]
* [cite_start]Asset: BTC/USD spot price data [cite: 37]
* [cite_start]Period: 2020-05-12~2024-04-20 (post-3rd halving period) [cite: 38]

### 3.2. [cite_start]Technical Indicators & Feature Engineering [cite: 39]
[cite_start]Technical Indicators: RSI (14-period), MACD (12,26,9), Moving Averages (7,14,20,60,120), Ichimoku Cloud, OHLCV [cite: 40]
[cite_start]Multi-timeframe Feature Engineering: Follows the Ablation Study design $(AO\rightarrow A1\rightarrow A2\rightarrow A3)$ with hierarchical structure $(W1\rightarrow D1\rightarrow H4)$ and leakage prevention methods $(shift=0)$[cite: 41].

[cite_start]**Labeling Scheme:** [cite: 42]
* [cite_start]Prediction Horizon: 30-day [cite: 43]
* [cite_start]Thresholds: -15% (sell), +5% (buy), otherwise (wait) [cite: 44]

## [cite_start]4. Expected Results & Contribution [cite: 45]

[cite_start]**Expected Results:** [cite: 46]
* [cite_start]Scenario 1: Meaningful performance improvement with MTF information addition $(AO<A1<A2<A3)$ [cite: 47]
* [cite_start]Scenario 2: Diminishing returns or noise introduction $(A0\ge A1\ge A2\ge A3)$ [cite: 47]
* [cite_start]Scenario 3: Mixed results where some timeframes help while others introduce noise [cite: 48]
* [cite_start]SHAP analysis will reveal which features are most important for 'sell' signal prediction [cite: 49]

## [cite_start]5. Validation & Evaluation Protocol [cite: 50]

[cite_start]**Time-Series Cross-Validation:** [cite: 51]
* [cite_start]Method: Purged & Embargoed CV $(H=30$, $E=30$) to prevent data leakage [cite: 52]
* [cite_start]Fold Design: Monthly test periods with embargo periods for temporal independence [cite: 53]
* [cite_start]Fair Comparison: All ablation steps $(AO\rightarrow A1\rightarrow A2\rightarrow A3)$ use identical fold indices [cite: 54]

[cite_start]**Evaluation Metrics:** [cite: 55]
* [cite_start]Primary: Macro-F1 (class imbalance consideration) [cite: 56]
* [cite_start]Secondary: Balanced Accuracy, OvR ROC-AUC, Brier Score [cite: 57]
* [cite_start]Statistical Testing: Directional Analysis, Effect Size (Cohen's d), Bootstrap CI (95%), Noise Detection [cite: 58]