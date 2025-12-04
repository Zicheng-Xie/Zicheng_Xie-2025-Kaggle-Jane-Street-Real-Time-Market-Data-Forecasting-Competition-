# Zicheng_Xie-2025-Kaggle-Jane-Street-Real-Time-Market-Data-Forecasting-Competition-
A _**Silver Medal (67th/3757 teams)**_ implementation for 2025 Jane Street Real-Time Market Data Forecasting competition on Kaggle

**Code Description**
Overall Submission Code:  kaggle-0-0083.ipynb

**Code Solution 1**
1. Test Score Code for NN Network:  jane-street-rmf-inference-nn-xgb.ipynb
2. Training Score Code for NN Network:  jane-street-rmf-training-nn.ipynb
3. Data Processing Code:  js24-preprocessing-create-lags.ipynb
4. GBDT Training Code:  js24-train-gbdt-model-with-lags-singlemodel.ipynb

**Code Solution 2**
1. Transformer Code:  jane-street-tabm-ft-transformer-inference (1).ipynb
2. GBDT + Rolling Code:  js-single-model-baseline-n-lags-chinese-ver.ipynb

## Jane Street Real-Time Market Prediction: Solution Overview

### 1. Objective and Overall Framework

In this Jane Street real-time market prediction competition, the main goal of the solution is to accurately predict **`responder_6`** and to maximize the **Weighted Zero-Mean R-squared (R²)** metric. The method is designed with financial time series properties in mind, such as **non-stationarity** and **heavy-tailed distributions**.

The overall framework follows a full pipeline:

> **Feature engineering → Multi-model training → Ensemble → Output adjustment**

It combines:
- **Gradient Boosting Decision Trees (GBDT)**  
- **Neural Networks (NN)**  
- **Multi-tree ensembles**

Special attention is given to **lagged features** and to the fact that the evaluation metric depends on the weighted covariance term \(\sum (w \cdot y)\), which allows for small but effective adjustments to the prediction scale at the output stage.

---

### 2. Feature Construction and Base Models

The solution first constructs time-series features from:
- **79 anonymous variables:** `feature_00` – `feature_78`  
- **Lag data:** `lags.parquet`

Key design choices:

- **Lagged responder features**  
  Lagged values of `responder_6` (e.g., the previous day’s value for each `symbol_id`) are added to capture short-term dynamics in the financial series.

- **Preservation of identifiers**  
  The identifiers `date_id`, `time_id`, and `symbol_id` are retained to model patterns specific to each asset and time step, while missing or sparse entries are handled carefully.

- **Three types of base models**

  | Model family | Examples                            | Role in the system                                  |
  |-------------|--------------------------------------|-----------------------------------------------------|
  | **GBDT**    | XGBoost, LightGBM, CatBoost          | Capture local nonlinear patterns and high-weight samples |
  | **NN**      | PyTorch deep neural network          | Learn global, complex feature interactions          |
  | **Ensemble**| Multi-tree ensemble (e.g. voting regressor) | Stabilize and smooth predictions across trees |

GBDT models focus on local nonlinear patterns and on samples with higher weights. The PyTorch neural network, with batch normalization and attention mechanisms, is used to learn complex global interactions. A multi-tree ensemble further combines several tree models (e.g., via a voting regressor) to stabilize performance.

At inference time:
- Model outputs are **combined using fixed weights** (with higher weight on the strongest GBDT/NN combination, and smaller weights on the multi-tree ensemble and the standalone NN).
- **Lag features are dynamically merged** with the current day’s test data.
- New or unseen `symbol_id` values are initialized using **global statistics** to maintain robustness and real-time suitability.

---

### 3. Metric-Aware Post-processing and Performance

To better match the **Weighted Zero-Mean R-squared** metric, the predictions are post-processed in two steps:

1. **Clipping**  
   All outputs are clipped to the range **[–5, 5]** to avoid extreme values that could harm the score.

2. **Mild scaling**  
   A small scaling factor is applied to part of the NN contribution to exploit the structure of the Weighted Zero-Mean R² formula, without changing the overall prediction direction.

Quantitatively:
- The use of **lag features** improves single-model cross-validation scores by **around 15%**.
- **Heterogeneous model ensembling** adds about **0.006** to the metric.
- The **clipping-plus-scaling** strategy brings an additional gain of **0.002–0.003**.

Overall, the final system achieves a **Weighted Zero-Mean R-squared of 0.0142** on the validation set (about a **21% improvement** over the baseline), with single models around **0.0117–0.0128** and the fused model **outperforming all individual components**.

