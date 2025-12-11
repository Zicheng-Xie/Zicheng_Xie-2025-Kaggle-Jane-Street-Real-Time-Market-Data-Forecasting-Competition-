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

## Jane Street Real-Time Market Prediction – Final Solution

![python](https://img.shields.io/badge/python-3.10+-blue.svg)
![kaggle](https://img.shields.io/badge/Kaggle-Jane_Street_Real--Time_Market_Prediction-20BEFF.svg)
![model](https://img.shields.io/badge/Model-GBDT%20%2B%20NN%20Ensemble-orange.svg)
![status](https://img.shields.io/badge/Status-Final_Submission-brightgreen.svg)

---

## 🎓 Introduction

This repository contains the final solution for the *Jane Street Real-Time Market Prediction* competition.  
The goal is to **predict `responder_6`** in a real-time setting and to **maximize the Weighted Zero-Mean R-squared (R²)** metric.

The pipeline is designed specifically for **financial time series** with:

- Non-stationarity  
- Heavy-tailed distributions  
- Strong short-term dependencies

At a high level, the solution follows:

> **Feature Engineering → Multi-Model Training → Ensemble → Metric-Aware Output Adjustment**

It blends **Gradient Boosting Decision Trees (GBDT)**, **Neural Networks (NN)** and **multi-tree ensembles** into a robust system.

---

## 🧠 Methodological Framework

### 1️⃣ Objective & Overall Framework

- **Target**: predict `responder_6`.
- **Metric**: Weighted Zero-Mean R²  
  - Optimized with the structure of the metric in mind, especially the weighted covariance term  
    \[
    \sum (w \cdot y)
    \]
- **Core idea**: combine heterogeneous models and then apply a **lightweight, metric-aware adjustment** on the final outputs.

---

### 2️⃣ Feature Construction & Base Models

#### 2.1 Base Features

The solution constructs a rich set of **time-series features** from:

- **79 anonymous features**: `feature_00` – `feature_78`
- **Lag data** from `lags.parquet`

These are augmented with:

- **Lagged responder features**  
  - Historical values of `responder_6` (e.g. previous day per `symbol_id`)  
  - Capture **short-term dynamics** and mean-reversion / momentum effects

- **Preserved identifiers**  
  - `date_id`, `time_id`, `symbol_id` are kept to model:
    - Instrument-specific patterns
    - Intra-day / inter-day behaviors
  - Missing / sparse entries are handled carefully to avoid leakage and instability.

---

#### 2.2 Base Model Families

Three main model families are used:

| Model family | Examples                        | Role in the system                                                  |
|-------------:|---------------------------------|----------------------------------------------------------------------|
| **GBDT**     | XGBoost, LightGBM, CatBoost     | Capture **local nonlinear patterns** and **high-weight samples**    |
| **NN**       | PyTorch deep neural network     | Learn **global, complex feature interactions**                      |
| **Ensemble** | Multi-tree ensemble (voting etc)| **Stabilize and smooth** predictions across different tree models   |

Key notes:

- **GBDTs** focus on strong local nonlinearities and naturally handle missingness.
- The **PyTorch NN** (with batch normalization and attention-style layers) focuses on global structure and cross-feature interactions.
- A **multi-tree ensemble** further combines various tree models to reduce variance.

---

#### 2.3 Inference & Real-Time Considerations

At inference time:

- **Dynamic feature merge**
  - Lag features are joined on-the-fly with the current day’s test data.
- **Weighted model fusion**
  - Model outputs are combined using **fixed weights**:
    - Higher weight on the **strongest GBDT/NN combination**
    - Smaller weights on the **multi-tree ensemble** and standalone NN
- **Robust handling of new symbols**
  - Unseen `symbol_id` values are initialized using **global statistics** so the model remains stable and **real-time ready**.

---

## 🎛 Metric-Aware Post-processing

To better match the **Weighted Zero-Mean R²** evaluation:

1. **Clipping**

   - All predictions are clipped to the range **[−5, 5]**  
   - Prevents rare extreme values from disproportionately harming the score.

2. **Mild scaling**

   - A small **scaling factor** is applied to part of the NN contribution.
   - Leverages the structure of the metric (via the weighted covariance term)  
     while **preserving the sign / direction** of predictions.

---

## 📈 Performance Summary

Quantitative improvements (on validation):

- ✅ **Lag features**  
  → ~**15%** improvement in single-model cross-validation scores.

- ✅ **Heterogeneous model ensembling**  
  → adds about **0.006** to the Weighted Zero-Mean R² over the best single model.

- ✅ **Clipping + scaling** post-processing  
  → additional **0.002 – 0.003** gain.

> 🔚 **Final score**:  
> Weighted Zero-Mean R² ≈ **0.0142** on validation  
> (single models around **0.0117 – 0.0128**, with the fused model outperforming all components).

---


