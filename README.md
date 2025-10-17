# DA5401: Assignment 6 — Imputation via Regression for Missing Data

## Student Details

* **Name:** Chirag
* **Roll Number:** DA25M008

---

## 📌 Assignment Overview

This project explores how **different imputation strategies for missing data** impact the performance of downstream machine learning models.
Using the **UCI Credit Card Default Clients dataset**, we simulate missing values and apply both **simple** and **regression-based** imputation methods.

The assignment compares:

* Simple median-based imputation (baseline)
* Linear regression-based imputation
* Non-linear regression-based imputations (KNN & Decision Tree)
* Listwise deletion (removing missing rows)

The goal is to **quantitatively assess** how imputation affects the classification of credit card defaults using a **Logistic Regression** model.

---

## ⚙️ Methodology

### 1. Dataset & Missing Value Simulation

* Dataset: **UCI Credit Card Default Clients** (Kaggle version).
* Target: `default.payment.next.month`.
* Artificially introduced **Missing At Random (MAR)** values (≈7%) in three numeric columns:

  * `AGE`
  * `BILL_AMT1`
  * `PAY_AMT1`
* Verified missingness via summary statistics and heatmaps.

---

### 2. Imputation Strategies

| Model | Strategy                                | Description                                                                      |
| :---- | :-------------------------------------- | :------------------------------------------------------------------------------- |
| **A** | **Median Imputation**                   | Filled missing values with the median of each column.                            |
| **B** | **Linear Regression Imputation**        | Predicted missing values using a linear model based on other numeric predictors. |
| **C** | **KNN Regression Imputation**           | Used neighboring samples (`k=5`) to estimate missing values.                     |
| **E** | **Decision Tree Regression Imputation** | Modeled non-linear feature relationships using tree splits.                      |
| **D** | **Listwise Deletion**                   | Dropped all rows containing missing values.                                      |

---

### 3. Model Training & Evaluation

* **Classifier:** Logistic Regression (`liblinear`, max_iter=500).
* **Data Split:** 80% training, 20% testing (stratified).
* **Preprocessing:** StandardScaler for normalization.
* **Metrics Evaluated:**

  * Accuracy
  * Precision
  * Recall
  * F1-Score

Each dataset (A–E) was trained and evaluated independently.

---

## 📊 Results Summary

| Dataset | Imputation Type          |  Accuracy  |  Precision |   Recall   |  F1-Score  |
| :-----: | :----------------------- | :--------: | :--------: | :--------: | :--------: |
|    A    | Median                   |   0.8080   |   0.6906   |   0.2389   |   0.3550   |
|    B    | Linear Regression        |   0.8145   |   0.7441   |   0.2456   |   0.3694   |
|    C    | KNN Regression           |   0.8143   |   0.7421   |   0.2456   |   0.3691   |
|    D    | Listwise Deletion        |   0.8142   |   0.7544   |   0.2409   |   0.3652   |
|    E    | Decision Tree Regression | **0.8146** | **0.7447** | **0.2465** | **0.3704** |

---

## 🔍 Comparative Analysis

* **Median Imputation (A)** – Fast, robust to outliers, but ignores feature interdependence.
* **Linear Regression (B)** – Incorporates linear dependencies and improves performance over the baseline.
* **KNN Regression (C)** – Captures local non-linear structure, with similar results to linear regression.
* **Decision Tree Regression (E)** – Slightly outperforms others by modeling threshold-based and interaction effects.
* **Listwise Deletion (D)** – Performs acceptably but wastes valuable data.

---

## ✅ Conclusion

* **Regression-based imputations** consistently outperform median imputation and listwise deletion.
* **Decision Tree Regression (Model E)** achieved the best balance of accuracy and recall, confirming its strength in modeling complex feature relations.
* Even modest F1 improvements are meaningful in **credit-risk modeling**, where detecting potential defaults (recall) is critical.
* Imputation is not just preprocessing — it’s an integral part of model reliability and performance.

> **Recommended Approach:** Decision Tree Regression Imputation provides the most effective, interpretable, and data-efficient method for this task.
