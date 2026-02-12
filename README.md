# Credit Card Fraud Detection: From Statistical EDA to Deep Learning

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow.svg)

## 📌 Project Overview
This project addresses the challenge of **Credit Card Fraud Detection** using a dataset of transactions made by European cardholders. The primary obstacle is the extreme class imbalance, where fraudulent transactions represent only **0.17%** of the total data.

The project explores several machine learning paradigms:
1. **Statistical EDA & Preprocessing** (Outlier removal via IQR).
2. **Supervised Learning** (Logistic Regression with Undersampling).
3. **Unsupervised Anomaly Detection** (Isolation Forest & Local Outlier Factor).
4. **Deep Learning** (Neural Networks with Class Weighting).

---

## 📊 Dataset Information
* **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Total Samples:** 284,807
* **Features:** 28 PCA-transformed features ($V1$–$V28$), `Time`, and `Amount`.
* **Target:** `Class` (1 = Fraud, 0 = Legitimate).



---

## 🛠️ Implementation Workflow

### 1. Exploratory Data Analysis (EDA)
* **Correlation Analysis:** Identified that features $V10$, $V12$, and $V14$ are highly negatively correlated with fraudulent transactions.
* **Visualization:** Used `Seaborn` heatmaps to compare feature correlations before and after undersampling.

### 2. Data Cleaning
* **Outlier Removal:** Applied the **Interquartile Range (IQR)** method to features $V10$, $V12$, and $V14$ to remove extreme values that could skew the model's decision boundary.
* **Scaling:** Scaled `Amount` and `Time` features using `StandardScaler` to bring them into the same range as the PCA components.

### 3. Dimensionality Reduction
* Implemented **t-SNE (t-Distributed Stochastic Neighbor Embedding)** to visualize the high-dimensional data in 2D space, confirming that fraud cases are often clustered but require non-linear separation.



### 4. Modeling Strategy
* **Logistic Regression:** Trained on a balanced subset created via `RandomUnderSampler`.
* **Isolation Forest:** An unsupervised approach that isolates anomalies instead of profiling normal points.
* **Neural Network:** A 3-layer architecture using `BatchNormalization` and `Dropout` to handle the complexity of the data without overfitting.

---

## 📈 Key Results
Given the imbalance, the models were evaluated primarily on **AUPRC (Area Under Precision-Recall Curve)** and **Recall**, rather than simple Accuracy.

| Model | Evaluation Metric | Focus |
| :--- | :--- | :--- |
| **Logistic Regression** | AUPRC: ~0.97 | High interpretability |
| **Isolation Forest** | Accuracy: 99.8% | Unsupervised anomaly detection |
| **Deep Learning** | Precision-Recall Focus | Capturing non-linear fraud patterns |



---

## 📂 Repository Structure
* `Fraud_Detection.ipynb`: The main Google Colab notebook containing all code.
* `README.md`: Project documentation.

## 🚀 How to Use
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/abhishekmohank/credit-card-fraud.git](https://github.com/abhishekmohank/Credit-Card-Fraud-Detection-using-Machine-Learning-Deep-Learning)


## Install Dependencies:

Bash

pip install pandas numpy matplotlib seaborn scikit-learn tensorflow imbalanced-learn
Dataset: Ensure creditcard.csv is placed in the directory specified in the notebook (e.g., Google Drive or local path).

