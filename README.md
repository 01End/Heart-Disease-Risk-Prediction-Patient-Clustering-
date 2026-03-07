#  Heart Disease Risk Prediction & Patient Clustering

> End-to-end ML pipeline for cardiovascular disease prediction using 70,000 patient records — featuring ensemble models, deep learning, and interactive Streamlit deployment.

---

##  Overview

This project is part of a university final project series applying a full machine learning pipeline to real-world healthcare data. Using the **Cardiovascular Disease Dataset** (70,000 patients, 12 clinical features), the goal is to predict cardiovascular disease risk and segment patients into clinically meaningful clusters for personalized prevention programs.

---

##  Project Structure

```
heart-disease-prediction/
│
├── Heart_Disease_Risk_Prediction.ipynb   # Main notebook
├── streamlit_app.py                      # Streamlit web application
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── deep_learning_baseline_model.h5
│   ├── deep_learning_autoencoder_model.h5
│   ├── wide_deep_model.h5
│   ├── scaler.pkl
│   └── scaler_dl.pkl
│
├── data/
│   └── cardio_train.csv                  # Source dataset (semicolon-separated)
│
└── README.md
```

---

##  Dataset

| Property | Details |
|---|---|
| **Source** | [Cardiovascular Disease Dataset — Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) |
| **Size** | 70,000 patients |
| **Features** | 12 clinical features |
| **Target** | `cardio` (1 = cardiovascular disease, 0 = healthy) |

**Features include:** age, gender, height, weight, systolic BP (`ap_hi`), diastolic BP (`ap_lo`), cholesterol, glucose, smoking, alcohol consumption, physical activity.

---

##  Pipeline

### 1.  Data Preprocessing
- Converted age from days to years
- Removed physiologically impossible blood pressure values (`ap_hi > 250`, `ap_lo > 200`)
- Handled duplicates and class imbalance using **SMOTE**
- Encoded categorical variables (cholesterol, glucose — ordinal)
- Scaled numerical features with **RobustScaler**

### 2.  Exploratory Data Analysis
- Correlation heatmap across all 12 clinical features
- Age distribution with 5-year bins and disease prevalence
- Blood pressure categorization (Normal / Stage 1 / Stage 2 Hypertension)
- Interactive Plotly dashboards for risk factor combinations
- Chi-square and t-tests for statistical significance across demographic groups

### 3.  Unsupervised ML (Clustering)
- **KMeans** (k=6) — distinct patient phenotypes
- **DBSCAN** — outlier patient detection
- **Hierarchical Clustering** — nested patient groupings via dendrogram
- **t-SNE** — 2D cluster visualization
- Silhouette score validation + per-cluster clinical profiling

### 4.  Supervised ML (Classification)
- **Logistic Regression** — interpretable baseline with L2 regularization
- **Random Forest** — 200 estimators with feature importance
- **XGBoost** — tuned with `n_estimators=500`, `max_depth=8`
- Stratified 5-fold cross-validation with AUC-ROC scoring
- Calibration curves for prediction reliability

### 5.  Deep Learning
- **Baseline DNN** — 5 hidden layers (256→128→64→32→16) + BatchNorm + Dropout(0.3)
- **Wide & Deep Architecture** — Keras functional API combining raw features and deep embeddings
- **Autoencoder** (12→8→4→8→12) for unsupervised feature learning
- DNN trained on autoencoder-encoded features + cluster memberships

##  Results

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 0.7272 | 0.7925 |
| Random Forest | 0.7103 | 0.7694 |
| **XGBoost** | **0.7348** | 0.7982 |
| DNN Baseline | 0.7327 | 0.8004 |
| DNN + Autoencoder | 0.7347 | **0.8009** |
| Wide & Deep | 0.7329 | 0.8000 |

> **XGBoost** achieved the best accuracy. **DNN + Autoencoder** achieved the best AUC-ROC, showing deep learning's advantage in capturing non-linear cardiovascular risk patterns.

---

##  Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-yellowgreen)
![XGBoost](https://img.shields.io/badge/XGBoost-1.x-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b)

- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **ML:** scikit-learn, xgboost, imbalanced-learn
- **Deep Learning:** TensorFlow / Keras
- **Deployment:** Streamlit
- **Explainability:** SHAP

---

##  Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and place in /data
#    https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

# 4. Run the notebook
jupyter notebook Heart_Disease_Risk_Prediction.ipynb

# 5. Launch the Streamlit app
streamlit run streamlit_app.py
```

---

##  Requirements

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
xgboost
imbalanced-learn
tensorflow
streamlit
shap
joblib
scipy
```
