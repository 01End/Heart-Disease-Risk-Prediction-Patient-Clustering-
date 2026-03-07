# Heart Disease Risk Prediction & Patient Clustering

End-to-end machine learning pipeline for cardiovascular disease prediction and patient segmentation using 70,000 clinical records — featuring ensemble models, deep learning, patient clustering, and an interactive Streamlit deployment with SHAP explainability.

---

## Demo

![App Showcase](Heart_Disease_Risk_Prediction___Patient_Clustering_app_showcase.gif)

---

## Overview

This project applies a full machine learning pipeline to the Cardiovascular Disease Dataset (70,000 patients, 12 clinical features). The goal is to predict cardiovascular disease risk and segment patients into clinically meaningful clusters for personalized prevention programs.

---

## Project Structure

```
heart-disease-prediction/
│
├── Heart_Disease_Risk_Prediction.ipynb     # Main notebook
├── streamlit_app.py                        # Streamlit web application
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── kmeans_model.pkl
│   ├── deep_learning_baseline_model.h5
│   ├── deep_learning_autoencoder_model.h5
│   ├── wide_deep_model.h5
│   ├── scaler.pkl
│   └── scaler_dl.pkl
│
├── data/
│   └── cardio_train.csv                    # Source dataset (not included)
│
├── requirements.txt
└── README.md
```

> Models (.pkl, .h5) and dataset are not included due to file size. Run the notebook to regenerate them locally.

---

## Dataset

| Property | Details |
|---|---|
| **Source** | [Cardiovascular Disease Dataset — Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) |
| **Size** | 70,000 patients |
| **Features** | 12 clinical features |
| **Target** | `cardio` (1 = cardiovascular disease, 0 = healthy) |

**Features include:** age, gender, height, weight, systolic BP, diastolic BP, cholesterol, glucose, smoking, alcohol consumption, physical activity.

---

## Pipeline

### 1. Data Preprocessing
- Converted age from days to years
- Removed physiologically impossible blood pressure values
- Handled class imbalance using SMOTE
- Applied RobustScaler to numerical features

### 2. Exploratory Data Analysis
- Correlation heatmap across all 12 clinical features
- Age distribution with 5-year bins and disease prevalence
- Blood pressure categorization (Normal / Stage 1 / Stage 2 Hypertension)
- Interactive Plotly dashboards for risk factor combinations
- Chi-square and t-tests for statistical significance

### 3. Unsupervised ML
- **KMeans** (k=6) — distinct patient phenotypes
- **DBSCAN** — outlier patient detection
- **Hierarchical Clustering** — nested patient groupings
- **t-SNE** — 2D cluster visualization
- Silhouette score validation and per-cluster clinical profiling

### 4. Supervised ML
- **Logistic Regression** — interpretable baseline with L2 regularization
- **Random Forest** — 200 estimators with feature importance
- **XGBoost** — tuned with n_estimators=500, max_depth=8
- Stratified 5-fold cross-validation with AUC-ROC scoring
- Calibration curves for prediction reliability

### 5. Deep Learning
- **Baseline DNN** — 5 hidden layers (256→128→64→32→16) + BatchNorm + Dropout(0.3)
- **Wide & Deep Architecture** — Keras functional API
- **Autoencoder** (12→8→4→8→12) for unsupervised feature learning
- DNN trained on autoencoder-encoded features + cluster memberships

### 6. Streamlit Deployment
- Patient data input with real-time cardiovascular risk prediction
- Cluster assignment showing similar patient profiles
- SHAP force plots explaining individual predictions
- Batch processing for multiple patient assessments

---

## Results

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 0.7272 | 0.7925 |
| Random Forest | 0.7103 | 0.7694 |
| **XGBoost** | **0.7348** | 0.7982 |
| DNN Baseline | 0.7327 | 0.8004 |
| DNN + Autoencoder | 0.7347 | **0.8009** |
| Wide & Deep | 0.7329 | 0.8000 |

**XGBoost** achieved the best accuracy. **DNN + Autoencoder** achieved the best AUC-ROC, showing deep learning's advantage in capturing non-linear cardiovascular risk patterns.

---

## Streamlit App Pages

| Page | Description |
|---|---|
| Patient Risk Prediction | Enter 11 clinical parameters, get real-time risk assessment from all 3 models |
| Cluster Assignment | Identify which patient cluster the input belongs to with clinical profile comparison |
| SHAP Explanation | SHAP force plot and feature importance bar chart for the XGBoost prediction |
| Batch Processing | Upload a CSV of multiple patients and download results with risk labels |

---

## Tech Stack

- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **ML:** scikit-learn, xgboost, imbalanced-learn
- **Deep Learning:** TensorFlow / Keras
- **Explainability:** SHAP
- **Deployment:** Streamlit

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/01End/Heart-Disease-Risk-Prediction.git
cd Heart-Disease-Risk-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and place in the project folder
#    https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

# 4. Run the notebook to generate all model files
jupyter notebook Heart_Disease_Risk_Prediction.ipynb

# 5. Launch the Streamlit app
streamlit run streamlit_app.py
```

---

## Requirements

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

---

## Authors

Developed as part of a university final project.

---

## License

This project is for academic purposes only.
