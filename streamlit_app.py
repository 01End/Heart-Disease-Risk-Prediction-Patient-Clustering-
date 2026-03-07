import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="CardioRisk AI", layout="wide", page_icon="❤️")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active',
]
NUMERIC_FEATURES = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
CLUSTER_FEATURES = ['age', 'weight', 'ap_hi', 'ap_lo', 'cholesterol']

CLUSTER_DESCRIPTIONS = {
    0: "Young, healthy patients with low blood pressure and normal cholesterol",
    1: "Middle-aged patients with elevated cholesterol levels",
    2: "Older patients with moderate blood pressure elevation",
    3: "Patients with high blood pressure and elevated disease risk",
    4: "Healthy middle-aged patients with normal clinical parameters",
    5: "High-risk patients with elevated BP, cholesterol, and older age",
}

# ---------------------------------------------------------------------------
# Cached model / scaler loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_ml_models():
    lr  = joblib.load('logistic_regression_model.pkl')
    rf  = joblib.load('random_forest_model.pkl')
    xgb = joblib.load('xgboost_model.pkl')
    return lr, rf, xgb


@st.cache_resource
def load_scaler():
    return joblib.load('scaler.pkl')


@st.cache_resource
def load_kmeans():
    return joblib.load('kmeans_model.pkl')


@st.cache_resource
def load_cluster_pipeline():
    """Reconstruct the clustering StandardScaler and compute cluster profiles
    from the raw CSV, replicating the notebook preprocessing pipeline."""
    data = pd.read_csv('cardio_train.csv', sep=';')
    data['age'] = (data['age'] / 365).astype(int)
    data = data[(data['ap_hi'] <= 250) & (data['ap_hi'] >= 60)]
    data = data[(data['ap_lo'] <= 200) & (data['ap_lo'] >= 40)]
    data = data[data['ap_hi'] >= data['ap_lo']]
    data = data.drop_duplicates().reset_index(drop=True)

    data_unscaled = data.copy()

    # Apply the same RobustScaler used during training
    scaler = load_scaler()
    data[NUMERIC_FEATURES] = scaler.transform(data[NUMERIC_FEATURES])

    # Fit a StandardScaler on cluster features (replicates the inline scaler
    # used in the notebook before KMeans)
    cluster_scaler = StandardScaler()
    cluster_scaler.fit(data[CLUSTER_FEATURES])

    # Assign clusters to original data for profile computation
    kmeans = load_kmeans()
    x_cluster_scaled = cluster_scaler.transform(data[CLUSTER_FEATURES])
    data_unscaled['cluster'] = kmeans.predict(x_cluster_scaled)

    # Compute average clinical stats per cluster (unscaled for readability)
    profiles = (
        data_unscaled.groupby('cluster')
        .agg(
            avg_age=('age', 'mean'),
            avg_weight=('weight', 'mean'),
            avg_sbp=('ap_hi', 'mean'),
            avg_dbp=('ap_lo', 'mean'),
            avg_cholesterol=('cholesterol', 'mean'),
            disease_rate=('cardio', 'mean'),
            patient_count=('cardio', 'size'),
        )
        .round(2)
    )
    return cluster_scaler, profiles


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def prepare_ml_input(raw: dict) -> pd.DataFrame:
    """Build a single-row DataFrame and apply RobustScaler to numerics."""
    df = pd.DataFrame([raw], columns=FEATURE_NAMES)
    scaler = load_scaler()
    df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])
    return df


def prepare_cluster_input(raw: dict) -> np.ndarray:
    """Scale features for KMeans: RobustScaler → StandardScaler."""
    df = pd.DataFrame([raw], columns=FEATURE_NAMES)
    scaler = load_scaler()
    df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])
    cluster_scaler, _ = load_cluster_pipeline()
    return cluster_scaler.transform(df[CLUSTER_FEATURES])


# ---------------------------------------------------------------------------
# Sidebar — navigation + patient input
# ---------------------------------------------------------------------------
st.sidebar.title(" CardioRisk AI")
st.sidebar.markdown(
    "AI-powered cardiovascular disease risk prediction and patient clustering "
    "using machine learning and clinical data."
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    " Navigation",
    ["Patient Risk Prediction", "Cluster Assignment",
    "Batch Processing"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Patient Information")

age        = st.sidebar.slider("Age", 20, 80, 50)
gender_str = st.sidebar.selectbox("Gender", ["Female", "Male"])
gender     = 1 if gender_str == "Female" else 2
height     = st.sidebar.slider("Height (cm)", 140, 220, 170)
weight     = st.sidebar.slider("Weight (kg)", 40.0, 200.0, 75.0, step=0.5)
ap_hi      = st.sidebar.slider("Systolic Blood Pressure", 60, 250, 120)
ap_lo      = st.sidebar.slider("Diastolic Blood Pressure", 40, 200, 80)

chol_map   = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
chol_str   = st.sidebar.selectbox("Cholesterol", list(chol_map.keys()))
cholesterol = chol_map[chol_str]

gluc_map   = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
gluc_str   = st.sidebar.selectbox("Glucose", list(gluc_map.keys()))
gluc       = gluc_map[gluc_str]

smoke  = int(st.sidebar.checkbox("Smoker"))
alco   = int(st.sidebar.checkbox("Alcohol Consumer"))
active = int(st.sidebar.checkbox("Physically Active", value=True))

patient_data = {
    'age': age, 'gender': gender, 'height': height, 'weight': weight,
    'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol,
    'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active,
}

# Global validation
if ap_hi < ap_lo:
    st.warning(" Systolic blood pressure must be higher than diastolic blood pressure.")


# ======================================================================
# PAGE 1 — Patient Risk Prediction
# ======================================================================
if page == "Patient Risk Prediction":
    st.title(" Cardiovascular Risk Assessment")
    st.markdown(
        "Enter patient clinical parameters in the sidebar and click "
        "**Assess Risk** to predict cardiovascular disease risk."
    )

    if st.button(" Assess Risk", type="primary"):
        input_df = prepare_ml_input(patient_data)
        lr, rf, xgb = load_ml_models()

        lr_prob  = lr.predict_proba(input_df)[0][1]
        rf_prob  = rf.predict_proba(input_df)[0][1]
        xgb_prob = xgb.predict_proba(input_df)[0][1]

        # Primary result
        st.subheader("Primary Assessment (XGBoost)")
        if xgb_prob >= 0.5:
            st.error("🔴 High Cardiovascular Risk")
        else:
            st.success("🟢 Low Cardiovascular Risk")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Probability", f"{xgb_prob:.1%}")
        with col2:
            st.markdown("**Risk Level**")
            st.progress(float(xgb_prob))

        # Model comparison
        st.subheader("Model Comparison")
        comparison = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
            'Prediction': [
                'High Risk' if p >= 0.5 else 'Low Risk'
                for p in (lr_prob, rf_prob, xgb_prob)
            ],
            'Probability': [
                f"{p:.1%}" for p in (lr_prob, rf_prob, xgb_prob)
            ],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)


# ======================================================================
# PAGE 2 — Cluster Assignment
# ======================================================================
elif page == "Cluster Assignment":
    st.title(" Patient Cluster Profile")
    st.markdown(
        "Identify which clinical cluster this patient belongs to "
        "based on K-Means clustering (k=6)."
    )

    if st.button(" Assign Cluster", type="primary"):
        cluster_input = prepare_cluster_input(patient_data)
        kmeans = load_kmeans()
        cluster_id = int(kmeans.predict(cluster_input)[0])

        st.info(f" This patient belongs to **Cluster {cluster_id}**")
        desc = CLUSTER_DESCRIPTIONS.get(cluster_id, "No description available.")
        st.markdown(f"**Cluster Description:** {desc}")

        _, profiles = load_cluster_pipeline()

        st.subheader("Cluster Profiles — Average Clinical Statistics")
        display = profiles.copy()
        display.index.name = "Cluster"
        display.columns = [
            "Avg Age", "Avg Weight", "Avg SBP", "Avg DBP",
            "Avg Cholesterol", "Disease Rate", "Patient Count",
        ]
        display["Disease Rate"] = (profiles["disease_rate"] * 100).round(1).astype(str) + "%"

        def _highlight_row(row):
            color = "background-color: #fff3cd" if row.name == cluster_id else ""
            return [color] * len(row)

        st.dataframe(
            display.style.apply(_highlight_row, axis=1),
            use_container_width=True,
        )

        # Cluster‐description summary table
        st.subheader("Cluster Descriptions")
        for cid in sorted(profiles.index):
            label = CLUSTER_DESCRIPTIONS.get(cid, "—")
            st.markdown(f"- **Cluster {cid}:** {label}")
# ======================================================================
# PAGE 3 — Batch Processing
# ======================================================================
elif page == "Batch Processing":
    st.title(" Batch Patient Assessment")
    st.markdown(
        "Upload a CSV file with patient data to assess cardiovascular risk "
        "for multiple patients simultaneously."
    )
    st.markdown(
        "**Expected columns:** `age, gender, height, weight, ap_hi, ap_lo, "
        "cholesterol, gluc, smoke, alco, active`"
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            missing = [c for c in FEATURE_NAMES if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
                st.stop()

            st.subheader("Raw Data Preview")
            st.dataframe(batch_df.head(20), use_container_width=True, hide_index=True)

            # Preprocess
            batch_features = batch_df[FEATURE_NAMES].copy()
            scaler = load_scaler()
            batch_features[NUMERIC_FEATURES] = scaler.transform(
                batch_features[NUMERIC_FEATURES]
            )

            _, _, xgb = load_ml_models()
            probs = xgb.predict_proba(batch_features)[:, 1]

            results_df = batch_df.copy()
            results_df['Risk_Probability'] = probs.round(4)
            results_df['Risk_Level'] = np.where(probs >= 0.5, 'High', 'Low')

            # Summary metrics
            st.subheader("Summary")
            total     = len(results_df)
            high_risk = int((results_df['Risk_Level'] == 'High').sum())
            low_risk  = int((results_df['Risk_Level'] == 'Low').sum())
            avg_prob  = float(probs.mean())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Patients", total)
            c2.metric("High Risk", high_risk)
            c3.metric("Low Risk", low_risk)
            c4.metric("Avg Risk Probability", f"{avg_prob:.1%}")

            # Color-coded results table
            st.subheader("Prediction Results")

            def _color_risk(row):
                bg = "#f8d7da" if row['Risk_Level'] == 'High' else "#d4edda"
                return [f"background-color: {bg}"] * len(row)

            st.dataframe(
                results_df.style.apply(_color_risk, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            # Bar chart
            st.subheader("Risk Distribution")
            risk_counts = results_df['Risk_Level'].value_counts()
            palette = {'High': '#e74c3c', 'Low': '#2ecc71'}

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(
                risk_counts.index,
                risk_counts.values,
                color=[palette.get(x, '#999') for x in risk_counts.index],
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(int(bar.get_height())),
                    ha='center', va='bottom', fontweight='bold',
                )
            ax.set_ylabel('Count')
            ax.set_title('High vs Low Risk Patients')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Download
            csv_export = results_df.to_csv(index=False)
            st.download_button(
                label=" Download Results CSV",
                data=csv_export,
                file_name="cardiorisk_batch_results.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")
