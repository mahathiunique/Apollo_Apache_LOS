import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/xgb_los_model.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="ICU AI Prediction",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

body {
    background-color: #eef2f7;
}

.main {
    background-color: #eef2f7;
}

.block-container {
    padding-top: 2rem;
}

/* Hide Streamlit header */

header {
    visibility: hidden;
}

/* Main Title */

.title-text {

    text-align: center;

    font-size: 48px;
    font-weight: bold;

    color: #1e293b;
}

.subtitle {

    text-align: center;

    font-size: 20px;

    color: #475569;

    margin-bottom: 30px;
}

/* Card */

.card {

    background: white;

    padding: 30px;

    border-radius: 25px;

    margin-bottom: 25px;

    box-shadow:
    0px 10px 30px rgba(0,0,0,0.08);
}

/* Section Title */

.section-title {

    font-size: 30px;
    font-weight: bold;

    color: #1e293b;

    margin-bottom: 20px;
}

/* Prediction Card */

.prediction-card {

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #06b6d4
    );

    padding: 40px;

    border-radius: 30px;

    color: white;

    text-align: center;

    box-shadow:
    0px 10px 30px rgba(0,0,0,0.15);
}

.big-number {

    font-size: 70px;
    font-weight: bold;
}

/* Buttons */

.stButton>button {

    width: 100%;

    height: 60px;

    border-radius: 18px;

    border: none;

    background:
    linear-gradient(
        90deg,
        #2563eb,
        #06b6d4
    );

    color: white;

    font-size: 22px;
    font-weight: bold;
}

.stButton>button:hover {

    color: white;
    opacity: 0.9;
}

/* Risk */

.low-risk {

    background: #dcfce7;
    color: #166534;

    padding: 18px;

    border-radius: 18px;

    text-align: center;

    font-size: 24px;
    font-weight: bold;
}

.medium-risk {

    background: #fef9c3;
    color: #854d0e;

    padding: 18px;

    border-radius: 18px;

    text-align: center;

    font-size: 24px;
    font-weight: bold;
}

.high-risk {

    background: #fee2e2;
    color: #991b1b;

    padding: 18px;

    border-radius: 18px;

    text-align: center;

    font-size: 24px;
    font-weight: bold;
}

.info-box {

    background: #f8fafc;

    padding: 15px;

    border-radius: 15px;

    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="title-text">
🏥 ICU AI Prediction Dashboard
</div>

<div class="subtitle">
Advanced APACHE-style ICU LOS Prediction System
</div>
""", unsafe_allow_html=True)

# ==========================================
# INPUT SECTION
# ==========================================

st.markdown("""
<div class="card">
<div class="section-title">
🩺 Patient Details
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:

    Age = st.number_input("Age", 0, 120, 45)

    Temperature = st.number_input(
        "Temperature",
        value=37.0
    )

    HeartRate = st.number_input(
        "Heart Rate",
        value=80.0
    )

    RespiratoryRate = st.number_input(
        "Respiratory Rate",
        value=18.0
    )

    MeanArterialPressure = st.number_input(
        "Mean Arterial Pressure",
        value=90.0
    )

    FiO2 = st.number_input(
        "FiO2",
        value=21.0
    )

    pO2 = st.number_input(
        "pO2",
        value=95.0
    )

    pCO2 = st.number_input(
        "pCO2",
        value=40.0
    )

with col2:

    ArterialpH = st.number_input(
        "Arterial pH",
        value=7.4
    )

    Sodium = st.number_input(
        "Sodium",
        value=140.0
    )

    UrineOutput = st.number_input(
        "Urine Output",
        value=1500.0
    )

    Creatinine = st.number_input(
        "Creatinine",
        value=1.0
    )

    Urea = st.number_input(
        "Urea",
        value=30.0
    )

    BSL = st.number_input(
        "Blood Sugar Level",
        value=100.0
    )

    Albumin = st.number_input(
        "Albumin",
        value=4.0
    )

    Bilirubin = st.number_input(
        "Bilirubin",
        value=1.0
    )

with col3:

    Hematocrit = st.number_input(
        "Hematocrit",
        value=40.0
    )

    WBC = st.number_input(
        "WBC",
        value=8000.0
    )

    ApsScore = st.number_input(
        "APS Score",
        value=50.0
    )

    ApacheivScore = st.number_input(
        "Apache IV Score",
        value=50.0
    )

    MechanicalVentilation = st.selectbox(
        "Mechanical Ventilation",
        ["No", "Yes"]
    )

    Gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Others"]
    )

    EmergencySurgery = st.selectbox(
        "Emergency Surgery",
        [0,1]
    )

    Readmission = st.selectbox(
        "Readmission",
        [0,1]
    )

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# PATIENT INFO DASHBOARD
# ==========================================

st.markdown("""
<div class="card">
<div class="section-title">
📋 Patient Information Dashboard
</div>
""", unsafe_allow_html=True)

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:

    st.markdown(f"""
    <div class="info-box">
    <b>Age:</b> {Age}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <b>Temperature:</b> {Temperature}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <b>Heart Rate:</b> {HeartRate}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <b>MAP:</b> {MeanArterialPressure}
    </div>
    """, unsafe_allow_html=True)

with info_col2:

    st.markdown(f"""
    <div class="info-box">
    <b>FiO2:</b> {FiO2}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <b>pO2:</b> {pO2}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <b>WBC:</b> {WBC}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <b>APS Score:</b> {ApsScore}
    </div>
    """, unsafe_allow_html=True)

with info_col3:

    st.markdown(f"""
    <div class="info-box">
    <b>Mechanical Ventilation:</b> {MechanicalVentilation}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <b>Gender:</b> {Gender}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <b>Emergency Surgery:</b> {EmergencySurgery}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <b>Readmission:</b> {Readmission}
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# PREDICT BUTTON
# ==========================================

predict_button = st.button(
    "Predict ICU LOS"
)

# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    input_data = {

        'Age': Age,
        'Temperature': Temperature,
        'MeanArterialPressure': MeanArterialPressure,
        'HeartRate': HeartRate,
        'RespiratoryRate': RespiratoryRate,
        'FiO2': FiO2,
        'pO2': pO2,
        'pCO2': pCO2,
        'ArterialpH': ArterialpH,
        'Sodium': Sodium,
        'UrineOutput': UrineOutput,
        'Creatinine': Creatinine,
        'Urea': Urea,
        'BSL': BSL,
        'Albumin': Albumin,
        'Bilirubin': Bilirubin,
        'Hematocrit': Hematocrit,
        'WBC': WBC,
        'IsGCSNotAvailable': 0,
        'GCSEyes': 4,
        'GCSVerbal': 5,
        'GCSMotor': 6,
        'CRF': 0,
        'Lymphoma': 0,
        'Cirrhosis': 0,
        'Leukemia': 0,
        'HepaticFailure': 0,
        'Immunosuppression': 0,
        'MetastaticCarcinoma': 0,
        'AIDS': 0,
        'PreICULengthOfStay': 0,
        'DiagnosisType': 1,
        'Origin': 1,
        'EmergencySurgery': EmergencySurgery,
        'Readmission': Readmission,
        'Thrombolysis': 0,
        'RespiratoryQuotient': 0.8,
        'AtmosphericPressure': 760,
        'CreatedBy': 1,
        'ApacheivScore': ApacheivScore,
        'ApsScore': ApsScore,
        'UNIT_ID': 1,
        'RNK': 1,
        'LOCATIONID': 1,
        'PERIOD_WID': 1,

        'MecanicalVentilation_b':
        1 if MechanicalVentilation == "Yes" else 0,

        'Gender_Male':
        1 if Gender == "Male" else 0,

        'Gender_Others':
        1 if Gender == "Others" else 0
    }

    input_df = pd.DataFrame([input_data])

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction_log = model.predict(
        input_df
    )[0]

    prediction = np.expm1(
        prediction_log
    )

    st.markdown(f"""
    <div class="prediction-card">

    <h2>
    🤖 AI Prediction
    </h2>

    <div class="big-number">
    {prediction:.1f}
    </div>

    <h3>
    ICU Days
    </h3>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction < 3:

        st.markdown(
            '<div class="low-risk">🟢 LOW RISK</div>',
            unsafe_allow_html=True
        )

    elif prediction < 7:

        st.markdown(
            '<div class="medium-risk">🟡 MODERATE RISK</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="high-risk">🔴 HIGH RISK</div>',
            unsafe_allow_html=True
        )