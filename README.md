# 🏥 ICU AI Prediction System

An Explainable AI-driven ICU Length of Stay (LOS) Prediction System developed using APACHE-style physiological variables, Extreme Gradient Boosting (XGBoost), SHAP Explainability, and Streamlit-based clinical visualization.

## 🚀 Core Features

- ICU Length of Stay (LOS) Prediction
- APACHE-style Severity Assessment
- Explainable AI using SHAP
- Interactive Clinical Dashboard
- Risk Stratification System
- Real-time Patient Parameter Analysis

---

## 🧠 Machine Learning Pipeline

### Model Architecture

- Algorithm: XGBoost Regressor (`XGBRegressor`)
- Learning Type: Supervised Regression
- Explainability Framework: SHAP (SHapley Additive Explanations)

### Data Preprocessing

- Duplicate Record Removal
- Negative LOS Filtering
- Missing Value Imputation
- Leakage Prevention
- One-Hot Encoding
- Log Transformation of LOS

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| R² Score | 0.867 |
| MAE | 0.48 |
| RMSE | 0.70 |

---

## 🏥 Clinical Variables

The model utilizes APACHE-style ICU physiological variables including:

- Age
- Temperature
- Mean Arterial Pressure
- Heart Rate
- Respiratory Rate
- FiO2
- pO2
- pCO2
- Arterial pH
- Sodium
- Creatinine
- Urea
- Bilirubin
- Hematocrit
- WBC
- APS Score
- Mechanical Ventilation
- GCS Parameters
- Urine Output
- Emergency Surgery
- Readmission Status

---

## 🛠️ Technology Stack

- Python
- Jupyter Notebook
- XGBoost
- SHAP
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 👩‍💻 Developer

Mahathi M  
Chennai Institute of Technology (CIT)

---

## 📌 Note

This project was developed for ICU AI research, explainable clinical prediction workflows, and healthcare machine learning experimentation.
