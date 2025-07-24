# -*- coding: utf-8 -*-
"""
Heart Disease Prediction Streamlit App

This script creates a web application to predict the likelihood of heart disease
based on user-provided medical data. It uses a pre-trained Random Forest model.
(Version 4: Corrected variable names for file handling)
"""

# STEP 1: IMPORT LIBRARIES
# -------------------------
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os # Import the os module to check for file existence

# STEP 2: LOAD DATA AND TRAIN THE MODEL (WITH CACHING)
# ----------------------------------------------------
# Use st.cache_data to load data and train the model only once.
# This improves performance significantly.
@st.cache_data
def load_and_train_model():
    """
    This function loads the dataset, preprocesses it, and trains the
    Random Forest model. It returns the trained model, the scaler,
    and the column names the model was trained on.
    """
    # --- Check if data file exists ---
    data_file = 'heart_disease.csv'
    if not os.path.exists(data_file):
        st.error(f"Error: The data file '{data_file}' was not found.")
        st.info("Please make sure the 'heart_disease.csv' file is in the same directory as your Streamlit app script.")
        return None, None, None # Return None to indicate failure

    # Load the dataset
    df = pd.read_csv(data_file)

    # --- Preprocessing ---
    # 1. Remove duplicates
    df.drop_duplicates(inplace=True)

    # 2. One-hot encode categorical features
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # --- Model Training ---
    # 1. Define features (X) and target (y)
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']

    # 2. Fit the scaler on the full dataset to be used for user input
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train the best model (Random Forest with tuned parameters)
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_scaled, y)

    # Return all the necessary objects for prediction
    return model, scaler, X.columns

# Load the trained model, scaler, and feature names
model, scaler, feature_names = load_and_train_model()

# Title of the app
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction App")

# --- Stop the app if the model failed to load ---
if model is None:
    st.stop()

# STEP 3: CREATE THE STREAMLIT USER INTERFACE
# -------------------------------------------
st.write("This app predicts whether a patient is likely to have heart disease based on their medical attributes. Please enter the patient's details in the sidebar.")

# Sidebar for user input
st.sidebar.header("Patient's Medical Data")

def user_input_features():
    """
    Creates sidebar widgets to get user input and correctly encodes it.
    """
    # --- Collect User Input ---
    age = st.sidebar.slider('Age', 17, 77, 52)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.selectbox('Chest Pain Type (CP)', ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 94, 200, 125)
    chol = st.sidebar.slider('Serum Cholestoral in mg/dl (chol)', 126, 564, 212)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', ('True', 'False'))
    restecg = st.sidebar.selectbox('Resting ECG Results (restecg)', ('Normal', 'ST-T wave abnormality', 'Probable or definite left ventricular hypertrophy'))
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (thalach)', 71, 202, 168)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', ('Yes', 'No'))
    oldpeak = st.sidebar.slider('ST depression (oldpeak)', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of peak exercise ST segment', ('Upsloping', 'Flat', 'Downsloping'))
    ca = st.sidebar.selectbox('Number of major vessels by flourosopy (ca)', (0, 1, 2, 3, 4))
    thal = st.sidebar.selectbox('Thalassemia (thal)', ('Normal', 'Fixed defect', 'Reversible defect'))

    # --- Manually One-Hot Encode the User Input ---
    data = {
        'age': age,
        'trestbps': trestbps,
        'chol': chol,
        'thalach': thalach,
        'oldpeak': oldpeak,
        'sex_1': 1 if sex == 'Male' else 0,
        'cp_1': 1 if cp == 'Atypical Angina' else 0,
        'cp_2': 1 if cp == 'Non-anginal Pain' else 0,
        'cp_3': 1 if cp == 'Asymptomatic' else 0,
        'fbs_1': 1 if fbs == 'True' else 0,
        'restecg_1': 1 if restecg == 'ST-T wave abnormality' else 0,
        'restecg_2': 1 if restecg == 'Probable or definite left ventricular hypertrophy' else 0,
        'exang_1': 1 if exang == 'Yes' else 0,
        'slope_1': 1 if slope == 'Flat' else 0,
        'slope_2': 1 if slope == 'Downsloping' else 0,
        'ca_1': 1 if ca == 1 else 0,
        'ca_2': 1 if ca == 2 else 0,
        'ca_3': 1 if ca == 3 else 0,
        'ca_4': 1 if ca == 4 else 0,
        'thal_1': 1 if thal == 'Normal' else 0,
        'thal_2': 1 if thal == 'Fixed defect' else 0,
        'thal_3': 1 if thal == 'Reversible defect' else 0,
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Ensure the input dataframe has the same columns in the same order as the training data
aligned_df = pd.DataFrame(columns=feature_names, index=[0])
aligned_df.fillna(0, inplace=True)

for col in input_df.columns:
    if col in aligned_df.columns:
        aligned_df[col] = input_df[col].values

# Display the user's input in a clean format
st.subheader('Patient Data Summary')
st.write(input_df)


# STEP 4: MAKE PREDICTION AND DISPLAY RESULT
# ------------------------------------------
if st.button('Predict Heart Disease Likelihood'):
    # Scale the user input
    input_scaled = scaler.transform(aligned_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader('Prediction Result')

    if prediction[0] == 1:
        st.error(f"The model predicts a **HIGH** likelihood of heart disease.")
        st.write(f"Confidence: **{prediction_proba[0][1]*100:.2f}%**")
    else:
        st.success(f"The model predicts a **LOW** likelihood of heart disease.")
        st.write(f"Confidence: **{prediction_proba[0][0]*100:.2f}%**")

    st.write("---")
    st.info("**Disclaimer:** This prediction is based on a machine learning model and is not a substitute for professional medical advice. Please consult a doctor for any health concerns.")
