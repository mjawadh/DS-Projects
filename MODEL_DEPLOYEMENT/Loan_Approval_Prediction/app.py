import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import xgboost as xgb  # For potential XGBoost model
import csv
from datetime import datetime

# Set page configuration for better layout
st.set_page_config(layout="wide")
st.title("🤖 Loan Approval Prediction App")
st.markdown("Enter loan application details to predict approval status and probability.")
st.markdown("Note: Predictions and inputs are logged to a temporary file (locally or in /tmp on cloud).")

# Load model and scaler
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, 'model.pkl')
scaler_path = os.path.join(script_dir, 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.write("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Model or scaler file not found: {e}. Ensure 'model.pkl' and 'scaler.pkl' are in {script_dir}.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Define expected feature names based on training data
expected_features = [
    'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
    'bank_asset_value', 'education_ Not Graduate', 'self_employed_ Yes'
]

# Create input form
st.header("Enter Loan Application Details")
with st.form(key="loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        no_of_dependents = st.number_input("No of Dependents", min_value=0, max_value=10, value=0)
        income_annum = st.number_input("Annual Income", min_value=0, value=5000000)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=15000000)
        loan_term = st.number_input("Loan Term (months)", min_value=0, value=180)
        cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
        
    with col2:
        residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=5000000)
        commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=0)
        luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=1000000)
        bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=2000000)
        education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", options=["Yes", "No"])
    
    submit_button = st.form_submit_button(label="Predict Loan Approval")

# Process form submission
if submit_button:
    with st.spinner("Predicting loan approval..."):
        # Create feature DataFrame with expected one-hot encoded columns
        education_Not_Graduate = 1 if education == "Not Graduate" else 0
        self_employed_Yes = 1 if self_employed == "Yes" else 0

        features_dict = {
            'no_of_dependents': [no_of_dependents],
            'income_annum': [income_annum],
            'loan_amount': [loan_amount],
            'loan_term': [loan_term],
            'cibil_score': [cibil_score],
            'residential_assets_value': [residential_assets_value],
            'commercial_assets_value': [commercial_assets_value],
            'luxury_assets_value': [luxury_assets_value],
            'bank_asset_value': [bank_asset_value],
            'education_Not Graduate': [education_Not_Graduate],
            'self_employed_Yes': [self_employed_Yes]
        }
        features = pd.DataFrame(features_dict, columns=expected_features)
        
        try:
            # Scale features if model is not XGBoost
            if isinstance(model, xgb.XGBClassifier):
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0][1]
            else:
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0][1]
            
            # Display results
            result = 'Approved' if prediction == 1 else 'Rejected'
            st.markdown("Prediction Complete!")
            st.header("Prediction Results")
            if(result == 'Approved'):
                st.success(f"**Loan Status:** `{result}`")
            else:
                st.error(f"**Loan Status:** `{result}`")

            st.markdown(f"**Approval Probability:** `{probability:.2%}`")
            
            # Display input summary
            st.subheader("Input Summary")
            input_summary = {
                'No of Dependents': no_of_dependents,
                'Annual Income': income_annum,
                'Loan Amount': loan_amount,
                'Loan Term (months)': loan_term,
                'CIBIL Score': cibil_score,
                'Residential Assets Value': residential_assets_value,
                'Commercial Assets Value': commercial_assets_value,
                'Luxury Assets Value': luxury_assets_value,
                'Bank Asset Value': bank_asset_value,
                'Education': education,
                'Self Employed': self_employed
            }
            for label, value in input_summary.items():
                st.write(f"- {label}: {value}")
            
            # Log to CSV (use /tmp for cloud, fallback to script_dir locally)
            log_dir = '/tmp' if os.path.exists('/tmp') else script_dir
            log_file = os.path.join(log_dir, 'predictions_log.csv')
            log_data = {
                'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'No of Dependents': [no_of_dependents],
                'Income Annually': [income_annum],
                'Loan Amount': [loan_amount],
                'Loan Term': [loan_term],
                'CIBIL Score': [cibil_score],
                'Residential Assets Value': [residential_assets_value],
                'Commercial Assets Value': [commercial_assets_value],
                'Luxury Assets Value': [luxury_assets_value],
                'Bank Asset Value': [bank_asset_value],
                'Education': [education],
                'Self Employed': [self_employed],
                'Prediction': [result],
                'Approval Probability': [probability]
            }
            log_df = pd.DataFrame(log_data)
            
            if not os.path.exists(log_file):
                log_df.to_csv(log_file, index=False)
            else:
                log_df.to_csv(log_file, mode='a', header=False, index=False)
            
            st.success("Prediction logged to 'predictions_log.csv'.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            # Print detailed feature mismatch for debugging
            if hasattr(model, 'feature_names_in_'):
                st.write("Feature names expected by model:", model.feature_names_in_.tolist())
            else:
                st.write("Feature names expected by model: Not available (check training data)")
            st.write("Feature names provided:", features.columns.tolist())
            st.stop()