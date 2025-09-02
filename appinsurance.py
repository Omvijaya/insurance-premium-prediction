import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline model
model = joblib.load(r"D:\GUVI AIML\Projects\Insurance Premium Prediction Model\premium_predictor.pkl")

st.title("💡 SmartPremium: Insurance Premium Prediction")

# Input form
age = st.slider("Age", 18, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
income = st.number_input("Annual Income", 1000, 200000, 50000)
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
dependents = st.number_input("Number of Dependents", 0, 10, 0)
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
health = st.slider("Health Score", 0.0, 100.0, 50.0)
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])

# ✅ Default values for missing columns
customer_feedback = "Average"      # default
vehicle_age = 5                    # default years
property_type = "House"            # default
smoking_status = "No"              # default
credit_score = 600                 # default score
previous_claims = 0                # default count
exercise_frequency = "Weekly"      # default
insurance_duration = 1             # default years

# Predict button
if st.button("Predict Premium"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Annual Income": income,
        "Marital Status": marital,
        "Number of Dependents": dependents,
        "Education Level": education,
        "Occupation": occupation,
        "Health Score": health,
        "Location": location,
        "Policy Type": policy_type,
        "Customer Feedback": customer_feedback,
        "Vehicle Age": vehicle_age,
        "Property Type": property_type,
        "Smoking Status": smoking_status,
        "Credit Score": credit_score,
        "Previous Claims": previous_claims,
        "Exercise Frequency": exercise_frequency,
        "Insurance Duration": insurance_duration
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Premium Amount: {prediction:,.2f}")
