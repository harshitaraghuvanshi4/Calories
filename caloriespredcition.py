
import streamlit as st
import pickle
import numpy as np

# Load the trained XGBoost model and scaler
with open('Calories.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('schr.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Title of the web app
st.title("Calories Burned Prediction App")

# Input fields
st.write("### Enter the required inputs:")
gender = st.radio("Gender", options=["Male", "Female"])
age = st.number_input("Age (years)", min_value=1, max_value=120, value=25, step=1)
height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.1)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1)
duration = st.number_input("Duration of Exercise (minutes)", min_value=0.0, max_value=300.0, value=30.0, step=1.0)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=220.0, value=75.0, step=0.1)
body_temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5, step=0.1)

# Convert gender to numeric encoding
gender_encoded = 0 if gender == "Male" else 1

# Prediction logic
if st.button("Predict Calories Burned"):
    try:
        # Create feature vector
        input_data = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])

        # Normalize the input data using the scaler
        input_data_scaled = scaler.transform(input_data)

        # Predict calories burned
        prediction = model.predict(input_data_scaled)

        # Display the result
        st.success(f"Predicted Calories Burned: {prediction[0]:,.2f} kcal")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    