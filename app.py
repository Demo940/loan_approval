import pickle
from pathlib import Path

import numpy as np
import streamlit as st


# Defining the paths for the trained model and scaler files.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "loan_model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"


# Loading the trained machine learning model from "loan_model.pkl" using pickle.
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)



# Loading the scaler from "scaler.pkl".
with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


# Building a simple UI using Streamlit.
st.title("Loan Approval Prediction")


# Adding input fields for the user.
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0.0, step=1.0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, step=1.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0, step=1.0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0.0, step=1.0)
credit_history = st.selectbox("Credit History", [0, 1])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])


# Converting user inputs into numerical format.
gender_value = 1 if gender == "Male" else 0
married_value = 1 if married == "Yes" else 0
dependents_value = int(dependents)
education_value = 0 if education == "Graduate" else 1
self_employed_value = 1 if self_employed == "Yes" else 0
credit_history_value = int(credit_history)

if property_area == "Urban":
    property_area_value = 2
elif property_area == "Semiurban":
    property_area_value = 1
else:
    property_area_value = 0


# Using a button to make prediction.
if st.button("Predict"):
    # Combining inputs into a NumPy array and reshaping it.
    input_data = np.array(
        [
            gender_value,
            married_value,
            dependents_value,
            education_value,
            self_employed_value,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history_value,
            property_area_value,
        ]
    ).reshape(1, -1)

    # Applying scaler transformation to input data.
    input_data = scaler.transform(input_data)

    # Using the model to make prediction.
    prediction = model.predict(input_data)

    # Displaying the result clearly.
    if int(prediction[0]) == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")
