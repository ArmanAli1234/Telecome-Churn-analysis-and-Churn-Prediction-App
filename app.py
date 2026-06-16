# streamlit app
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# load the model
model=joblib.load("churn_model.pkl")

# set the title of the app
st.title("Customer Churn Predictions App")

# about app
st.write("Fill Customer Detailes and predict customer will be churn or not")

# set sidebar title
st.sidebar.title("Customer Detailes")

# tenure slider
tenure=st.sidebar.slider("tenure",0,72,12)

# make the selectbox of the internet service
internet_service=st.sidebar.selectbox(
    "InternetService",
    ["DSL", "Fiber optic", "No"]
)

# make the selectbox of techsupport
tech_support=st.sidebar.selectbox(
    "techSupport",
    ["Yes","No"]
)

# make the selectbox of the contract
contract=st.sidebar.selectbox(
    "Contract",
    ["Month-to-month","One year","Two year"]
)

# make a selectbox of payment method

payment_method=st.sidebar.selectbox(
    "PaymentMethod",
     ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# make a number input of monthly charges and totalcharges
monthly_charges=st.sidebar.number_input("MonthlyCharges",0,200,50)
total_charges=st.sidebar.number_input("TotalCharges",0,1000,100)

# create a Dataframe
input_data=pd.DataFrame({
    "tenure":[tenure],
    "InternetService":[internet_service],
    "TechSupport":[tech_support],
    "Contract":[contract],
    "PaymentMethod":[payment_method],
    "MonthlyCharges":[monthly_charges],
    "TotalCharges":[total_charges]
})

st.subheader("input Data")
st.write(input_data)

# make a button of predictions
if st.button("Predict"):
    predictions=model.predict(input_data)
    prob=model.predict_proba(input_data)

    st.subheader("Results")


    if predictions[0]==1:
        st.error("The customer is likely to churn")
    else :
        st.success("The Customer is not likely to churn")
    
    st.info(f"Churn Probablity: {prob[0][1]:.2f}")