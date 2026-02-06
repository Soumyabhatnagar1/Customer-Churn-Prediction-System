import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load artifacts
# -----------------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("📉 Customer Churn Prediction")

st.write("Enter customer details to predict churn risk")

# -----------------------------
# User Input
# -----------------------------
input_data = {
    "Customer_Age": st.number_input("Customer Age", 18, 100, 40),
    "Dependent_count": st.number_input("Dependent Count", 0, 5, 1),
    "Months_on_book": st.number_input("Months on Book", 0, 60, 24),
    "Total_Relationship_Count": st.number_input("Total Relationship Count", 1, 6, 3),
    "Months_Inactive_12_mon": st.number_input("Inactive Months (Last 12)", 0, 12, 2),
    "Contacts_Count_12_mon": st.number_input("Contacts Count (Last 12)", 0, 10, 2),
    "Credit_Limit": st.number_input("Credit Limit", 1000, 50000, 10000),
    "Total_Revolving_Bal": st.number_input("Total Revolving Balance", 0, 30000, 2000),
    "Avg_Open_To_Buy": st.number_input("Avg Open To Buy", 0, 50000, 8000),
    "Total_Amt_Chng_Q4_Q1": st.number_input("Amount Change Q4/Q1", 0.0, 5.0, 1.2),
    "Total_Trans_Amt": st.number_input("Total Transaction Amount", 0, 100000, 12000),
    "Total_Trans_Ct": st.number_input("Total Transaction Count", 0, 200, 40),
    "Total_Ct_Chng_Q4_Q1": st.number_input("Transaction Count Change", 0.0, 5.0, 1.1),
    "Avg_Utilization_Ratio": st.number_input("Avg Utilization Ratio", 0.0, 1.0, 0.3),
    "Gender": st.selectbox("Gender", ["M", "F"]),
    "Education_Level": st.selectbox("Education Level", [
        "Graduate", "High School", "Post-Graduate", "Doctorate", "Unknown"
    ]),
    "Marital_Status": st.selectbox("Marital Status", [
        "Married", "Single", "Divorced", "Unknown"
    ]),
    "Income_Category": st.selectbox("Income Category", [
        "Less than $40K", "$40K - $60K", "$60K - $80K",
        "$80K - $120K", "$120K +", "Unknown"
    ]),
    "Card_Category": st.selectbox("Card Category", [
        "Blue", "Silver", "Gold", "Platinum"
    ])
}

df_input = pd.DataFrame([input_data])

# -----------------------------
# Preprocessing (SAFE)
# -----------------------------
cat_cols = encoder.feature_names_in_
num_cols = scaler.feature_names_in_

df_input_cat = encoder.transform(df_input[cat_cols])
df_input_num = scaler.transform(df_input[num_cols])

X_final = np.hstack([df_input_num, df_input_cat])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):
    prob = model.predict_proba(X_final)[0][1]

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{prob:.2%}")

    if prob < 0.3:
        st.success("Low Churn Risk")
    elif prob < 0.6:
        st.warning("Medium Churn Risk")
    else:
        st.error("High Churn Risk")
