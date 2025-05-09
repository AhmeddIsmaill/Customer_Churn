
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from monitor_and_log import log_prediction

mlflow.set_tracking_uri("http://127.0.0.1:5000")
run_id = "dec9555ed1df4a3c8c8ec503e639736d"  
model_uri = f"mlartifacts/0/{run_id}/artifacts/model"
model = mlflow.sklearn.load_model(model_uri)

def preprocess_data(df):
    expected_columns = [
        'gender',
        'SeniorCitizen',
        'Partner',
        'Dependents',
        'PhoneService',
        'PaperlessBilling',
        'MonthlyCharges',
        'InternetService_Fiber optic',
        'InternetService_No',
        'Contract_One year',
        'Contract_Two year',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check',
        'TotalCharges_boxcox',
        'tenure_cbrt',
        'TotalServices',
        'HasPremiumSupport',
        'AvgMonthlySpend'
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    return df[expected_columns]

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("Customer Churn Prediction Dashboard")

st.subheader("Upload CSV Data for Churn Prediction")
uploaded_file = st.file_uploader("Upload your customer data file", type="csv")

if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview", raw_data.head())

    processed_data = preprocess_data(raw_data)
    preds = model.predict(processed_data)
    probs = model.predict_proba(processed_data)[:, 1]

    raw_data["Churn Prediction"] = preds
    raw_data["Churn Probability"] = probs
    st.write("Prediction Results", raw_data)

    st.subheader("Churn Insights")
    st.bar_chart(raw_data["Churn Prediction"].value_counts())

    if 'gender' in raw_data.columns:
        fig, ax = plt.subplots()
        sns.barplot(x='gender', y='Churn Prediction', data=raw_data, ax=ax)
        ax.set_title("Churn by Gender")
        st.pyplot(fig)

    if 'Contract_One year' in raw_data.columns:
        fig, ax = plt.subplots()
        contract_churn = raw_data.groupby(['Contract_One year', 'Contract_Two year'])["Churn Prediction"].mean().reset_index()
        sns.heatmap(contract_churn.pivot("Contract_One year", "Contract_Two year", "Churn Prediction"), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Churn by Contract Type")
        st.pyplot(fig)

st.sidebar.header("🔧 Manual Input")
manual_input = {
    'gender': st.sidebar.selectbox("Gender (0=Female, 1=Male)", [0, 1]),
    'SeniorCitizen': st.sidebar.selectbox("Senior Citizen", [0, 1]),
    'Partner': st.sidebar.selectbox("Partner", [0, 1]),
    'Dependents': st.sidebar.selectbox("Dependents", [0, 1]),
    'PhoneService': st.sidebar.selectbox("Phone Service", [0, 1]),
    'PaperlessBilling': st.sidebar.selectbox("Paperless Billing", [0, 1]),
    'MonthlyCharges': st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0),
    'InternetService_Fiber optic': st.sidebar.selectbox("Fiber Optic Internet", [0, 1]),
    'InternetService_No': st.sidebar.selectbox("No Internet Service", [0, 1]),
    'Contract_One year': st.sidebar.selectbox("One Year Contract", [0, 1]),
    'Contract_Two year': st.sidebar.selectbox("Two Year Contract", [0, 1]),
    'PaymentMethod_Credit card (automatic)': st.sidebar.selectbox("Payment: Credit Card (Auto)", [0, 1]),
    'PaymentMethod_Electronic check': st.sidebar.selectbox("Payment: Electronic Check", [0, 1]),
    'PaymentMethod_Mailed check': st.sidebar.selectbox("Payment: Mailed Check", [0, 1]),
    'TotalCharges_boxcox': st.sidebar.slider("BoxCox Total Charges", 0.0, 10.0, 4.5),
    'tenure_cbrt': st.sidebar.slider("Cube Root of Tenure", 0.0, 10.0, 2.5),
    'TotalServices': st.sidebar.slider("Number of Services", 0, 10, 5),
    'HasPremiumSupport': st.sidebar.selectbox("Has Premium Support", [0, 1]),
    'AvgMonthlySpend': st.sidebar.slider("Avg Monthly Spend", 0.0, 150.0, 60.0),
}

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([manual_input])
    input_df = preprocess_data(input_df)
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    log_prediction(manual_input, prediction=pred, probability=prob)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Prediction Result")
    st.sidebar.write(f"Prediction: **{'Churn' if pred == 1 else 'No Churn'}**")
    st.sidebar.write(f"Churn Probability: **{prob:.2%}**")
