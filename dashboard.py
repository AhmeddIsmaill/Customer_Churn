import streamlit as st
import pandas as pd
import plotly.express as px

# === CONFIG ===
st.set_page_config(page_title="Customer Churn Analysis", layout="wide")

# === LOAD DATA ===
@st.cache_data
def load_data():
    return pd.read_csv("preprocessed_churn_data.csv")

df = load_data()

# === SIDEBAR FILTERS ===
st.sidebar.title("🔍 Filters")
gender = st.sidebar.multiselect("Gender", options=[0, 1], default=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
senior = st.sidebar.multiselect("Senior Citizen", options=[0, 1], default=[0, 1])
partner = st.sidebar.multiselect("Has Partner", options=[0, 1], default=[0, 1])
dependent = st.sidebar.multiselect("Has Dependents", options=[0, 1], default=[0, 1])

filtered_df = df[
    df["gender"].isin(gender) &
    df["SeniorCitizen"].isin(senior) &
    df["Partner"].isin(partner) &
    df["Dependents"].isin(dependent)
]

# === METRICS ===
st.title("📊 Customer Churn Dashboard with Insights and Recommendations")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", len(filtered_df))
with col2:
    churned = filtered_df[filtered_df["Churn"] == 1]
    st.metric("Churned Customers", len(churned))
with col3:
    churn_rate = len(churned) / len(filtered_df) if len(filtered_df) else 0
    st.metric("Churn Rate", f"{churn_rate:.2%}")

st.markdown("---")

# === CHURN DISTRIBUTION ===
st.subheader("📉 Churn Distribution")
fig_pie = px.pie(filtered_df, names="Churn", title="Churn vs No Churn", hole=0.4,
                 color_discrete_map={0: "lightblue", 1: "salmon"},
                 labels={"Churn": "Churn Status"})
st.plotly_chart(fig_pie, use_container_width=True)

# === GENDER & CONTRACT ANALYSIS ===
col4, col5 = st.columns(2)
with col4:
    st.subheader("👥 Churn by Gender")
    fig_gender = px.histogram(filtered_df, x="gender", color="Churn", barmode="group",
                              labels={"gender": "Gender"}, 
                              category_orders={"gender": [0, 1]},
                              color_discrete_sequence=["#2ca02c", "#d62728"])
    fig_gender.update_xaxes(ticktext=["Female", "Male"], tickvals=[0, 1])
    st.plotly_chart(fig_gender, use_container_width=True)

with col5:
    st.subheader("📄 Churn by Contract Type")
    contract_map = {
        "Month-to-Month": 0,
        "One Year": 1,
        "Two Year": 2
    }
    contract_data = filtered_df.copy()
    contract_data["Contract"] = contract_data[["Contract_One year", "Contract_Two year"]].apply(
        lambda row: 2 if row["Contract_Two year"] == 1 else 1 if row["Contract_One year"] == 1 else 0, axis=1)
    fig_contract = px.histogram(contract_data, x="Contract", color="Churn", barmode="group",
                                labels={"Contract": "Contract Type"}, 
                                category_orders={"Contract": [0, 1, 2]},
                                color_discrete_sequence=["#1f77b4", "#ff7f0e"])
    fig_contract.update_xaxes(ticktext=list(contract_map.keys()), tickvals=list(contract_map.values()))
    st.plotly_chart(fig_contract, use_container_width=True)

# === AVERAGE CHURN PROBABILITY GAUGE ===
import plotly.graph_objects as go

avg_prob = filtered_df["Churn"].mean() * 100
st.subheader("📟 Estimated Average Churn Probability")
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=avg_prob,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Avg. Churn Probability"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 30], 'color': "lightgreen"},
            {'range': [30, 70], 'color': "yellow"},
            {'range': [70, 100], 'color': "red"}
        ]
    }
))
st.plotly_chart(fig_gauge, use_container_width=True)

# === RECOMMENDATIONS TABLE ===
st.subheader("📌 Customer Recommendations")
rec_df = filtered_df.copy()
rec_df["Recommendation"] = rec_df.apply(lambda row: "Offer Long-Term Plan" if row["Contract_Two year"] == 0 and row["Churn"] == 1 else "Retain", axis=1)
st.dataframe(rec_df[["Churn", "MonthlyCharges", "Contract_Two year", "Recommendation"]].head(20))
