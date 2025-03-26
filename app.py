import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from datetime import datetime, timedelta

# Load the dataset
data = pd.read_csv('Processed_data_latest.csv', encoding='latin1')

# Dashboard Page
def dashboard():
    try:
        st.image('logo.jpeg', use_container_width=True)
    except Exception:
        st.warning("Logo image not found.")

    st.subheader("💡 Abstract")
    st.write("""
    Islamic Family and Social Services Association (IFSSA) provides food hampers to individuals and families in need.
    This app predicts client retention and supports proactive outreach and planning using machine learning.
    """)

    st.subheader("👨🏻‍💻 Project Purpose")
    st.write("""
    Our goal is to forecast which clients are likely to return for food hamper pickups. This improves outreach, minimizes waste, and supports better planning.
    """)

# EDA Page
def exploratory_data_analysis():
    st.title("📊 IFSSA Client Data Analysis")

    with st.expander("📌 Dataset Preview"):
        st.write(data.head())

    if 'age_group' in data.columns:
        st.plotly_chart(px.histogram(data, x="age_group", nbins=20, title="Age Distribution"))

    if 'Sex' in data.columns:
        st.plotly_chart(px.histogram(data, x="Sex", title="Gender Distribution"))

    if 'distance_km' in data.columns:
        st.plotly_chart(px.histogram(data, x="distance_km", title="Distance from IFSSA"))

    if 'scheduled_weekday' in data.columns:
        st.plotly_chart(px.histogram(data, x="scheduled_weekday", title="Pickup Day Distribution"))

# Modeling Page
def machine_learning_modeling_1m():
    st.title("🔮 Predict Client Return for 1 month")

    st.write("Enter client details below to predict return likelihood:")

    # Input Fields matching model's required features
    distance = st.slider("Distance from IFSSA (km)", 0, 50, 10)
    pickup_day = st.selectbox("Pickup Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    dependents_qty = st.slider("Number of Dependents", 0, 10, 1)
    age_group = st.selectbox("Age Group", ['0-18', '19-30', '31-45', '46-65', '65+'])
    scheduled_month = st.selectbox("Scheduled Month", list(range(1, 13)))
    scheduled_weekday = st.selectbox("Scheduled Weekday", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    if st.button("Predict"):
        try:
            model = joblib.load("1m_XGBoost_smote.pkl")

            # Encode categorical fields
            pickup_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            age_group_map = {'0-18': 0, '19-30': 1, '31-45': 2, '46-65': 3, '65+': 4}

            input_df = pd.DataFrame([{
                'distance_km': distance,
                'pickup_day': pickup_map[pickup_day],
                'dependents_qty': dependents_qty,
                'age_group_encoded': age_group_map[age_group],
                'scheduled_month': scheduled_month,
                'scheduled_weekday_encoded': pickup_map[scheduled_weekday]
            }])

            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            # Select a random family_name to personalize suggestion
            family_name = data['family_name'].dropna().sample(1).values[0]

            st.success("✅ Will Return" if prediction == 1 else "❌ Not Return")
            st.write(f"Probability → Will Return: {proba[1]:.2f} | Not Return: {proba[0]:.2f}")

            if prediction == 1:
              st.success(f"✅ The client is likely to return.")
              st.info(f"Suggestion for {family_name}: Continue routine outreach and record future visits.")
            else:
              st.error(f"❌ The client is unlikely to return.")
              st.warning(f"Suggestion for {family_name}: Consider a follow-up call, support check-in, or sending a reminder message.")
            # Initialize SHAP explainer (this works for XGBoost)
            explainer = shap.Explainer(model)

            # Calculate SHAP values for this input
            shap_values = explainer(input_df)

            # Show explanation
            st.subheader("🧠 SHAP Explanation")
            st.write("This shows how each input contributed to the model's decision:")

            # Generate SHAP waterfall plot using matplotlib backend
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], max_display=6, show=False)
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Modeling Page
def machine_learning_modeling_3m():
    st.title("🔮 Predict Client Return for 3 month")

    st.write("Enter client details below to predict return likelihood:")

    # Input Fields matching model's required features
    distance = st.slider("Distance from IFSSA (km)", 0, 50, 10)
    pickup_day = st.selectbox("Pickup Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    dependents_qty = st.slider("Number of Dependents", 0, 10, 1)
    age_group = st.selectbox("Age Group", ['0-18', '19-30', '31-45', '46-65', '65+'])
    scheduled_month = st.selectbox("Scheduled Month", list(range(1, 13)))
    scheduled_weekday = st.selectbox("Scheduled Weekday", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    if st.button("Predict"):
        try:
            model = joblib.load("3m_XGBoost_smote.pkl")

            # Encode categorical fields
            pickup_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            age_group_map = {'0-18': 0, '19-30': 1, '31-45': 2, '46-65': 3, '65+': 4}

            input_df = pd.DataFrame([{
                'distance_km': distance,
                'pickup_day': pickup_map[pickup_day],
                'dependents_qty': dependents_qty,
                'age_group_encoded': age_group_map[age_group],
                'scheduled_month': scheduled_month,
                'scheduled_weekday_encoded': pickup_map[scheduled_weekday]
            }])

            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            # Select a random family_name to personalize suggestion
            family_name = data['family_name'].dropna().sample(1).values[0]
            st.success("✅ Will Return" if prediction == 1 else "❌ Not Return")
            st.write(f"Probability → Will Return: {proba[1]:.2f} | Not Return: {proba[0]:.2f}")
            
            if prediction == 1:
              st.success(f"✅ The client is likely to return.")
              st.info(f"Suggestion for {family_name}: Continue routine outreach and record future visits.")
            else:
              st.error(f"❌ The client is unlikely to return.")
              st.warning(f"Suggestion for {family_name}: Consider a follow-up call, support check-in, or sending a reminder message.")
 
            # Initialize SHAP explainer (this works for XGBoost)
            explainer = shap.Explainer(model)

            # Calculate SHAP values for this input
            shap_values = explainer(input_df)

            # Show explanation
            st.subheader("🧠 SHAP Explanation")
            st.write("This shows how each input contributed to the model's decision:")

            # Generate SHAP waterfall plot using matplotlib backend
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], max_display=6, show=False)
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Main App Logic
def main():
    st.sidebar.title("IFSSA_Predicting Client Retention")
    app_page = st.sidebar.radio("Select a Page", [
    "Dashboard",
    "EDA",
    "Predicting Client Return in 1 month",
    "Predicting Client Return in 3 month"
])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "Predicting Client Return in 1 month":
        machine_learning_modeling_1m()
    elif app_page == "Predicting Client Return in 3 month":
        machine_learning_modeling_3m()

if __name__ == "__main__":
    main()
