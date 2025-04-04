import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xgboost as xgb

# Load the dataset
data = pd.read_csv('Processed_data_latest.csv', encoding='latin1')

# Dashboard Page
def dashboard():
    try:
        st.image('logo.jpeg', use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ Logo image not found. Please upload 'logo.jpeg' to your project directory.")

    st.subheader("💡 Abstract")
    st.write("""
    Islamic Family and Social Services Association (IFSSA) provides food hampers to individuals and families in need.
    This app predicts client retention and supports proactive outreach and planning using machine learning.
    """)

    st.subheader("👨‍💼 Project Purpose")
    st.write("""
    Our goal is to forecast which clients are likely to return for food hamper pickups. This improves outreach, minimizes waste, and supports better planning.
    """)

# EDA Page
def exploratory_data_analysis():
    st.title("📊 IFSSA Client Data Analysis")
    st.markdown("""
    <iframe width="600" height="450" src="https://lookerstudio.google.com/reporting/f21f2db2-6992-4e62-89e1-1d7ac1b699ac" frameborder="0" style="border:0" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>
    """, unsafe_allow_html=True)

# Prediction Page Template
def predict_page(month_label, model_file):
    st.title(f"🔮 Predict Client Return for {month_label} Month")
    st.write("Enter client details below:")

    distance = st.slider("Distance from IFSSA (km)", 0, 50, 10)
    pickup_day = st.selectbox("Pickup Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    dependents_qty = st.slider("Number of Dependents", 0, 10, 1)
    age_group = st.selectbox("Age Group", ['0-18', '19-30', '31-45', '46-65', '65+'])
    scheduled_month = st.selectbox("Scheduled Month", list(range(1, 13)))
    scheduled_weekday = st.selectbox("Scheduled Weekday", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    if st.button("Predict"):
        try:
            model = joblib.load(model_file)
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
            family_name = data['family_name'].dropna().sample(1).values[0]


            # Probability bar
            fig = go.Figure(go.Bar(
                x=[proba[0], proba[1]],
                y=['Not Return', 'Will Return'],
                orientation='h',
                marker=dict(color=['crimson', 'green'])
            ))
            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Probability",
                yaxis_title="Class",
                xaxis=dict(range=[0, 1]),
                height=300
            )
            st.plotly_chart(fig)

            # Suggestions
            if prediction == 1:
                st.success("✅ The client is likely to return.")
                st.info(f"Suggestion for {family_name}: Continue routine outreach and record future visits.")
            else:
                st.error("❌ The client is unlikely to return.")
                st.warning(f"Suggestion for {family_name}: Consider a follow-up call, support check-in, or sending a reminder message.")

            # SHAP
            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)
            st.subheader("🧠 SHAP Explanation")
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], max_display=6, show=False)
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Thank You Page
def thank_you_page():
    st.title("🙏 Thank You")
    st.write("We appreciate your interest in our project. For more information about IFSSA:")
    st.markdown("[Visit IFSSA Official Website](https://albertamentors.ca/islamic-family-social-services-association-ifssa/)")
    try:
        st.image("IFFSA_Family_2.png", caption="Islamic Family & Social Services Association")
    except:
        st.warning("IFSSA logo image not found.")

# Main App Logic
def main():
    st.sidebar.title("IFSSA Client Retention Prediction")
    app_page = st.sidebar.radio("Select a Page", [
        "Dashboard",
        "EDA",
        "Predicting Return in 1 Month",
        "Predicting Return in 3 Month",
        "Predicting Return in 6 Month",
        "Thank You"
    ])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "Exploratory Data Analysis":
        exploratory_data_analysis()
    elif app_page == "Predicting Return in 1 Month":
        predict_page("1", "1m_XGBoost_smote.pkl")
    elif app_page == "Predicting Return in 3 Month":
        predict_page("3", "3m_XGBoost_smote.pkl")
    elif app_page == "Predicting Return in 6 Month":
        predict_page("6", "6m_XGBoost_smote.pkl")
    elif app_page == "Thank You":
        thank_you_page()

if __name__ == "__main__":
    main()
