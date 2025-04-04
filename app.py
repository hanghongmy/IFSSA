import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xgboost as xgb
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the dataset
data = pd.read_csv('Processed_data_latest.csv', encoding='latin1')
docs = pd.read_csv('processed_chunks.csv')

# Embed documents
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(docs['chunk'].tolist(), convert_to_numpy=True)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Dashboard Page
def dashboard():
    st.markdown("""
    <div style='text-align: center;'>
        <img src='logo.jpeg' width='300'/>
        <h1 style='color: #2E8B57;'>Welcome to IFSSA Client Retention Predictor</h1>
        <p style='font-size:18px;'>Empowering community service with data-driven decisions</p>
    </div>
    """, unsafe_allow_html=True)

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
    <iframe width="600" height="450" src="https://lookerstudio.google.com/embed/reporting/f21f2db2-6992-4e62-89e1-1d7ac1b699ac/page/0NzEF" frameborder="0" style="border:0" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>""", unsafe_allow_html=True)

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

            fig = go.Figure(go.Bar(
                x=[proba[0], proba[1]],
                y=['Not Return', 'Will Return'],
                orientation='h',
                marker=dict(color=['crimson', 'green'])
            ))
            fig.update_layout(title="Prediction Probabilities", xaxis_title="Probability", yaxis_title="Class", xaxis=dict(range=[0, 1]), height=300)
            st.plotly_chart(fig)

            if prediction == 1:
                st.success("✅ The client is likely to return.")
                st.info(f"Suggestion for {family_name}: Continue routine outreach and record future visits.")
            else:
                st.error("❌ The client is unlikely to return.")
                st.warning(f"Suggestion for {family_name}: Consider a follow-up call, support check-in, or sending a reminder message.")

            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)
            st.subheader("🧠 SHAP Explanation")
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], max_display=6, show=False)
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Chatbot Page
def chatbot_page():
    st.title("🤖 IFSSA Chatbot & Predictor")
    mode = st.radio("What would you like to do?", ["Ask a general question", "Make a prediction"], index=0)

    if mode == "Ask a general question":
        user_input = st.text_input("Ask something about IFSSA, food hampers, how to register, etc.")
        if st.button("Ask") and user_input:
            try:
                query_embedding = embedder.encode([user_input])
                D, I = index.search(np.array(query_embedding), k=3)
                results = docs.iloc[I[0]]['chunk'].tolist()
                for r in results:
                    st.write("-", r)
            except Exception as e:
                st.error(f"Failed to get response: {e}")

    elif mode == "Make a prediction":
        user_input = st.text_area("Ask here (e.g. A 42-year-old with 2 dependents picks up Friday, lives 12 km away):")

        if st.button("🔍 Predict") and user_input:
            try:
                features = {
                    'distance_km': 10,
                    'pickup_day': 4,
                    'dependents_qty': 2,
                    'age_group_encoded': 2,
                    'scheduled_month': 6,
                    'scheduled_weekday_encoded': 4
                }

                dist = re.search(r'(\d+)\s*km', user_input)
                dep = re.search(r'(\d+)\s*dependents?', user_input)
                age = re.search(r'(\d+)[\s-]*year[\s-]*old', user_input)

                for d, v in {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}.items():
                    if d.lower() in user_input.lower():
                        features['pickup_day'] = v
                        features['scheduled_weekday_encoded'] = v
                        break

                if dist:
                    features['distance_km'] = int(dist.group(1))
                if dep:
                    features['dependents_qty'] = int(dep.group(1))
                if age:
                    a = int(age.group(1))
                    features['age_group_encoded'] = 0 if a <= 18 else 1 if a <= 30 else 2 if a <= 45 else 3 if a <= 65 else 4

                input_df = pd.DataFrame([features])

                model_1 = joblib.load("1m_XGBoost_smote.pkl")
                model_3 = joblib.load("3m_XGBoost_smote.pkl")
                model_6 = joblib.load("6m_XGBoost_smote.pkl")

                pred_1 = model_1.predict(input_df)[0]
                pred_3 = model_3.predict(input_df)[0]
                pred_6 = model_6.predict(input_df)[0]

                def yesno(p): return "✅ Likely to return" if p == 1 else "❌ Unlikely to return"

                st.markdown("### 📊 Prediction Results")
                st.markdown(f"- 1 Month: {yesno(pred_1)}")
                st.markdown(f"- 3 Months: {yesno(pred_3)}")
                st.markdown(f"- 6 Months: {yesno(pred_6)}")

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
        "Exploratory Data Analysis",
        "Predicting Return in 1 Month",
        "Predicting Return in 3 Month",
        "Predicting Return in 6 Month",
        "Chat with Prediction Bot",
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
    elif app_page == "Chat with Prediction Bot":
        chatbot_page()
    elif app_page == "Thank You":
        thank_you_page()

if __name__ == "__main__":
    main()
