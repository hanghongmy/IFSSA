<p align="center" draggable="false">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR8HNB-ex4xb4H3-PXRcywP5zKC_3U8VzQTPA&usqp=CAU" 
       width="200px"
       height="auto"/>
</p>

# <h1 align="center" id="heading"># IFSSA Client Retention Prediction - Team Members: My Hang Hong, Kiranpreet Kaur, Gurraj Kaur</h1>

---

### PROJECT TITLE: Predicting Client Return for IFSSA Food Hamper Program

Welcome to the repository for our Capstone project at NorQuest College. This project was developed in collaboration with the **Islamic Family and Social Services Association (IFSSA)** to improve food program planning and resource allocation.

This initiative aims to **predict the likelihood of clients returning for food hamper services**. With a predictive model, IFSSA can better support at-risk individuals, reduce food waste, and improve operational efficiency.

---

### 🧩 Problem Statement

IFSSA faces challenges in predicting which clients will return after receiving a food hamper. This uncertainty makes it difficult to forecast demand, plan inventory, and deliver targeted outreach to clients who may be food insecure but are not returning.

---

### 💡 Solution

We developed a **machine learning model** to predict the likelihood of client return based on:
- 🧾 Hamper pickup history  
- 👤 Demographic features  
- 🎟️ Event participation  
- 📊 Service usage patterns

**Tech Used:**
- **Python** for data processing and modeling  
- **Pandas & NumPy** for data manipulation  
- **Scikit-learn, XGBoost** for machine learning modeling  
- **Imbalanced-learn (SMOTE)** for handling class imbalance  
- **SHAP** for model explainability  
- **Matplotlib & Plotly** for visualization  
- **Streamlit** for interactive web app development  
- **Git & GitHub** for version control and collaboration 

The best-performing model achieved over **90% accuracy**, helping identify at-risk clients for targeted engagement.

---

📂 Repository Structure

The repository contains the following files and folders:

```text
📁 data/
   ├── client_data.csv              # Anonymized demographic and service usage data
   └── hamper_data.csv              # Pickup history and hamper service details
   └── Processed_data_latest.csv    # The merged and cleaned data with new features

📁 notebooks/
   └── Data cleaning, EDA, feature engineering, and model training

📄 app.py                     # Streamlit app for live prediction
📄 README.md                  # Project overview (this file)
📄 requirements.txt           # Python dependencies
📄 logs.txt
📄 package-lock.json
📄 logo.jpeg
📄 IFFSA_Family_2.png
📄 1m_XGBoost_smote.pkl        # Best trained models required for deploying application
📄 3m_XGBoost_smote.pk
📄 6m_XGBoost_smote.pk


---
```

👥 Team Members

Our team consists of the following members:

- [My Hang Hong (Helen)] 
- [Kiranpreet Kaur]
- [Gurraj Kaur] 

---

📫 Get in Touch

Have questions or want to collaborate? Reach out to us via email:
**Notion:** https://gurrajkaur13-1742574967747.atlassian.net/wiki/external/NGI3MDQ4NTY3ZWVmNDA4NmE3YmFmYjI5MmZhNGQxYTQ
**Streamlit App:** https://9vbk24bvsyjc9uxg3s6ppq.streamlit.app

- **My Hang Hong:** hhong@norquest.ca  
- **Kiranpreet Kaur:** klnu72norquest.ca
- **Gurraj Kaur:** ggill533@norquest.ca


🎉 Closing Note

> “Looking forward to making a meaningful impact through data-driven solutions. Thank you for visiting our project!”

