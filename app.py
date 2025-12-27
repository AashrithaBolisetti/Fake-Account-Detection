import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(page_title="Cyber-Audit: Fake Account Detector", layout="wide")

# Load pre-trained model data
@st.cache_resource
def load_model_data():
    try:
        # Ensure trained_model.pkl is uploaded to your GitHub repo
        data = joblib.load('trained_model.pkl')
        return data['classifier'], data['tfidf'], data['numerical_cols']
    except FileNotFoundError:
        st.error("Missing File: 'trained_model.pkl' not found in GitHub. Please upload it.")
        return None, None, None

model, tfidf, num_cols = load_model_data()

st.title("🛡️ Cyber-Incident Account Auditor")

if model:
    st.header("1. Individual Profile Audit")
    content = st.text_area("Profile Content/Bio")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        reposts = st.number_input("Reposts", value=0)
    with col2:
        comments = st.number_input("Comments", value=0)
    with col3:
        likes = st.number_input("Likes", value=0)
    
    hour = st.slider("Post Hour (0-23)", 0, 23, 12)

    if st.button("Check Account"):
        # Process and Predict
        text_v = tfidf.transform([content if content else " "]).toarray()
        input_data = np.hstack([[reposts, comments, likes, hour], text_v[0]])
        prediction = model.predict([input_data])[0]
        
        if prediction == 1:
            st.error("🚩 FLAG: Potential Fake Account Detected")
        else:
            st.success("✅ AUTHENTIC: Likely Legitimate Account")
