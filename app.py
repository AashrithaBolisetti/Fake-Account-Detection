import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Cyber-Audit: Fake Account Detector", layout="wide")

@st.cache_resource
def load_model_data():
    return joblib.load('trained_model.pkl')

try:
    data = load_model_data()
    model = data['classifier']
    tfidf = data['tfidf']
    num_cols = data['numerical_cols']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🛡️ Cyber-Incident Account Auditor")

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
    # 1. Vectorize text
    text_v = tfidf.transform([content if content else " "]).toarray()
    
    # 2. Combine features (MUST match the 4+1000 order)
    input_data = np.hstack([[reposts, comments, likes, hour], text_v[0]])
    
    # 3. Predict with safety check
    try:
        prediction = model.predict([input_data])[0]
        if prediction == 1:
            st.error("🚩 FLAG: Potential Fake Account Detected")
        else:
            st.success("✅ AUTHENTIC: Likely Legitimate Account")
    except ValueError as ve:
        st.error(f"Feature Mismatch: The model expects {model.n_features_in_} inputs, but you provided {len(input_data)}.")
