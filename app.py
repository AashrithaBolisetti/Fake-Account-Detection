import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Cyber-Audit", layout="wide")

@st.cache_resource
def load_assets():
    # This must match the dictionary keys in the Colab script
    data = joblib.load('trained_model.pkl')
    return data['model'], data['tfidf'], data['features']

try:
    model, tfidf, num_cols = load_assets()
except Exception as e:
    st.error(f"Critical Error loading model: {e}")
    st.stop()

st.title("🛡️ Cyber-Incident Account Auditor")

# UI for Inputs
content = st.text_area("Profile Bio/Content")
col1, col2, col3 = st.columns(3)
with col1: r = st.number_input("Reposts", 0)
with col2: c = st.number_input("Comments", 0)
with col3: l = st.number_input("Likes", 0)
h = st.slider("Post Hour", 0, 23, 12)

if st.button("Audit Profile"):
    # 1. Vectorize text
    text_v = tfidf.transform([content if content else " "]).toarray()
    # 2. Combine exactly like training: [Reposts, Comments, Likes, Post_Hour, ...text...]
    input_data = np.hstack([[r, c, l, h], text_v[0]])
    
    try:
        prediction = model.predict([input_data])[0]
        if prediction == 1:
            st.error("🚩 FLAG: Potential Fake Account")
        else:
            st.success("✅ AUTHENTIC: Legitimate Account")
    except ValueError:
        st.error(f"**Feature Mismatch!**")
        st.info(f"Model expects: {model.n_features_in_} features. App sent: {len(input_data)} features.")
        st.warning("Please re-run the Colab script and upload the NEW trained_model.pkl.")
