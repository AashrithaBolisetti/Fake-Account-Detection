import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Cyber-Audit", layout="wide")

@st.cache_resource
def load_assets():
    # Load the fresh 5004-feature model
    data = joblib.load('trained_model.pkl')
    return data['model'], data['tfidf'], data['features']

try:
    model, tfidf, num_cols = load_assets()
except Exception as e:
    st.error(f"Critical Error: {e}. Ensure 'trained_model.pkl' is uploaded.")
    st.stop()

st.title("🛡️ Cyber-Incident Account Auditor")

with st.expander("1. Individual Profile Audit", expanded=True):
    content = st.text_area("Profile Bio/Content", key="bio")
    c1, c2, c3 = st.columns(3)
    with c1: r = st.number_input("Reposts", 0)
    with c2: c = st.number_input("Comments", 0)
    with c3: l = st.number_input("Likes", 0)
    h = st.slider("Post Hour", 0, 23, 12)

    if st.button("Audit Account"):
        # 1. Transform text -> 5000 features
        text_v = tfidf.transform([content if content else " "]).toarray()
        
        # 2. Add 4 numbers -> Total 5004 features
        input_data = np.hstack([[r, c, l, h], text_v[0]])
        
        try:
            prediction = model.predict([input_data])[0]
            if prediction == 1:
                st.error("🚩 FLAG: Potential Fake Account")
            else:
                st.success("✅ AUTHENTIC: Legitimate Account")
        except ValueError:
            st.error(f"Feature Mismatch: App sent {len(input_data)} features, but model expects {model.n_features_in_}.")
            st.info("Ensure you uploaded the NEW trained_model.pkl from Colab.")
