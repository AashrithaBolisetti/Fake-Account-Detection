import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Cyber-Audit", layout="wide")

@st.cache_resource
def load_assets():
    # Load and check keys
    data = joblib.load('trained_model.pkl')
    return data['model'], data['tfidf'], data['num_cols']

try:
    model, tfidf, num_cols = load_assets()
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

st.title("🛡️ Cyber-Incident Account Auditor")

# Individual Audit
with st.expander("1. Individual Profile Audit", expanded=True):
    content = st.text_area("Profile Bio/Content")
    c1, c2, c3 = st.columns(3)
    with c1: r = st.number_input("Reposts", 0)
    with c2: c = st.number_input("Comments", 0)
    with c3: l = st.number_input("Likes", 0)
    h = st.slider("Post Hour", 0, 23, 12)

    if st.button("Check Account"):
        # 1. Transform text (results in 5000 features)
        text_v = tfidf.transform([content if content else " "]).toarray()
        
        # 2. Add 4 numerical features (Total: 5004)
        input_data = np.hstack([[r, c, l, h], text_v[0]])
        
        try:
            prediction = model.predict([input_data])[0]
            if prediction == 1:
                st.error("🚩 FLAG: Potential Fake Account Detected")
            else:
                st.success("✅ AUTHENTIC: Likely Legitimate Account")
        except ValueError as ve:
            st.error(f"Mismatch: App sent {len(input_data)} features, but model expects {model.n_features_in_}.")
            st.info("Please ensure you uploaded the NEW trained_model.pkl from Colab.")
