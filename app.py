import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_assets():
    data = joblib.load('trained_model.pkl')
    return data['model'], data['tfidf'], data['num_cols']

model, tfidf, num_cols = load_assets()

st.title("🛡️ Cyber-Incident Account Auditor")

with st.expander("1. Individual Profile Audit", expanded=True):
    content = st.text_area("Profile Bio/Content")
    c1, c2, c3 = st.columns(3)
    with c1: r = st.number_input("Reposts", 0)
    with c2: c = st.number_input("Comments", 0)
    with c3: l = st.number_input("Likes", 0)
    h = st.slider("Post Hour", 0, 23, 12)

    if st.button("Audit Account"):
        # 1. Text to numbers
        text_v = tfidf.transform([content if content else " "]).toarray()
        # 2. Add Length
        con_len = len(content)
        # 3. Combine in order: R, C, L, Hour, Len
        input_data = np.hstack([[r, c, l, h, con_len], text_v[0]])
        
        # 4. Predict
        prediction = model.predict([input_data])[0]
        
        if prediction == 1:
            st.error("🚩 FLAG: Potential Fake Account Detected")
        else:
            st.success("✅ AUTHENTIC: Likely Legitimate Account")
