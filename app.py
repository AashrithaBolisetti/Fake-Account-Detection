import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Config
st.set_page_config(page_title="Cyber-Audit: Fake Account Detection", layout="wide")

# Load the pre-trained model and engines from Colab
@st.cache_resource
def load_model_data():
    try:
        # This file must be in the same GitHub folder as app.py
        data = joblib.load('trained_model.pkl')
        return data['classifier'], data['tfidf'], data['numerical_cols']
    except FileNotFoundError:
        st.error("Error: 'trained_model.pkl' not found. Please upload it to your GitHub repository.")
        return None, None, None

model, tfidf_engine, num_cols = load_model_data()

st.title("🛡️ Cyber-Incident Account Auditor")
st.markdown("Automated detection of suspicious social profiles for government reporting.")

if model is not None:
    # --- SECTION 1: MANUAL CHECK ---
    st.header("1. Individual Profile Audit")
    with st.expander("Enter Profile Metadata"):
        col1, col2, col3 = st.columns(3)
        with col1:
            reposts = st.number_input("Reposts Count", min_value=0, value=0)
            comments = st.number_input("Comments Count", min_value=0, value=0)
        with col2:
            likes = st.number_input("Likes Count", min_value=0, value=0)
            hour = st.slider("Post Hour (0-23)", 0, 23, 12)
        with col3:
            content = st.text_area("Profile Bio/Post Content", placeholder="Enter the text here...")

        if st.button("Audit Profile"):
            # Process input using the pre-trained TF-IDF
            text_v = tfidf_engine.transform([content if content else " "]).toarray()
            input_data = np.hstack([[reposts, comments, likes, hour], text_v[0]])
            prediction = model.predict([input_data])[0]
            
            if prediction == 1:
                st.error("🚩 **FLAGGED**: Potential Fake/Automated Account Detected")
            else:
                st.success("✅ **AUTHENTIC**: Likely Legitimate Account")

    st.divider()

    # --- SECTION 2: BATCH AUDIT FOR REPORTING ---
    st.header("2. Batch Incident Reporting")
    st.write("Upload a CSV file of suspicious IDs to generate a report for authorities.")

    uploaded_file = st.file_uploader("Upload suspicious account list (TSV/CSV)", type=["csv", "txt"])

    if uploaded_file:
        # Read the data to audit (Expects tab-separated based on your sample)
        audit_df = pd.read_csv(uploaded_file, sep='\t', names=['ID', 'Timestamp', 'Reposts', 'Comments', 'Likes', 'Content'])
        audit_df['Content'] = audit_df['Content'].fillna('')
        
        # Process for Prediction: Fix Timestamp to avoid warnings
        audit_df['Timestamp'] = pd.to_datetime(audit_df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        audit_df['Post_Hour'] = audit_df['Timestamp'].dt.hour.fillna(-1)
        
        # Vectorize text and combine numerical data
        batch_text = tfidf_engine.transform(audit_df['Content']).toarray()
        batch_num = audit_df[['Reposts', 'Comments', 'Likes', 'Post_Hour']].fillna(0).values
        X_batch = np.hstack((batch_num, batch_text))
        
        # Run Predictions
        preds = model.predict(X_batch)
        audit_df['Audit_Result'] = ["FAKE" if p == 1 else "LEGIT" for p in preds]
        
        # Show Flagged Accounts for Cyber Reporting
        flagged = audit_df[audit_df['Audit_Result'] == "FAKE"]
        st.warning(f"Audited {len(audit_df)} accounts. Found {len(flagged)} suspicious profiles.")
        st.dataframe(flagged)
        
        # Export for Government
        csv_report = flagged.to_csv(index=False)
        st.download_button(
            label="📥 Download Evidence Report for Cyber-Cell",
            data=csv_report,
            file_name="flagged_accounts_report.csv",
            mime="text/csv"
        )

st.sidebar.info("This model is pre-trained on social engagement metrics and multilingual text content.")
