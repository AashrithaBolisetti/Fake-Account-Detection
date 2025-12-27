import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Page Config
st.set_page_config(page_title="Cyber-Audit: Fake Account Detection", layout="wide")

@st.cache_resource
def train_and_prepare_model():
    # Load your datasets
    # Note: Using the exact filenames and format discussed
    cols = ['ID', 'Timestamp', 'Reposts', 'Comments', 'Likes', 'Content']
    
    fake_df = pd.read_csv('fake_account.csv', sep='\t', names=cols, on_bad_lines='skip')
    legit_df = pd.read_csv('legitimate_account.csv', sep='\t', names=cols, on_bad_lines='skip')
    
    fake_df['Label'] = 1
    legit_df['Label'] = 0
    
    df = pd.concat([fake_df, legit_df], ignore_index=True)
    df['Content'] = df['Content'].fillna('No Content')
    
    # Feature Engineering: Fix Timestamp warning
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['Post_Hour'] = df['Timestamp'].dt.hour.fillna(-1)
    
    # NLP processing (handles English & Chinese)
    tfidf = TfidfVectorizer(max_features=1000)
    text_vectors = tfidf.fit_transform(df['Content']).toarray()
    
    # Prep features
    numerical_features = ['Reposts', 'Comments', 'Likes', 'Post_Hour']
    X_num = df[numerical_features].fillna(0).values
    X = np.hstack((X_num, text_vectors))
    y = df['Label']
    
    # Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    return clf, tfidf, numerical_features

# Load model logic
with st.spinner("Preparing Detection Engine..."):
    model, tfidf_engine, num_cols = train_and_prepare_model()

st.title("🛡️ Cyber-Incident Account Auditor")
st.markdown("Automated detection of suspicious LinkedIn/Social profiles for government reporting.")

# --- SECTION 1: MANUAL CHECK ---
st.header("1. Individual Profile Audit")
with st.expander("Enter Profile Metadata"):
    col1, col2, col3 = st.columns(3)
    with col1:
        reposts = st.number_input("Reposts Count", min_value=0)
        comments = st.number_input("Comments Count", min_value=0)
    with col2:
        likes = st.number_input("Likes Count", min_value=0)
        hour = st.slider("Post Hour (0-23)", 0, 23, 12)
    with col3:
        content = st.text_area("Profile Bio/Post Content", placeholder="Enter the text here...")

    if st.button("Audit Profile"):
        # Process input
        text_v = tfidf_engine.transform([content]).toarray()
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
    # Read the data to audit
    audit_df = pd.read_csv(uploaded_file, sep='\t', names=['ID', 'Timestamp', 'Reposts', 'Comments', 'Likes', 'Content'])
    audit_df['Content'] = audit_df['Content'].fillna('')
    
    # Process for Prediction
    audit_df['Timestamp'] = pd.to_datetime(audit_df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    audit_df['Post_Hour'] = audit_df['Timestamp'].dt.hour.fillna(-1)
    
    batch_text = tfidf_engine.transform(audit_df['Content']).toarray()
    batch_num = audit_df[['Reposts', 'Comments', 'Likes', 'Post_Hour']].fillna(0).values
    X_batch = np.hstack((batch_num, batch_text))
    
    # Predictions
    preds = model.predict(X_batch)
    audit_df['Audit_Result'] = ["FAKE" if p == 1 else "LEGIT" for p in preds]
    
    # Show Flagged Accounts
    flagged = audit_df[audit_df['Audit_Result'] == "FAKE"]
    st.warning(f"Audited {len(audit_df)} accounts. Found {len(flagged)} suspicious profiles.")
    st.dataframe(flagged)
    
    # Export for Government
    st.download_button(
        label="📥 Download Evidence Report for Cyber-Cell",
        data=flagged.to_csv(index=False),
        file_name="flagged_accounts_report.csv",
        mime="text/csv"
    )

st.sidebar.info("This model is trained to detect patterns in engagement metrics and text content common in bot networks.")
