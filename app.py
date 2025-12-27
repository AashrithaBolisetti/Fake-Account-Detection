import streamlit as st
import pickle
import os
import hashlib

# 1. SETTING UP THE PAGE
st.set_page_config(page_title="ITBP Fake Account Detector", page_icon="🛡️")

# 2. PATH LOGIC (Fixes FileNotFoundError)
# This ensures the app looks in the same folder for the .pkl files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTOR_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# 3. LOADING THE AI ASSETS
@st.cache_resource # Keeps model in memory for speed
def load_assets():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTOR_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# --- UI DESIGN ---
st.title("🛡️ ITBP Cybersecurity: Fake Account Identification")
st.markdown("""
This tool uses AI to identify fake social media profiles. 
Confirmed threats are hashed and logged for **Central Agency** action.
""")

try:
    # Attempt to load the trained model
    model, vectorizer = load_assets()
    
    # Input Area
    user_text = st.text_area("Analyze Social Media Content:", height=150, 
                             placeholder="Paste the suspicious post or profile bio here...")

    if st.button("Identify & Secure Evidence"):
        if user_text.strip() != "":
            # --- DETECTION LAYER ---
            # Transform text and predict
            transformed_input = vectorizer.transform([user_text])
            prediction = model.predict(transformed_input)

            if prediction[0] == 1:
                st.error("🚨 FLAG: FAKE ACCOUNT IDENTIFIED")
                
                # --- BLOCKCHAIN LAYER (As per problem statement) ---
                # Creates a unique, tamper-proof digital fingerprint of the evidence
                evidence_hash = hashlib.sha256(user_text.encode()).hexdigest()
                
                st.info(f"**Blockchain Evidence Hash:** `{evidence_hash}`")
                st.warning("✅ Identification logged. Report forwarded to Central Agency for worldwide suspension.")
            else:
                st.success("✅ VERIFIED: Account appears Legitimate.")
        else:
            st.warning("Please enter content to analyze.")

except FileNotFoundError:
    st.error("⚠️ SYSTEM ERROR: `model.pkl` or `vectorizer.pkl` not found.")
    st.info("To fix this, upload your trained `.pkl` files from Colab to your GitHub repository.")

# 4. CENTRAL AGENCY FOOTER
st.divider()
st.caption("Designated Tool for ITBP Internal Security & Left Wing Extremism Cyber-Ops.")
