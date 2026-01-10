import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
import json

# --- 1. SETUP FIREBASE (The Connection) ---
# Check if we are already connected to avoid errors
if not firebase_admin._apps:
    try:
        # Load the secrets
        firebase_creds = dict(st.secrets["firebase_key"])
        
        # Fix the newlines in the private key
        if "private_key" in firebase_creds:
            firebase_creds["private_key"] = firebase_creds["private_key"].replace("\\n", "\n")
            
        # Connect to Firebase
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        st.stop()

# Get the database client (THIS LINE IS CRITICAL)
db = firestore.client()

# --- 2. LOAD AI MODEL ---
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "logesh1962/sms-spam-detector" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model, device

try:
    tokenizer, model, device = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 3. THE APP INTERFACE ---
st.title("🚀 Spam Detector")
st.write("Paste your message below — I'll scan for spam/scams!")

# Input Area
text = st.text_area("Enter message:", height=100, placeholder="E.g., 'Win free iPhone! Click here...'")

if st.button("Detect Spam!") and text.strip():
    # Run Prediction
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=96).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prob_spam = torch.softmax(outputs.logits, dim=-1)[0][1].item()
    label = "Spam/Fake" if prob_spam > 0.3 else "Legitimate"
    
    # Show Results
    st.subheader(f"**Result: {label}**")
    st.metric("Spam Confidence", f"{prob_spam:.1%}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("SAFE" if label == "Legitimate" else "🚨 ALERT")
    with col2:
        st.info(f"Score: {prob_spam:.4f}")
    
    if label == "Spam/Fake":
        st.warning("⚠️ This looks scammy — avoid clicking links or replying!")

    # --- 4. FEEDBACK SYSTEM ---
    st.divider()
    st.write("**Was this prediction correct?**")
    
    # Generate a unique ID for this specific prediction text
    feedback_key = f"feedback_{hash(text)}"
    
    # Only show buttons if the user hasn't voted yet
    if feedback_key not in st.session_state:
        col_yes, col_no = st.columns(2)
        
        with col_yes:
            if st.button("✅ Correct"):
                # Save 'Correct' feedback to Firebase
                db.collection('feedback').document(str(uuid.uuid4())).set({
                    'message': text,
                    'prediction': label,
                    'confidence': prob_spam,
                    'user_feedback': 'Correct',
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                st.session_state[feedback_key] = True
                st.rerun()

        with col_no:
            if st.button("❌ Wrong"):
                # Save 'Wrong' feedback to Firebase
                db.collection('feedback').document(str(uuid.uuid4())).set({
                    'message': text,
                    'prediction': label,
                    'confidence': prob_spam,
                    'user_feedback': 'Wrong',
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                st.session_state[feedback_key] = True
                st.rerun()

    else:
        st.success("🎉 Thanks for your help! Feedback saved.")

# Sidebar
with st.sidebar:
    st.info("The database is connected!")
    if st.button("Clear Feedback History (Local)"):
        st.session_state.clear()
