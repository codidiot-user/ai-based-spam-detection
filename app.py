import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
import json
# --- DEBUG SECTION (PASTE AT TOP OF APP.PY) ---
st.write("🔧 DEBUG MODE: Testing Database Connection...")

if not firebase_admin._apps:
    try:
        # Load keys
        key_dict = dict(st.secrets["firebase_key"])
        if "private_key" in key_dict:
            key_dict["private_key"] = key_dict["private_key"].replace("\\n", "\n")
        
        # Connect
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
        st.success("✅ 1. Auth Successful")
    except Exception as e:
        st.error(f"❌ 1. Auth Failed: {e}")

try:
    # Force a write test
    db = firestore.client()
    db.collection("debug_test").add({"timestamp": firestore.SERVER_TIMESTAMP, "status": "working"})
    st.success("✅ 2. Write Successful! Check Firestore for 'debug_test' collection.")
except Exception as e:
    st.error(f"❌ 2. Write Failed: {e}")
    st.info("💡 If Auth worked but Write failed, check your IAM Roles in Google Cloud!")
st.divider()
# --- END DEBUG SECTION ---

# --- FIXED FIREBASE INITIALIZATION ---
# --- FIND THIS SECTION IN YOUR APP.PY AND REPLACE IT ---

# Initialize Firebase
if not firebase_admin._apps:
    try:
        # 1. Get the secrets as a dictionary (mutable)
        firebase_creds = dict(st.secrets["firebase_key"])

        # 2. THIS IS THE LINE YOU ASKED ABOUT (The Fix)
        # It looks for "private_key" and fixes the newlines if needed
        if "private_key" in firebase_creds:
            firebase_creds["private_key"] = firebase_creds["private_key"].replace("\\n", "\n")

        # 3. Connect using the fixed credentials
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
        
    except Exception as e:
        st.error(f"Failed to connect to Firebase: {e}")

# Get the database client
db = firestore.client()

# --- MODEL LOADING ---
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

# --- UI & LOGIC ---
st.title("🚀 Spam Detector")
st.write("Paste your message below — I'll scan for spam/scams!")

# Input
text = st.text_area("Enter message:", height=100, placeholder="E.g., 'Win free iPhone! Click here...'")

if st.button("Detect Spam!") and text.strip():
    # Predict
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=96).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prob_spam = torch.softmax(outputs.logits, dim=-1)[0][1].item()
    label = "Spam/Fake" if prob_spam > 0.3 else "Legitimate"
    
    # Display
    st.subheader(f"**Result: {label}**")
    st.metric("Spam Confidence", f"{prob_spam:.1%}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("SAFE" if label == "Legitimate" else "🚨 ALERT")
    with col2:
        st.info(f"Score: {prob_spam:.4f}")
    
    if label == "Spam/Fake":
        st.warning("⚠️ This looks scammy — avoid clicking links or replying!")
    
    # ------------------- FEEDBACK & DATA COLLECTION -------------------
    st.divider()
    st.write("**Was this prediction correct?**")
    
    # Use session state to store feedback status for the specific text
    feedback_key = f"feedback_{hash(text)}"
    
    col1, col2 = st.columns(2)
    
    # Only show buttons if feedback hasn't been given for this specific text
    if feedback_key not in st.session_state:
        with col1:
            if st.button("✅ Correct"):
                # Save to Firebase
                doc_ref = db.collection('feedback').document(str(uuid.uuid4()))
                doc_ref.set({
                    'message': text.strip(),
                    'prediction': label,
                    'confidence': prob_spam,
                    'feedback': 'Correct',
                    'user_label': label,
                    'ip': "unknown",
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                st.session_state[feedback_key] = True
                st.rerun() # Rerun to update UI
        
        with col2:
            if st.button("❌ Wrong"):
                corrected_label = "Legitimate" if label == "Spam/Fake" else "Spam/Fake"
                # Save to Firebase
                doc_ref = db.collection('feedback').document(str(uuid.uuid4()))
                doc_ref.set({
                    'message': text.strip(),
                    'prediction': label,
                    'confidence': prob_spam,
                    'feedback': 'Wrong',
                    'user_label': corrected_label,
                    'ip': "unknown",
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                st.session_state[feedback_key] = True
                st.rerun()
    else:
        st.success("✓ Thank you for your feedback!")

# Footer
st.sidebar.title("About")
st.sidebar.info("Powered by RoBERTa fine-tuned on 50k SMS messages.")


