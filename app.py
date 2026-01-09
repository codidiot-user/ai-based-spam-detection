import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
import json

# Initialize Firebase
if not firebase_admin._apps:
    firebase_key = st.secrets["firebase_key"]
    if isinstance(firebase_key, str):
        firebase_key = json.loads(firebase_key)
    # Fix escaped newlines in private_key
    if 'private_key' in firebase_key:
        firebase_key['private_key'] = firebase_key['private_key'].replace('\\n', '\n')
    cred = credentials.Certificate(firebase_key)
    firebase_admin.initialize_app(cred)

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase_key"]))
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "logesh1962/sms-spam-detector"  # Files are in root!
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

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
    
    if 'feedback' not in st.session_state:
        st.session_state.feedback = {}
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Correct"):
            st.session_state.feedback[text] = label
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
            st.success("✓ Thanks! We'll keep improving.")
    
    with col2:
        if st.button("❌ Wrong"):
            corrected_label = "Legitimate" if label == "Spam/Fake" else "Spam/Fake"
            st.session_state.feedback[text] = corrected_label
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
            st.success("✓ Thanks! We'll fix this.")

# Footer
st.sidebar.title("About")
st.sidebar.info("Powered by RoBERTa fine-tuned on 50k SMS messages.")



