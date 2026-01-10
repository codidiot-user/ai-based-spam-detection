import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import firebase_admin
from firebase_admin import credentials, firestore
import uuid

# --- 1. ROBUST FIREBASE SETUP ---
if not firebase_admin._apps:
    try:
        # Load secrets
        key_dict = dict(st.secrets["firebase_key"])
        if "private_key" in key_dict:
            key_dict["private_key"] = key_dict["private_key"].replace("\\n", "\n")
        
        # Connect
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"❌ Critical Connection Error: {e}")
        st.stop()

# Get Database Client
db = firestore.client()

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = "logesh1962/sms-spam-detector" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

try:
    tokenizer, model = load_model()
except:
    st.error("Model failed to load.")
    st.stop()

# --- 3. UI & LOGIC ---
st.title("🚀 Spam Detector")


# Main Input
text = st.text_area("Enter message:", height=100)

if st.button("Analyze") and text:
    # Predict
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=96)
    with torch.no_grad():
        outputs = model(**inputs)
    prob = torch.softmax(outputs.logits, dim=-1)[0][1].item()
    label = "Spam/Fake" if prob > 0.3 else "Legitimate"
    
    # Store results in Session State so they persist
    st.session_state['last_text'] = text
    st.session_state['last_label'] = label
    st.session_state['last_prob'] = prob
    st.session_state['analyzed'] = True

# Display Results & Feedback
if st.session_state.get('analyzed'):
    st.divider()
    
    # Show Prediction
    lbl = st.session_state['last_label']
    score = st.session_state['last_prob']
    st.subheader(f"Result: {lbl} ({score:.1%})")
    
    st.write("---")
    st.write("👇 **Choose any one:**")
    
    col1, col2 = st.columns(2)
    
    # BUTTON 1: CORRECT
    with col1:
        if st.button("✅ It is Correct"):
            st.write("⏳ Saving...") # Debug message
            try:
                # Direct write - No complex logic
                doc_ref = db.collection('feedback').document()
                doc_ref.set({
                    'message': st.session_state['last_text'],
                    'prediction': lbl,
                    'confidence': score,
                    'user_feedback': 'Correct',
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                st.success(f"Saved to DB! ID: {doc_ref.id}") # If you see this, IT WORKED.
            except Exception as e:
                st.error(f"Save Failed: {e}")

    # BUTTON 2: WRONG
    with col2:
        if st.button("❌ It is Wrong"):
            st.write("⏳ Saving...") 
            try:
                doc_ref = db.collection('feedback').document()
                doc_ref.set({
                    'message': st.session_state['last_text'],
                    'prediction': lbl,
                    'confidence': score,
                    'user_feedback': 'Wrong',
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                st.success(f"Saved to DB! ID: {doc_ref.id}")
            except Exception as e:
                st.error(f"Save Failed: {e}")


