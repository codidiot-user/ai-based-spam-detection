# File: pages/1_ℹ️_About_the_Model.py
import streamlit as st
from style_config import load_css
from PIL import Image

# Page Setup
st.set_page_config(page_title="About the Model", page_icon="ℹ️")
load_css()

st.title("ℹ️ About Our Technology")

st.markdown("""
Our spam detector isn't just looking for keywords. It uses a sophisticated **Deep Learning** approach based on the **RoBERTa** architecture.
It understands context, sarcasm, and typical scam sentence structures.
""")

st.divider()

# --- Model Stats Using Metrics ---
st.header("🤖 Model Specifications")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Architecture", value="RoBERTa", delta="Transformer")
with col2:
    st.metric(label="Parameters", value="125 Million", help="The number of trainable weights in the model network.")
with col3:
    st.metric(label="Dataset Size", value="50k+ SMS", help="Trained on a large corpus of real-world messages.")

st.divider()

# --- The Image Section ---
st.header("📊 Performance Visualization")
st.write("The image below illustrates the model's training process or architecture.")

# Make sure your image_3041af.jpg is in the 'assets' folder!
try:
    image = Image.open('assets/image_3041af.png')
    st.image(image, caption="Our model working architecture", use_column_width=True)
except FileNotFoundError:
    st.warning("⚠️ Image not found. Please ensure 'image_3041af.jpg' is placed inside the 'assets' folder.")

st.divider()
st.subheader("Dataset Transparency")
st.write("We believe in ethical AI. Our model was trained on publicly available datasets of SMS spam and legitimate messages, ensuring a diverse understanding of communication patterns while respecting privacy.")
