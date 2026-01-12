# File: pages/2_📬_Contact_Us.py
import streamlit as st
from style_config import load_css

# Page Setup
st.set_page_config(page_title="Contact Us", page_icon="📬")
load_css()

st.title("📬 Get in Touch")
st.write("Have questions about the model, suggestions for improvement, or want to report an issue? We'd love to hear from you.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📧 Email Us
    For general inquiries or collaboration:
    
    **spammodel123@gmail.com**
    """)

with col2:
    st.markdown("""
    ### 🐙 GitHub / Project Page
    View the source code or contribute:
    
    [**https://github.com/codidiot-user/spam-model-a1**](https://github.com/codidiot-user/spam-model-a1)
    """)

st.divider()

st.info("ℹ️ **Note:** This is a research and demonstration project. While highly accurate, always exercise caution with suspicious messages.")
