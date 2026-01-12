# File: style_config.py
import streamlit as st

def load_css():
    st.markdown("""
        <style>
        /* --- 1. FORCE BACKGROUND --- */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
            background-size: cover;
        }
        
        /* --- 2. MAKE TEXT READABLE --- */
        h1, h2, h3, h4, h5, h6, p, div, span, label {
            color: white !important;
        }
        
        /* --- 3. WHITE CARDS FOR INPUTS --- */
        /* This targets the input text area to keep it white and readable */
        .stTextArea textarea {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        /* --- 4. BUTTON STYLING --- */
        .stButton>button {
            background: #ff4b1f;  /* Bright Orange button for contrast */
            background: -webkit-linear-gradient(to right, #ff9068, #ff4b1f);
            background: linear-gradient(to right, #ff9068, #ff4b1f);
            color: white;
            border: none;
            border-radius: 25px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
