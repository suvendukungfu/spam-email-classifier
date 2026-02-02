import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os

# Configuration
MODEL_PATH = "ml/spam_model.h5"
TOKENIZER_PATH = "ml/tokenizer.pkl"
MAX_LENGTH = 100

# Set Page Config
st.set_page_config(
    page_title="Spam Detective AI",
    page_icon="📧",
    layout="centered"
)

# Custom CSS for UI polish
st.markdown("""
    <style>
    .stTextArea textarea {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        width: 100%;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Helper Function: Clean Text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Load Resources (Cached)
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer: {e}")
        return None, None

# Main App
def main():
    st.title("📧 Spam Detective AI")
    st.markdown("### Intelligent Email Classification System")
    st.write("Enter an email or SMS message below to check if it's **Spam** or **Ham**.")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("This AI uses a Bidirectional LSTM network trained on the SMS Spam Collection dataset.")
        st.markdown("---")
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/suvendukungfu/spam-email-classifier)")
        st.markdown("Created by **Suvendu Sahoo**")

    # Load Model
    model, tokenizer = load_resources()

    if model is None or tokenizer is None:
        st.warning("Model artifacts not found. Please train the model first.")
        return

    # User Input
    message = st.text_area("Message Content", height=150, placeholder="Type your message here...")

    if st.button("Analyze Message"):
        if message.strip():
            with st.spinner("Analyzing text patterns..."):
                # Preprocess
                cleaned_text = clean_text(message)
                sequences = tokenizer.texts_to_sequences([cleaned_text])
                padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

                # Predict
                prediction = model.predict(padded)[0][0]
                confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
                is_spam = prediction > 0.5

                # Display Result
                st.markdown("---")
                if is_spam:
                    st.error(f"🚨 **SPAM DETECTED**")
                else:
                    st.success(f"✅ **NOT SPAM (HAM)**")
                
                # Confidence Meter
                st.progress(confidence)
                st.caption(f"AI Confidence Score: {confidence*100:.2f}%")
                
        else:
            st.info("Please enter a message to analyze.")

    # Footer
    st.markdown("---")
    st.markdown("*Built with TensorFlow & Streamlit*")

if __name__ == "__main__":
    main()
