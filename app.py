import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Function to clean input text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower()
    return text

# Load tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

# Rebuild the model architecture
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    GRU(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Build the model to initialize weights
model.build(input_shape=(None, 200))

# Load trained weights
model.load_weights("gru_model.weights.h5")

# Streamlit UI
st.title("🎬 IMDB Sentiment Analyzer (GRU-Based)")
st.write("Enter a movie review below to analyze its sentiment.")

# Text input
text_input = st.text_area("📝 Movie Review")

# Prediction button
if st.button("Analyze Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = clean_text(text_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=200)
        prediction = model.predict(padded)[0][0]
        sentiment = "😊 Positive" if prediction >= 0.5 else "😠 Negative"
        st.success(f"**Sentiment**: {sentiment} ({prediction:.2f})")

