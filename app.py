import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

def clean_text(text):
 text = re.sub(r'<.*?>', '', text)
 text = re.sub(r'[^a-zA-Z]', ' ', text)
 text = text.lower()
 return text

model = tf.keras.models.load_model("sentiment_gru_model.h5")

with open("tokenizer.pkl", "rb") as f:
 tokenizer = pickle.load(f)

st.title("IMDB Review Sentiment Analyzer")
input_text = st.text_area("Enter a movie review:")

if st.button("Analyze"):
 clean_input = clean_text(input_text)
 seq = tokenizer.texts_to_sequences([clean_input])
 padded = pad_sequences(seq, maxlen=200)
 prediction = model.predict(padded)[0][0]
 sentiment = "Positive" if prediction >= 0.5 else "Negative"
 st.write(f"Prediction: {sentiment} ({prediction:.2f})")
