import streamlit as st
st.set_page_config(
    page_title="Fake Review Detection - Prediction",
    page_icon="🤖",
    layout="wide"
)

import os
# Set NLTK_DATA to a writable directory (/tmp/nltk_data)
os.environ["NLTK_DATA"] = "/tmp/nltk_data"

import nltk
# Create the directory if it doesn't exist
os.makedirs("/tmp/nltk_data", exist_ok=True)
# Add /tmp/nltk_data to nltk.data.path
nltk.data.path.insert(0, "/tmp/nltk_data")

# Download required NLTK data if not already present
try:
    nltk.data.find("tokenizers/punkt/english.pickle")
except LookupError:
    nltk.download("punkt", download_dir="/tmp/nltk_data", quiet=True)
try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger", download_dir="/tmp/nltk_data", quiet=True)

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import joblib
import numpy as np
import altair as alt
import re
import string
from textblob import TextBlob
from wordcloud import WordCloud

# -------------------------------
# Load Model & Vectorizer
# -------------------------------
@st.cache_resource
def load_model_vectorizer():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "model.pkl")
    vectorizer_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model files not found. Please ensure 'model.pkl' and 'tfidf_vectorizer.pkl' exist in the app directory.")
        return None, None
    m = joblib.load(model_path)
    v = joblib.load(vectorizer_path)
    return m, v

model, vectorizer = load_model_vectorizer()
if model is None or vectorizer is None:
    st.stop()

# -------------------------------
# Single Prediction Section
# -------------------------------
st.title("Fake Review Detection - Single Prediction")

review_text = st.text_area("Enter review text:", height=150)
threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05)

if st.button("Predict"):
    if not review_text.strip():
        st.error("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            x_text = vectorizer.transform([review_text])
            dummy = np.zeros((1, 11))
            feats = np.hstack((x_text.toarray(), dummy))
            try:
                probabilities = model.predict_proba(feats)[0]
            except Exception:
                probabilities = None
                prediction = model.predict(feats)[0]
            else:
                prediction = 1 if probabilities[1] >= threshold else 0
            result = "Computer generated" if prediction == 0 else "Original"
            st.success(f"Prediction: {result}")
            if probabilities is not None:
                data = {
                    "Label": ["Computer Generated", "Original"],
                    "Probability": [probabilities[0], probabilities[1]]
                }
                df_probs = np.array(data)  # Optionally, you can display as a table or use Altair chart.
                # Using Altair bar chart with horizontal labels:
                df_probs = pd.DataFrame(data)
                bar_chart = alt.Chart(df_probs).mark_bar(color="#66b3ff").encode(
                    x=alt.X("Label", axis=alt.Axis(labelAngle=0)),
                    y="Probability"
                ).properties(title="Prediction Probability Distribution", width=300, height=400)
                st.altair_chart(bar_chart, use_container_width=True)
