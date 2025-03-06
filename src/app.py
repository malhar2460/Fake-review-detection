import streamlit as st
st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import joblib
import numpy as np
import pandas as pd
import altair as alt
import nltk
import re
import string
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
from io import StringIO

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# -------------------------------
# Text Analysis Helper Functions
# -------------------------------
def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0

def average_word_length(text):
    words = text.split()
    return sum(len(word) for word in words) / len(words) if words else 0

def sentiment_polarity(text):
    return TextBlob(text).sentiment.polarity

def subjectivity_score(text):
    return TextBlob(text).sentiment.subjectivity

def flesch_reading_ease(text):
    words = text.split()
    sentences = nltk.sent_tokenize(text)
    syllables = sum(len(re.findall(r"[aeiouy]+", word.lower())) for word in words)
    if not words or not sentences:
        return 0
    ASL = len(words) / len(sentences)
    ASW = syllables / len(words)
    return 206.835 - 1.015 * ASL - 84.6 * ASW

def sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    return sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0

def punctuation_count(text):
    return sum(1 for c in text if c in string.punctuation)

def count_pos_tags(text, tag_prefix):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    return sum(1 for _, tag in pos_tags if tag.startswith(tag_prefix))

# -------------------------------
# Load Model & Vectorizer
# -------------------------------
@st.cache_resource
def load_model_vectorizer():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    vectorizer_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")
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
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Single Prediction", "Insights"])

# -------------------------------
# Home Page
# -------------------------------
if page == "Home":
    st.title("Fake Review Detection Web App")
    st.subheader("Welcome")
    st.write("Analyze any review text to see if it's computer-generated or original. Use the sidebar to navigate.")
    st.write("This app provides text analytics like lexical diversity, sentiment, and more. Enjoy exploring the insights!")

# -------------------------------
# Single Prediction Page
# -------------------------------
elif page == "Single Prediction":
    st.title("Single Prediction")
    review_text = st.text_area("Enter review text:", height=120)
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05)
    
    if st.button("Predict"):
        if not review_text.strip():
            st.error("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                # Transform input text
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
                st.session_state["pred_text"] = review_text
                st.session_state["pred_result"] = result
                st.session_state["pred_probs"] = probabilities
            st.success(f"Prediction: {result}")
            if probabilities is not None:
                data = {
                    "Label": ["Computer Generated", "Original"],
                    "Probability": [probabilities[0], probabilities[1]]
                }
                df_probs = pd.DataFrame(data)
                bar_chart = alt.Chart(df_probs).mark_bar(color="#66b3ff").encode(
                    x=alt.X("Label", axis=alt.Axis(labelAngle=0)),
                    y="Probability"
                ).properties(title="Prediction Probability Distribution", width=300, height=400)
                st.altair_chart(bar_chart, use_container_width=True)

# -------------------------------
# Insights Page
# -------------------------------
elif page == "Insights":
    st.title("Text Insights")
    if "pred_text" not in st.session_state or not st.session_state["pred_text"]:
        st.info("No text analyzed yet. Go to 'Single Prediction' first.")
    else:
        text = st.session_state["pred_text"]
        prediction = st.session_state["pred_result"]
        probabilities = st.session_state["pred_probs"]
        st.subheader(f"Prediction: {prediction}")
        if probabilities is not None:
            data = {
                "Label": ["Computer Generated", "Original"],
                "Probability": [probabilities[0], probabilities[1]]
            }
            df_probs = pd.DataFrame(data)
            bar_chart = alt.Chart(df_probs).mark_bar(color="#66b3ff").encode(
                x=alt.X("Label", axis=alt.Axis(labelAngle=0)),
                y="Probability"
            ).properties(title="Prediction Probability Distribution", width=300, height=400)
            st.altair_chart(bar_chart, use_container_width=True)
        # Compute text metrics
        lex_div = lexical_diversity(text)
        avg_wl = average_word_length(text)
        pol = sentiment_polarity(text)
        subj = subjectivity_score(text)
        fre = flesch_reading_ease(text)
        avg_len = sentence_length(text)
        punct = punctuation_count(text)
        nn = count_pos_tags(text, "NN")
        vb = count_pos_tags(text, "VB")
        jj = count_pos_tags(text, "JJ")
        rb = count_pos_tags(text, "RB")
        
        st.markdown("### Text Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lexical Diversity", f"{lex_div:.3f}")
        col2.metric("Avg Word Length", f"{avg_wl:.2f}")
        col3.metric("Polarity", f"{pol:.3f}")
        col4.metric("Subjectivity", f"{subj:.3f}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Flesch Ease", f"{fre:.2f}")
        col2.metric("Avg Sentence Len", f"{avg_len:.2f}")
        col3.metric("Punctuation", str(punct))
        col4.metric("Adverbs", str(rb))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nouns", str(nn))
        col2.metric("Verbs", str(vb))
        col3.metric("Adjectives", str(jj))
        
        st.markdown("### Word Cloud")
        wc = WordCloud(width=600, height=300, background_color="white").generate(text)
        st.image(wc.to_array(), caption="Word Cloud", width=600)
