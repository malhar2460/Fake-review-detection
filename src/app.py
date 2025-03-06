import streamlit as st
import joblib
import numpy as np
import pandas as pd
import nltk
import re
import string
import altair as alt
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
from io import StringIO

# --- NLTK Downloads ---
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# --- Text Analysis Helper Functions ---
def lexical_diversity(text):
    w = text.split()
    return len(set(w)) / len(w) if w else 0

def average_word_length(text):
    w = text.split()
    return sum(len(i) for i in w) / len(w) if w else 0

def sentiment_polarity(text):
    return TextBlob(text).sentiment.polarity

def subjectivity_score(text):
    return TextBlob(text).sentiment.subjectivity

def flesch_reading_ease(text):
    w = text.split()
    s = nltk.sent_tokenize(text)
    syl = sum(len(re.findall(r"[aeiouy]+", wd.lower())) for wd in w)
    if not w or not s:
        return 0
    ASL = len(w) / len(s)
    ASW = syl / len(w)
    return 206.835 - 1.015 * ASL - 84.6 * ASW

def sentence_length(text):
    s = nltk.sent_tokenize(text)
    return sum(len(i.split()) for i in s) / len(s) if s else 0

def punctuation_count(text):
    return sum(1 for c in text if c in string.punctuation)

def count_pos_tags(text, tag_prefix):
    t = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(t)
    return sum(1 for _, tg in pos_tags if tg.startswith(tag_prefix))

# --- Load Model & Vectorizer ---
@st.cache_resource
def load_model_vectorizer():
    m = joblib.load("model.pkl")
    v = joblib.load("tfidf_vectorizer.pkl")
    return m, v

model, vectorizer = load_model_vectorizer()

def predict_review(text, threshold):
    x_text = vectorizer.transform([text])
    # Our model expects an extra 11-dimensional dummy (if that’s how your model was trained)
    dummy = np.zeros((1, 11))
    feats = np.hstack((x_text.toarray(), dummy))
    try:
        p = model.predict_proba(feats)[0]
    except Exception:
        p = None
    if p is not None:
        pr = 1 if p[1] >= threshold else 0
    else:
        pr = model.predict(feats)[0]
    return "Computer generated" if pr == 0 else "Original", p

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Inject a bit of CSS for metrics (especially helpful for dark mode) ---
st.markdown(
    """
    <style>
    /* Make the metrics more visible on dark background */
    [data-testid="metric-container"] {
        background-color: #2c2f38;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
# Removed About page; added "Batch Analysis" as a new feature.
page = st.sidebar.radio("Go to", ["Home", "Single Prediction", "Insights", "Batch Analysis"])

# --- Pages ---
if page == "Home":
    st.title("Fake Review Detection Web App")
    st.subheader("Welcome")
    st.write("Analyze any review text to see if it's computer-generated or original. Use the sidebar to navigate.")
    st.write("This app also provides text analytics like lexical diversity, sentiment, and more.")
    st.write("You can also perform batch analysis of multiple reviews by uploading a CSV file.")

elif page == "Single Prediction":
    st.title("Single Prediction")
    review_text = st.text_area("Enter review text:", height=120)
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05)

    if st.button("Predict"):
        if not review_text.strip():
            st.error("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                prediction, probs = predict_review(review_text, threshold)
                st.session_state["pred_text"] = review_text
                st.session_state["pred_result"] = prediction
                st.session_state["pred_probs"] = probs

            st.success(f"Prediction: {prediction}")

            # --- Plot Probability Distribution as Bar Chart ---
            if probs is not None:
                data = {
                    "Label": ["Computer Generated", "Original"],
                    "Probability": [probs[0], probs[1]]
                }
                df_probs = pd.DataFrame(data)
                bar_chart = alt.Chart(df_probs).mark_bar(color="#66b3ff").encode(
                    x=alt.X("Label", axis=alt.Axis(labelAngle=0)),
                    y="Probability"
                ).properties(title="Prediction Probability Distribution", width=300, height=400)
                st.altair_chart(bar_chart, use_container_width=True)

elif page == "Insights":
    st.title("Text Insights")
    if "pred_text" not in st.session_state or not st.session_state["pred_text"]:
        st.info("No text analyzed yet. Go to 'Single Prediction' first.")
    else:
        text = st.session_state["pred_text"]
        prediction = st.session_state["pred_result"]
        probs = st.session_state["pred_probs"]

        st.subheader(f"Prediction: {prediction}")
        if probs is not None:
            data = {
                "Label": ["Computer Generated", "Original"],
                "Probability": [probs[0], probs[1]]
            }
            df_probs = pd.DataFrame(data)
            bar_chart = alt.Chart(df_probs).mark_bar(color="#66b3ff").encode(
                x=alt.X("Label", axis=alt.Axis(labelAngle=0)),
                y="Probability"
            ).properties(title="Prediction Probability Distribution", width=300, height=400)
            st.altair_chart(bar_chart, use_container_width=True)

        # --- Compute Text Metrics ---
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
        col1, col2, col3, col0 = st.columns(4)
        with col1:
            st.metric("Lexical Diversity", f"{lex_div:.3f}")
        with col2:
            st.metric("Avg Word Length", f"{avg_wl:.2f}")
        with col3:
            st.metric("Polarity", f"{pol:.3f}")
        with col0:
            st.metric("Adverbs", str(rb))

        col4, col5, col6, col7 = st.columns(4)
        with col4:
            st.metric("Subjectivity", f"{subj:.3f}")
        with col5:
            st.metric("Flesch Ease", f"{fre:.2f}")
        with col6:
            st.metric("Avg Sentence Len", f"{avg_len:.2f}")
        with col7:
            st.metric("Punctuation", str(punct))
        
        col8, col9, col10, col11 = st.columns(4)
        with col8:
            st.metric("Nouns", str(nn))
        with col9:
            st.metric("Verbs", str(vb))
        with col10:
            st.metric("Adjectives", str(jj))

        # --- Word Cloud ---
        st.markdown("### Word Cloud")
        wc = WordCloud(width=600, height=300, background_color="white").generate(text)
        st.image(wc.to_array(), caption="Word Cloud", width=600)
elif page == "Batch Analysis":
    st.title("Batch Analysis")
    st.write("Upload a CSV file containing your reviews.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error("Error reading CSV file.")
        else:
            # Ask user to select which column should be treated as review text
            selected_column = st.selectbox(
                "Select the column to treat as review text:",
                options=df_batch.columns,
                index=df_batch.columns.tolist().index("review_text") if "review_text" in df_batch.columns else 0
            )
            st.write(f"Processing reviews from column: **{selected_column}**")
            
            threshold = st.slider("Prediction Threshold for Batch Analysis", 0.0, 1.0, 0.5, 0.05)
            
            # Add a button to trigger processing
            if st.button("Process Reviews"):
                predictions = []
                metrics = []
                for review in df_batch[selected_column].fillna("").astype(str):
                    pred, probs = predict_review(review, threshold)
                    predictions.append(pred)
                    metrics.append({
                        "Lexical Diversity": lexical_diversity(review),
                        "Avg Word Length": average_word_length(review),
                        "Polarity": sentiment_polarity(review),
                        "Subjectivity": subjectivity_score(review),
                        "Flesch Ease": flesch_reading_ease(review),
                        "Avg Sentence Len": sentence_length(review),
                        "Punctuation": punctuation_count(review)
                    })
                df_batch["Prediction"] = predictions
                df_metrics = pd.DataFrame(metrics)
                st.subheader("Batch Prediction Results")
                st.dataframe(df_batch.head())

                st.subheader("Aggregated Text Metrics")
                st.dataframe(df_metrics.describe())

                # Display count of predictions as a bar chart
                count_df = df_batch["Prediction"].value_counts().reset_index()
                count_df.columns = ["Prediction", "Count"]
                bar_chart = alt.Chart(count_df).mark_bar(color="#66b3ff").encode(
                    x=alt.X("Prediction", axis=alt.Axis(labelAngle=0)),
                    y="Count"
                ).properties(title="Distribution of Predictions", width=300, height=400)
                st.altair_chart(bar_chart, use_container_width=True)
                
                # Add a download button for the results
                csv = df_batch.to_csv(index=False)
                st.download_button("Download Predictions as CSV", csv, "batch_predictions.csv", "text/csv")
