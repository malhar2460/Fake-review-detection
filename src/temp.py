import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

df = pd.read_csv("../Data/Feature-Engineered/preprocessed_lemmatization_features.csv")

texts = df["processed_text"].dropna().astype(str).tolist()

vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(texts)

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("tfidf_vectorizer.pkl created successfully.")
