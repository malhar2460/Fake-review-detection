# Fake Review Detection Project Documentation

Welcome to the comprehensive documentation for our Fake Review Detection project. This project is designed to distinguish between computer-generated (fake) reviews and original (human-written) reviews by leveraging a variety of natural language processing (NLP) techniques, traditional machine learning models, and deep learning methods. In this document, we detail the project background, data processing, feature engineering, model experimentation, experiment tracking using MLflow, and finally, the deployment of our application.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Collection & Preprocessing](#data-collection--preprocessing)
3. [Feature Engineering & Embeddings](#feature-engineering--embeddings)
4. [Modeling Approaches](#modeling-approaches)
   - [Traditional Machine Learning Models](#traditional-machine-learning-models)
   - [Deep Learning Models](#deep-learning-models)
5. [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
6. [Deployment](#deployment)
7. [Conclusion](#conclusion)

---

## Project Overview

Fake reviews are a growing problem in the online ecosystem, impacting consumer trust and business reputations. Our project aims to automatically detect and flag these computer-generated reviews by analyzing textual content. To achieve this, we have employed various techniques including:

- **Text Preprocessing:** Cleaning and normalizing text data.
- **Feature Engineering:** Extracting useful metrics such as lexical diversity, sentiment scores, and syntactic patterns.
- **Embeddings:** Generating different embeddings using methods such as TF-IDF, Count Vectorization, and precomputed embeddings from models like BERT and GloVe.
- **Model Training:** Experimenting with a range of traditional machine learning models (Logistic Regression, Random Forest, SVC) and a more complex deep learning model using a two-layer LSTM.
- **Experiment Tracking:** Logging every experiment with detailed metrics, hyperparameters, and artifacts using MLflow.

---

## Data Collection & Preprocessing

Our data consists of review texts along with corresponding labels indicating whether a review is computer-generated or original. The preprocessing pipeline includes:

- **Text Cleaning:** Removing unwanted characters, punctuation, and noise.
- **Normalization:** Converting text to lower case, tokenizing sentences and words, and applying techniques such as stemming, lemmatization, and stop word removal.
- **Feature Extraction:** Calculating metrics like lexical diversity, average word length, sentiment polarity, subjectivity, Flesch Reading Ease, sentence length, and various part-of-speech counts.

Processed data files are stored in our `../Data/Feature-Engineered/` folder.

---

## Feature Engineering & Embeddings

To capture the nuances of textual data, we explored several embedding techniques:

- **TF-IDF Embeddings:** Transforming text into weighted term-frequency representations.
- **Count Vectorization:** Creating basic term-frequency vectors.
- **Precomputed Embeddings:** Using models such as BERT and GloVe to generate embeddings, which are stored as CSV files in the `../../embeddings/` folder.

These methods provide diverse representations of the data, enabling our models to learn both syntactic and semantic patterns.
[All CSV Files Used In Project](https://drive.google.com/file/d/1DNBx44dBOd0kvqR-lWxq74RI5_hsP6pN/view?usp=drive_link)
---

## Modeling Approaches

### Traditional Machine Learning Models

We experimented with several scikit-learn models:

- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Classifier (SVC)**

For each model, we performed hyperparameter tuning using GridSearchCV with feasible parameter grids. Experiment results, including confusion matrices and metrics (accuracy, precision, recall, F1 score), are logged in MLflow.

### Deep Learning Models

To capture complex patterns, we built a deep learning model using TensorFlow Keras:

- **Two-Layer LSTM:** The model includes an embedding layer, two LSTM layers, and dense layers with dropout for regularization.
- **Text Tokenization & Padding:** We convert raw text into sequences using Keras’ Tokenizer and pad them to a uniform length.
- **Evaluation:** Model performance is evaluated on standard metrics and confusion matrices are logged.

---

## Experiment Tracking with MLflow

Our experiments are fully tracked using MLflow. For every run, we log:

- **Parameters:** File names, model types, hyperparameters, and embedding types.
- **Metrics:** Accuracy, precision, recall, and F1 score.
- **Artifacts:** Confusion matrices (as PNG images) and model artifacts.
- **Datasets:** Using the `mlflow.data` API, our dataset information is logged and appears under the MLflow UI's "Datasets" tab (for MLflow ≥ 2.4). When unavailable, the CSV files are logged as artifacts.

You can view all our experiments on Dagshub through MLflow here:  
[View MLflow Experiments on Dagshub](https://dagshub.com/malhar.c.prajapati/my-first-repo.mlflow/)

A progress log (`progress_log.csv`) is maintained to ensure experiments are not re-run unnecessarily.

---

## Deployment

The Fake Review Detection web application is deployed and accessible online. Users can enter review text to receive predictions on whether the review is computer-generated or original. The application also provides various text analytics and visualizations for better interpretability.

**Access the deployed web app here:**  
[Fake Review Detection Web App](https://fake-review-detection-mkgwujmh2b6dzcgb6gka2r.streamlit.app/)

---

## Conclusion

Our project integrates advanced NLP techniques, comprehensive feature engineering, rigorous experimentation, and robust MLflow-based tracking to build a reproducible system for fake review detection. This documentation provides an end-to-end overview of our process, from data preprocessing and model training to deployment. We encourage you to explore the code repository and MLflow experiment dashboard for more details.

*Thank you for your interest in our project!*
