# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:48:03 2023

@author: USER
"""

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string
import streamlit as st
import nltk

# Download NLTK data
nltk.download('punkt')

# Function to process text
def text_process(text):
    if isinstance(text, float) and np.isnan(text):
        return ""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

# Function to justify text and add paragraphs
def justify_and_paragraphs(text, sentences_per_paragraph=10):
    sentences = nltk.sent_tokenize(text)
    paragraphs = [sentences[i:i + sentences_per_paragraph] for i in range(0, len(sentences), sentences_per_paragraph)]
    justified_text = '\n\n'.join([' '.join(paragraph) for paragraph in paragraphs])
    return justified_text

# Load TF-IDF Vectorizer and MinMaxScaler
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
minmax_scaler = joblib.load('minmax_scaler.joblib')

# Load the Best Model
best_model = load_model('best_model.h5')

# Define category names
category_names = ['ARTS & CULTURE', 'BUSINESS', 'COMEDY', 'CRIME', 'EDUCATION',
                  'ENTERTAINMENT', 'ENVIRONMENT', 'MEDIA', 'POLITICS', 'RELIGION',
                  'SCIENCE', 'SPORTS', 'TECH', 'WOMEN']

st.title("News Website")

# Load the data
data = pd.read_csv('news-article-categories.csv', encoding='latin-1')

# Display category select box
selected_category = st.selectbox("Select a Category", [""] + category_names)

# Dynamic selectbox for news articles in the selected category
selected_article_title = st.selectbox("Select a News Article", [""] + list(data[data['category'] == selected_category]['title']))

if selected_category and selected_article_title:
    # Display the selected news article title
    st.title(selected_article_title)

    # Fetch the corresponding news article body
    selected_article_body = data[(data['category'] == selected_category) & (data['title'] == selected_article_title)]['body'].iloc[0]

    # Justify text and add paragraphs
    justified_and_paragraphs_text = justify_and_paragraphs(selected_article_body)

    # Display the news article body
    st.markdown(justified_and_paragraphs_text, unsafe_allow_html=True)  # Using markdown to render HTML content

# # Streamlit App
# st.title("Text Classification App")

# # Input for Title
# title_input = st.text_input("Enter Title:")

# # Input for Text
# text_input = st.text_area("Enter Text:")

# # Button to trigger prediction
# if st.button("Predict"):
#     # Check if both title and text are provided
#     if not title_input or not text_input:
#         st.warning("Please enter both title and text for prediction.")
#     else:
#         # Combine Title and Text
#         combined_text = title_input + ' ' + text_input

#         # Apply the same text processing as during training
#         combined_text = text_process(combined_text)

#         # Use the loaded TF-IDF Vectorizer and MinMaxScaler
#         X_new_tfidf = tfidf_vectorizer.transform([combined_text])
#         X_new_scaled = minmax_scaler.transform(X_new_tfidf.toarray())

#         # Reshape input data for RNN
#         X_new_rnn = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))

#         # Make predictions using the loaded model
#         y_pred_one_hot = best_model.predict(X_new_rnn)

#         # Find the index of the maximum value in the array
#         predicted_label_index = np.argmax(y_pred_one_hot)
        
#         # Map the index to the corresponding category
#         predicted_label = category_names[predicted_label_index]

#         # Display Predicted Labels using st.success
#         st.success("Predicted Category: " + predicted_label)
