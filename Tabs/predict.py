# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:33:51 2023

@author: USER
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string
import nltk
from PIL import Image

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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

def app():
    # Load TF-IDF Vectorizer and MinMaxScaler
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    minmax_scaler = joblib.load('minmax_scaler.joblib')
    
    # Load the Best Model
    best_model = load_model('best_model.h5')
    
    # Define category names
    category_names = ['ARTS & CULTURE', 'BUSINESS', 'COMEDY', 'CRIME', 'EDUCATION',
                      'ENTERTAINMENT', 'ENVIRONMENT', 'MEDIA', 'POLITICS', 'RELIGION',
                      'SCIENCE', 'SPORTS', 'TECH', 'WOMEN']
    
    # Load the data
    data = pd.read_csv('news-article-categories.csv', encoding='latin-1')
    
    st.markdown("The **:green[Dataset]** that using for **Train Model**")
    if st.button("Another Sample"):
        data = pd.read_csv('news-article-categories.csv', encoding='latin-1')
        updated_df = data.sample(10)
        st.dataframe(updated_df)
    else:
        st.dataframe(data.sample(10))
    df = pd.DataFrame(data)
    dataset_link = "https://www.kaggle.com/datasets/timilsinabimal/newsarticlecategories"
    st.markdown(f"Link for Dataset: [Click Here]({dataset_link})")
    st.text("")
    
    # Display dataset value count per class as a column
    st.subheader("Dataset Value Count per Class:")
    value_counts = data['category'].value_counts()
    st.bar_chart(value_counts)
    
    st.markdown(
        "Because the data is imbalanced, so for balanced the data, in this case using the **:green[SMOTE]** technique, so after using the **:green[SMOTE]**, the data from before 6877 can be 14028 data, with the 11222 data for **training** and 2806 data for **testing**."
    )
    
    img4 = Image.open('RNN.jpg')
    st.image(img4, use_column_width=True)
    
    st.markdown(
        "For model that used, would use the **:green[Reccurent Neural Network (RNN)]** method, that first the model is tune using **:green[Grid Search CV]** so can find the best hyperparameter that would be used, in this research using the number **:green[epoch is 50]** for training the model."
    )
    
    st.subheader("Best Hyperparameters:")
    best_hyperparameters_code = """
    {
        'units1': 128,
        'activation1': 'relu',
        'dropout1': 0.30000000000000004,
        'num_rnn_layers': 3,
        'units_rnn_0': 96,
        'activation_rnn_0': 'relu',
        'dropout_rnn_0': 0.30000000000000004,
        'num_dense_layers': 1,
        'units_dense_0': 48,
        'activation_dense_0': 'relu',
        'dropout_dense_0': 0.2,
        'units_dense_1': 32,
        'activation_dense_1': 'tanh',
        'dropout_dense_1': 0.30000000000000004,
        'units_dense_2': 48,
        'activation_dense_2': 'tanh',
        'dropout_dense_2': 0.2,
        'units_rnn_1': 32,
        'activation_rnn_1': 'tanh',
        'dropout_rnn_1': 0.30000000000000004,
        'units_rnn_2': 32,
        'activation_rnn_2': 'relu',
        'dropout_rnn_2': 0.2
    }
    """
    st.code(best_hyperparameters_code, language="python")
    
    # img5 = Image.open('7.png')
    # st.image(img5, use_column_width=True)
    
    # img6 = Image.open('8.png')
    # st.image(img6, use_column_width=True)
    
    # Load images
    img5 = Image.open('7.png')
    img6 = Image.open('8.png')
    
    # Display images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img5, use_column_width=True)
    
    with col2:
        st.image(img6, use_column_width=True)
    
    st.markdown(
        "In figure before, shown the training and validation curve. From that curve, had been seen the value of training and validation is not overfitting when using **:green[SMOTE]** method for the imbalance dataset. After Conducting Training and Testing Process, the model RNN give an accuracy of **:green[94.93941553813258%]**, for classifying the news. For the confusion matrix after testing process is given below."
    )
    
    img = Image.open('cnf.png')
    st.image(img, use_column_width=True)
    
    # Streamlit App
    st.title("News Classification")
    st.markdown("<p style='color: gold; font-size: 18px;'>Enter the body of your article here</p>", unsafe_allow_html=True)
    
    # Input for Text
    text_input = st.text_area("Enter Text:")
    
    # Initialize session state for saving predicted articles
    if 'predicted_articles' not in st.session_state:
        st.session_state.predicted_articles = []
    
    # Button to trigger prediction
    if st.button("Predict"):
        # Check if both title and text are provided
        if not text_input:
            st.warning("Please enter text article for prediction.")
        else:
            # Combine Title and Text
            combined_text = text_process(text_input)
            X_new_tfidf = tfidf_vectorizer.transform([combined_text])
            X_new_scaled = minmax_scaler.transform(X_new_tfidf.toarray())
            X_new_rnn = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))
            y_pred_one_hot = best_model.predict(X_new_rnn)
            predicted_label_index = np.argmax(y_pred_one_hot)
            predicted_label = category_names[predicted_label_index]
    
            # Save the predicted article to session state
            st.session_state.predicted_articles.append({
                'category': predicted_label,
                'article': combined_text
            })
    
            # Display Predicted Labels using st.success
            st.success("Predicted Category: " + predicted_label)
    
            # # Now, you can add the predicted article to the "Top News" section
            # st.header("Predicted Article")
            # with st.expander(f"Predicted Article_{len(st.session_state.predicted_articles)}"):
            #     st.title("Predicted Article")
            #     st.markdown(f"**Category:** {predicted_label}")
            #     st.markdown(combined_text)
    
    # Display previous predicted articles
    st.header("Predicted Articles")
    for i, article in enumerate(st.session_state.predicted_articles):
        with st.expander(f"Previous Article_{i}"):
            st.title(f"Category: {article['category']}")
            st.markdown(article['article'])
