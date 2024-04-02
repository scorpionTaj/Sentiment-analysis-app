import streamlit as st
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from Data_Mining_Project import rf_model, vectorizer, scaler

def preprocess_comment(comment):
    comment = comment.lower()
    comment = ''.join([char for char in comment if char not in string.punctuation])
    tokens = word_tokenize(comment)
    stopwords_english = set(stopwords.words('english'))
    comment = [word for word in tokens if word not in stopwords_english]
    lemmatizer = WordNetLemmatizer()
    comment = [lemmatizer.lemmatize(word) for word in comment]
    return ' '.join(comment)

def predict_sentiment(comment):
    preprocessed_comment = preprocess_comment(comment)
    comment_vectorized = vectorizer.transform([preprocessed_comment])
    # Combine features
    comment_length = len(preprocessed_comment)
    word_count = len(preprocessed_comment.split())
    puncts = sum(1 for char in preprocessed_comment if char in string.punctuation) / comment_length if comment_length > 0 else 0
    # Convert other features to 2D arrays
    features = np.array([[comment_length, word_count, puncts]])  # Convert to 2D array
    # Combine all features
    features = np.hstack([features, comment_vectorized.toarray()])
    # Scale features (excluding the first three columns)
    features_scaled = scaler.transform(features[:, :3])  # Scale only the first three columns
    # Concatenate the scaled features with the remaining features
    features_scaled = np.hstack([features_scaled, features[:, 3:]])
    # Predict sentiment
    prediction = rf_model.predict(features_scaled)
    return prediction[0]

def main():
    st.title("Sentiment Analysis App")
    st.write("Welcome to the sentiment analysis app!")

    comment = st.text_area("Enter your comment here:")

    if st.button("Analyze"):
        if comment:
            prediction = predict_sentiment(comment)
            if prediction == 1:
                st.write("The sentiment of the comment is: <span style='color:green'>Positive</span>", unsafe_allow_html=True)
            else:
                st.write("The sentiment of the comment is: <span style='color:red'>Negative</span>", unsafe_allow_html=True)
        else:
            st.write("Please enter a comment to analyze.")

if __name__ == "__main__":
    main()
