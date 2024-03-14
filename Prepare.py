import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import string
import numpy as np
import numpy as np

# Load the trained model
with open('naive_bayes_model.pkl', 'rb') as file:
    trained_model = pickle.load(file)

# Define a function to preprocess new email text
def preprocess_new_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize text and remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    
    return text
def final_text(text):
    preprocessed_text = preprocess_new_text(text)
    new_vectorizer = CountVectorizer(vocabulary=feature)

    # Use the trained CountVectorizer to transform the preprocessed text into vectorized form
    vectorized_text = new_vectorizer.transform([preprocessed_text])
    return vectorized_text 



def load_feature_names(file_path="feature_names.txt"):
    # Load feature names from the file
    feature_names = np.loadtxt(file_path, dtype=str)
    return feature_names

feature = load_feature_names()
