import os
from Prepare import preprocess_new_text
from typing import Tuple
import sklearn
import pickle
from Prepare import load_feature_names
from sklearn.feature_extraction.text import CountVectorizer
from Prepare import load_feature_names
from Prepare import final_text

feature = load_feature_names

with open('naive_bayes_model.pkl', 'rb') as file:
    trained_model = pickle.load(file)

def score(text: str, model: sklearn.base.BaseEstimator, threshold: float) -> Tuple[bool, float]:
    """
    Score a trained model on a text.

    Args:
        text (str): Input text to be scored.
        model (sklearn.base.BaseEstimator): Trained model to be used for scoring.
        threshold (float): Threshold value for classification.

    Returns:
        Tuple[bool, float]: Tuple containing prediction (bool) and propensity score (float).
    """
    # Perform preprocessing on the text if required
    # For example, tokenization, vectorization, etc.
    # Then use the trained model to make predictions
    # Example:
    # prediction = model.predict(text)
    # propensity_score = model.predict_proba(text)[1]  # Probability of being in class 1
    
    text= final_text(text)
    propensity_score = trained_model.predict_proba(text)[0][1]
    prediction = propensity_score > threshold
    return bool(prediction), propensity_score


