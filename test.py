# test.py
import unittest
import joblib
import pickle
from Score import *
from Prepare import *
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
import time
from app import *
import requests
import subprocess
class TestScoreFunction(unittest.TestCase):
    def setUp(self):
        # Load the pretrained model
        with open('naive_bayes_model.pkl', 'rb') as file:
            trained_model = pickle.load(file)
        self.model = trained_model

    def test_smoke_test(self):
        # Smoke test: function should produce some output without crashing
        text = "This is a test text."
        prediction, propensity = score(text, self.model, 0.5)
        self.assertIsInstance(prediction, bool)
        self.assertIsInstance(propensity, float)

    def test_format_test(self):
        # Format test: check input/output formats/types
        text = "This is another test text."
        prediction, propensity = score(text, self.model, 0.5)
        self.assertIsInstance(prediction, bool)
        self.assertIsInstance(propensity, float)

    def test_threshold_zero(self):
        # If threshold is 0, prediction should always be 1
        text = "Spam text"
        prediction, propensity = score(text, self.model, 0)
        self.assertTrue(prediction)
        self.assertGreater(propensity, 0)

    def test_threshold_one(self):
        # If threshold is 1, prediction should always be 0
        text = "Non-spam text"
        prediction, propensity = score(text, self.model, 1)
        self.assertFalse(prediction)
        self.assertLess(propensity, 1)

    def test_spam_input(self):
        # On obvious spam input, prediction should be 1
        text = "Get rich quick! Click here!"
        prediction, propensity = score(text, self.model, 0.5)
        self.assertTrue(prediction)

    def test_non_spam_input(self):
        # On obvious non-spam input, prediction should be 0
        text = "Hello, how are you?"
        prediction, propensity = score(text, self.model, 0.5)
        self.assertFalse(prediction)

import os
import requests
import pytest
import subprocess
import time

@pytest.fixture
def flask_server():
    # Launch Flask app in a subprocess
    process = subprocess.Popen(['python', 'app.py'])

    # Wait for the server to start
    time.sleep(1)

    yield

    # Terminate the server process
    process.terminate()

def test_flask(flask_server):
    # Test the endpoint
    url = 'http://localhost:5000/score'  # Update the URL to match the endpoint
    data = "URGENT: Exclusive Insurance Offer! Act Now to Secure Your Future! Limited Time Only: Claim Your Policy Before It's Too Late! Don't Miss Out on This Life-Saving Opportunity!"
    response = requests.post(url, json=data)
    result = response.json()

    assert 'prediction' in result
    assert 'propensity' in result




if __name__ == "__main__":
    unittest.main()
    