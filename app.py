from flask import Flask, request, jsonify
import pickle
from Prepare import *
import os
from Score import score

app = Flask(__name__)

# Load the trained model
with open('naive_bayes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Endpoint to score text
@app.route('/score', methods=['POST'])
def score_text():
    text = request.json('text')
    text = final_text(text)
    prediction, propensity = score(text, model,0.5)

    response = {
        'prediction': int(prediction),
        'propensity': propensity

    }
    return jsonify(response)
if __name__ == '__main__':
   app.run(debug=True, host='localhost', port=5000)


