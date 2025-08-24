# app.py

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'model.pkl' not found. Make sure the model file is in the same directory.")
    model = None

# Define the home page route, which displays the input form
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route, which handles the form submission
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html', prediction_text="Model not loaded. Please check the server logs.")

    try:
        # Get the values from the form and convert them to float
        features = [float(x) for x in request.form.values()]
        
        # Create a DataFrame for the input features
        feature_names = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']
        final_features = pd.DataFrame([features], columns=feature_names)
        
        # Make a prediction
        prediction = model.predict(final_features)

        # Determine the output text
        if prediction[0] == 1:
            output = "Result: You may have Anemia. Please consult a doctor. 🩺"
        else:
            output = "Result: You likely do not have Anemia. 👍"

        return render_template('result.html', prediction_text=output)

    except Exception as e:
        return render_template('result.html', prediction_text=f"An error occurred: {e}")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)