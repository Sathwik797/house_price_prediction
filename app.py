# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load saved models
linear_model = joblib.load('model_linear.pkl')
poly_model = joblib.load('model_poly.pkl')
poly_transformer = joblib.load('poly_transformer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        size = float(request.form['size'])
        bedrooms = int(request.form['bedrooms'])
        location = int(request.form['location'])  # Simple encoding for location

        features = np.array([[size, bedrooms, location]])

        # Linear regression prediction
        linear_prediction = linear_model.predict(features)

        # Polynomial regression prediction
        poly_features = poly_transformer.transform(features)
        poly_prediction = poly_model.predict(poly_features)

        return render_template('index.html', 
                               linear_prediction=linear_prediction[0], 
                               poly_prediction=poly_prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
