# model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import joblib

# Sample dataset
data = {
    'size': [1500, 1800, 2400, 3000, 3500],
    'bedrooms': [3, 4, 3, 5, 4],
    'location': [1, 1, 2, 2, 1],  # Example encoding for location
    'price': [400000, 500000, 600000, 650000, 700000]
}

df = pd.DataFrame(data)

# Features
X = df[['size', 'bedrooms', 'location']]
y = df['price']

# Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Polynomial Regression model
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Save models and poly transformer
joblib.dump(linear_model, 'model_linear.pkl')
joblib.dump(poly_model, 'model_poly.pkl')
joblib.dump(poly, 'poly_transformer.pkl')

print("Models are saved successfully!")
