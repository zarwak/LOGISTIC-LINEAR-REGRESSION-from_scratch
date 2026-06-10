import pandas as pd
import numpy as np

# Step 1: Load the dataset
def load_data(filename):
    df = pd.read_csv(filename)
    X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
    Y = df['price']
    return X, Y

# Step 2: Preprocess the features (Min-Max scaling)
def preprocess_features(X):
    return (X - X.min()) / (X.max() - X.min())

# Step 3: Add bias term (column of 1s)
def add_bias_term(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))

# Step 4: Compute theta using Normal Equation
def compute_theta(X_b, Y):
    XTX = X_b.T.dot(X_b)
    XTY = X_b.T.dot(Y)
    return np.linalg.inv(XTX).dot(XTY)

# Step 5: Predict using learned theta
def predict(X_b, theta):
    return X_b.dot(theta)

# Step 6: Calculate Mean Squared Error
def calculate_mse(Y, Y_pred):
    return np.mean((Y - Y_pred) ** 2)

# Step 7: Main
def main():
    X, Y = load_data('housing.csv')
    X_scaled = preprocess_features(X)
    X_b = add_bias_term(X_scaled)
    theta = compute_theta(X_b, Y)
    Y_pred = predict(X_b, theta)
    mse = calculate_mse(Y, Y_pred)

    print("Theta (weights):", theta)
    print("Mean Squared Error:", mse)

main()
