
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Print basic info
print(f"Training rows: {train_data.shape[0]}, Columns: {train_data.shape[1]}")
print(f"Test rows: {test_data.shape[0]}")

# Check missing values in selected features
important_cols = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath", "SalePrice"]
missing = train_data[important_cols].isnull().sum()
print("Missing values:\n", missing)

# Combine Full and Half baths into a new 'TotalBaths' feature
train_data["TotalBaths"] = train_data["FullBath"] + 0.5 * train_data["HalfBath"]
test_data["TotalBaths"] = test_data["FullBath"] + 0.5 * test_data["HalfBath"]

# Define predictors and target variable
features = ["GrLivArea", "BedroomAbvGr", "TotalBaths"]
X = train_data[features]
y = train_data["SalePrice"]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate on validation set
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
r2 = r2_score(y_val, y_pred)

print(f"Model RMSE: {rmse:.2f}")
print(f"Model R² Score: {r2:.3f}")

# Optional: visualize predicted vs actual prices
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.6, color='teal')
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()
