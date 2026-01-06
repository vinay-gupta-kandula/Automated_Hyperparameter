from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

# Create data directory
os.makedirs("data", exist_ok=True)

# Fetch dataset
X, y = fetch_california_housing(as_frame=True, return_X_y=True)

# Combine features + target
df = X.copy()
df["target"] = y

# Save clean CSV
df.to_csv("data/california_housing.csv", index=False)

print("Saved:", df.shape)
print(df.head())
