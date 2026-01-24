# -------------------------
# train.py
# -------------------------

# Imports
import pandas as pd
import numpy as np
import joblib
import load_data        # import the module that has your CSV loader
from features import engineer_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your CSV correctly
data = load_data.load_data('data.csv')  # now this works
# Feature engineering
data = engineer_features(data)

# Separate features and target (assume last column is target)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model in the same folder
joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")


# Print accuracy
score = model.score(X_test, y_test)
print(f"Model trained. Test accuracy: {score}")
