import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def evaluate_models(data):
    X = data.drop('churn', axis=1)
    y = data['churn']

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the models
    lr = joblib.load('logistic_model.joblib')
    rf = joblib.load('random_forest_model.joblib')
    gb = joblib.load('gradient_boosting_model.joblib')

    # Make predictions
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)

    # Calculate accuracy
    lr_accuracy = accuracy_score(y_test, lr_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    gb_accuracy = accuracy_score(y_test, gb_pred)

    print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
    print(f"Gradient Boosting Accuracy: {gb_accuracy:.2f}")

if __name__ == "__main__":
    data = load_data('data.csv')
    data = engineer_features(data)
    evaluate_models(data)
