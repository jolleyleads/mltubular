from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from features import engineer_features
import load_data

# Initialize FastAPI app
app = FastAPI(title="ML Model API")

# Load the trained model
model = joblib.load("model.pkl")

# Define request structure
class PredictRequest(BaseModel):
    data: list  # list of dictionaries, each representing a row

# Define response structure
class PredictResponse(BaseModel):
    predictions: list

# Prediction endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Convert input list of dicts to DataFrame
    df = pd.DataFrame(request.data)
    
    # Apply feature engineering
    df = engineer_features(df)
    
    # Predict
    preds = model.predict(df)
    
    # Return predictions
    return {"predictions": preds.tolist()}
