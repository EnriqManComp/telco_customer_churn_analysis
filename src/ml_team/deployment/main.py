from fastapi.middleware.cors import CORSMiddleware
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd



with open('model/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

class DataIn(BaseModel):
    features: pd.DataFrame

class PredictionOut(BaseModel):
    churn: pd.DataFrame

@app.post("/predict", response_model=PredictionOut)
def predict(payload: DataIn):
    #contents = await file.read()
    #predictions = model.predict()
    return {
        "predictions": "A"
    }