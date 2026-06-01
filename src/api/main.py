from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Credit Risk API")
model = joblib.load('models/best_model.pkl')

class CustomerData(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float

@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([[data.Recency, data.Frequency, data.Monetary]],
                      columns=['Recency', 'Frequency', 'Monetary'])
    prob = model.predict_proba(df)[0][1]
    risk = "High Risk" if prob > 0.5 else "Low Risk"
    return {"risk_probability": float(prob), "risk_level": risk}
