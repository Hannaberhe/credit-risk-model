from pydantic import BaseModel

class CustomerInput(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float

class PredictionOutput(BaseModel):
    risk_probability: float
    risk_level: str
