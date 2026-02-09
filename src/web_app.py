from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from src.predict import predict_disease

app = FastAPI(
    title="Chronic Disease Risk API",
    version="1.0.0"
)

# -------------------------
# Health check (REQUIRED)
# -------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Chronic Disease Prediction API is running"
    }

# -------------------------
# Request schema
# -------------------------
class PatientInput(BaseModel):
    disease: str
    symptoms: Dict[str, float]
    clinical: Dict[str, float]

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
def predict(data: PatientInput):
    user_input = {
        **data.symptoms,
        **data.clinical
    }

    risk, accuracy = predict_disease(
        data.disease.lower(),
        user_input
    )

    return {
        "disease": data.disease,
        "risk_percent": float(risk),
        "model_accuracy_percent": float(accuracy) if accuracy is not None else None
    }
