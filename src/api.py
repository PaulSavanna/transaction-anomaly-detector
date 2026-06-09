from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from pathlib import Path

app = FastAPI(title="Transaction Anomaly Detector")

# глобальные переменные для модели
model = None
scaler = None
threshold = None


class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    class Config:
        json_schema_extra = {
            "example": {
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018306,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
                "Amount": 149.62
            }
        }


@app.on_event("startup")
def load_model():
    global model, scaler, threshold

    model_path = Path("models/isolation_forest.pkl")
    scaler_path = Path("models/scaler.pkl")

    if model_path.exists():
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        threshold = 0.5  # TODO: save actual threshold
    else:
        print("WARNING: model files not found, /predict will fail")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        return {"error": "model not loaded"}

    features = np.array([[
        transaction.V1, transaction.V2, transaction.V3, transaction.V4,
        transaction.V5, transaction.V6, transaction.V7, transaction.V8,
        transaction.V9, transaction.V10, transaction.V11, transaction.V12,
        transaction.V13, transaction.V14, transaction.V15, transaction.V16,
        transaction.V17, transaction.V18, transaction.V19, transaction.V20,
        transaction.V21, transaction.V22, transaction.V23, transaction.V24,
        transaction.V25, transaction.V26, transaction.V27, transaction.V28,
        transaction.Amount, np.log1p(transaction.Amount),
        0  # Hour placeholder
    ]])

    features_scaled = scaler.transform(features)
    score = -model.decision_function(features_scaled)[0]
    is_anomaly = score > threshold

    return {
        "anomaly_score": float(score),
        "is_anomaly": bool(is_anomaly),
        "threshold": float(threshold)
    }
