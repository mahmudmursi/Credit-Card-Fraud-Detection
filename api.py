from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

from src.inference import predict_transaction
from src.logger import logger


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Fraud Detection Model API",
    version="1.0"
)

# Load model
try:
    model = joblib.load("models/final_fraud_pipeline.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None


class TransactionInput(BaseModel):
    Time: float
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


@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(transaction: TransactionInput):

    try:
        if model is None:
            raise HTTPException(
                status_code=500,
                detail="Model not loaded"
            )

        input_data = transaction.model_dump()
        input_df = pd.DataFrame([input_data])

        result = predict_transaction(
            model,
            input_df
        )

        logger.info(
            f"Prediction made | "
            f"Amount={transaction.Amount} | "
            f"Prediction={result['prediction']} | "
            f"Probability={result['fraud_probability']}"
        )

        return result

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )