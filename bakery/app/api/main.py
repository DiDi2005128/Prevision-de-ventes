from fastapi import FastAPI

from app.api.predictor import predict_next_week
from app.api.schemas import PredictRequest, PredictResponse

app = FastAPI(title="Bakery weekly revenue forecast")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    value, week = predict_next_week(req.history)
    return PredictResponse(prediction_eur=value, week_start=week)
