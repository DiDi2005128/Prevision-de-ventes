from fastapi import FastAPI, HTTPException

from app.api.predictor import predict_next_week
from app.api.schemas import PredictRequest, PredictResponse

app = FastAPI(title="Bakery weekly revenue forecast")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        value, week = predict_next_week(req.history)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return PredictResponse(prediction_eur=value, week_start=week)
