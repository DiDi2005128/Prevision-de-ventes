from datetime import date, timedelta
import os
import joblib
import numpy as np
from app.api.schemas import WeekPoint

CHEMIN_MODELE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "models",
    "ridge_model.joblib",
)


def predict_next_week(history: list[WeekPoint]) -> tuple[float, date]:
    chemin_absolu = os.path.normpath(CHEMIN_MODELE)
    if not os.path.exists(chemin_absolu):
        raise FileNotFoundError(f"Modèle introuvable : {chemin_absolu}")

    modele = joblib.load(chemin_absolu)

    lag_1 = history[-1].revenue
    lag_2 = history[-2].revenue
    lag_4 = history[-4].revenue

    X = np.array([[lag_1, lag_2, lag_4]])
    value = round(float(modele.predict(X)[0]), 2)
    next_week = history[-1].week_start + timedelta(days=7)

    return value, next_week
