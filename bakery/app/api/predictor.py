import os
from datetime import date, timedelta
from functools import lru_cache

import joblib
import numpy as np

from app.api.schemas import WeekPoint

CHEMIN_MODELE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "ridge_model.joblib")
)


@lru_cache(maxsize=1)
def charger_modele():
    """Charge le modèle une seule fois (mis en cache), pas à chaque requête."""
    if not os.path.exists(CHEMIN_MODELE):
        raise FileNotFoundError(f"Modèle introuvable : {CHEMIN_MODELE}")
    return joblib.load(CHEMIN_MODELE)


def predict_next_week(history: list[WeekPoint]) -> tuple[float, date]:
    modele = charger_modele()

    # Mêmes variables, dans le même ordre qu'à l'entraînement (train_model.py)
    lag_1 = history[-1].revenue
    lag_2 = history[-2].revenue
    lag_4 = history[-4].revenue

    X = np.array([[lag_1, lag_2, lag_4]])
    value = round(float(modele.predict(X)[0]), 2)
    next_week = history[-1].week_start + timedelta(days=7)

    return value, next_week
