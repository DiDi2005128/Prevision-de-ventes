"""FAUSSE prédiction pour débloquer P4.
Quand P2 livre : remplacer uniquement le corps de predict_next_week()
(charger model.joblib, construire les retards 1, 2, 4, appeler Ridge)."""
from datetime import date, timedelta

from app.api.schemas import WeekPoint


def predict_next_week(history: list[WeekPoint]) -> tuple[float, date]:
    last4 = [p.revenue for p in history[-4:]]
    value = round(sum(last4) / len(last4), 2)
    next_week = history[-1].week_start + timedelta(days=7)
    return value, next_week
