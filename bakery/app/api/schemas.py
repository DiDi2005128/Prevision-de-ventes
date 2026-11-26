from datetime import date, timedelta
from itertools import pairwise

from pydantic import BaseModel, Field, field_validator

MIN_WEEKS = 4


class WeekPoint(BaseModel):
    week_start: date
    revenue: float = Field(gt=0, description="Chiffre d'affaires de la semaine en €")


class PredictRequest(BaseModel):
    history: list[WeekPoint]

    @field_validator("history")
    @classmethod
    def check_history(cls, h: list[WeekPoint]) -> list[WeekPoint]:
        if len(h) < MIN_WEEKS:
            raise ValueError(f"Il faut au moins {MIN_WEEKS} semaines consécutives")
        for prev, cur in pairwise(h):
            if cur.week_start - prev.week_start != timedelta(days=7):
                raise ValueError(
                    "Les semaines doivent être triées et consécutives (7 jours d'écart)"
                )
        return h


class PredictResponse(BaseModel):
    prediction_eur: float
    week_start: date
