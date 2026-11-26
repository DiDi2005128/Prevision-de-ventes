import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.ui.csv_utils import lire_csv

DEMO = Path(__file__).resolve().parents[1] / "demo"


def test_dimanche_devient_lundi():
    df = lire_csv(io.StringIO("date,ca\n2022-09-18,100\n2022-09-25,200\n"))
    assert [d.date().isoformat() for d in df["week_start"]] == [
        "2022-09-12",
        "2022-09-19",
    ]


def test_colonnes_api_acceptees_et_lundis_inchanges():
    df = lire_csv(io.StringIO("week_start,revenue\n2022-09-12,100\n"))
    assert df["week_start"].iloc[0].date().isoformat() == "2022-09-12"


def test_colonnes_manquantes():
    with pytest.raises(ValueError, match="colonnes manquantes"):
        lire_csv(io.StringIO("a,b\n1,2\n"))


def test_demo_valide_de_bout_en_bout():
    df = lire_csv(DEMO / "demo_valide.csv")
    history = [
        {"week_start": d.date().isoformat(), "revenue": float(r)}
        for d, r in zip(df["week_start"], df["revenue"])
    ]
    r = TestClient(app).post("/predict", json={"history": history})
    assert r.status_code == 200
    assert r.json()["week_start"] == "2022-09-26"  # le lundi suivant le 19/09


def test_demo_invalide_refusee():
    df = lire_csv(DEMO / "demo_invalide_3_semaines.csv")
    history = [
        {"week_start": d.date().isoformat(), "revenue": float(r)}
        for d, r in zip(df["week_start"], df["revenue"])
    ]
    assert (
        TestClient(app).post("/predict", json={"history": history}).status_code == 422
    )
