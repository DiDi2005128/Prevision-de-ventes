import os

import pandas as pd
import requests
import streamlit as st

from app.ui.csv_utils import lire_csv

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Prévision du CA de la semaine suivante")
st.caption(
    "CSV hebdomadaire attendu : colonnes `date` et `ca` (en €), "
    "ou `week_start` et `revenue`. Dates au dimanche ou au lundi."
)

file = st.file_uploader("Fichier CSV", type="csv")


if file is not None:
    try:
        df = lire_csv(file)
        history = [
            {"week_start": d.date().isoformat(), "revenue": float(r)}
            for d, r in zip(df["week_start"], df["revenue"])
        ]
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        st.error(f"CSV illisible : {e}")
        st.stop()

    st.line_chart(df.set_index("week_start")["revenue"])

    if st.button("Prévoir"):
        try:
            r = requests.post(
                f"{API_URL}/predict", json={"history": history}, timeout=10
            )
        except requests.RequestException:
            st.error("API injoignable.")
            st.stop()
        if r.status_code == 200:
            out = r.json()
            st.metric("CA prévu", f"{out['prediction_eur']:,.2f} €")
            st.write(f"Semaine commençant le **{out['week_start']}**")
        elif r.status_code == 422:
            msgs = [e["msg"] for e in r.json()["detail"]]
            st.error("Données refusées : " + " ; ".join(msgs))
        else:
            st.error(f"Erreur API ({r.status_code})")
