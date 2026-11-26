import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Prévision du CA de la semaine suivante")
st.caption("CSV hebdomadaire attendu : colonnes `week_start` et `revenue` (en €).")

file = st.file_uploader("Fichier CSV", type="csv")

if file is not None:
    try:
        df = pd.read_csv(file, parse_dates=["week_start"]).sort_values("week_start")
        history = [
            {"week_start": d.date().isoformat(), "revenue": float(r)}
            for d, r in zip(df["week_start"], df["revenue"])
        ]
    except Exception as e:
        st.error(f"CSV illisible : {e}")
        st.stop()

    st.line_chart(df.set_index("week_start")["revenue"])

    if st.button("Prévoir"):
        try:
            r = requests.post(f"{API_URL}/predict", json={"history": history}, timeout=10)
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
