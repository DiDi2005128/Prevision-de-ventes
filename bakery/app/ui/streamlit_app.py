import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Noms de colonnes attendus par l'API (ne pas modifier)
COL_DATE    = "week_start"
COL_REVENUE = "revenue"

# Correspondance avec nos colonnes internes (ventes_hebdo.csv)
# Modifier ici si votre CSV utilise d'autres noms
ALIASES = {
    "date_semaine": COL_DATE,
    "date":         COL_DATE,
    "ca":           COL_REVENUE,
    "chiffre_affaires": COL_REVENUE,
}

st.title("Prévision du CA de la semaine suivante")
st.caption(
    "CSV hebdomadaire accepté : colonnes `week_start`/`revenue` **ou** `date_semaine`/`ca`."
)

file = st.file_uploader("Fichier CSV", type="csv")

if file is not None:
    try:
        df = pd.read_csv(file)

        # Renommage automatique si l'on reçoit nos colonnes internes
        df = df.rename(columns=ALIASES)

        # Vérification que les deux colonnes requises sont bien présentes
        for col in (COL_DATE, COL_REVENUE):
            if col not in df.columns:
                st.error(
                    f"Colonne manquante : `{col}`. "
                    f"Colonnes trouvées : {list(df.columns)}"
                )
                st.stop()

        df[COL_DATE] = pd.to_datetime(df[COL_DATE])
        df = df.sort_values(COL_DATE).reset_index(drop=True)

        history = [
            {"week_start": d.date().isoformat(), "revenue": float(r)}
            for d, r in zip(df[COL_DATE], df[COL_REVENUE])
        ]
    except Exception as e:
        st.error(f"CSV illisible : {e}")
        st.stop()

    st.line_chart(df.set_index(COL_DATE)[COL_REVENUE])

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
