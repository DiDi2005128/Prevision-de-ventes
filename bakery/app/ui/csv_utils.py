import pandas as pd

# Le fichier livré par l'équipe Données utilise `date` (le dimanche de la
# semaine) et `ca`. L'API, elle, attend `week_start` et `revenue`.
RENOMMAGE = {"date": "week_start", "ca": "revenue"}


def lire_csv(f) -> pd.DataFrame:
    """Lit le CSV, harmonise les colonnes et ramène chaque date au lundi."""
    df = pd.read_csv(f).rename(columns=RENOMMAGE)
    manquantes = {"week_start", "revenue"} - set(df.columns)
    if manquantes:
        raise ValueError(f"colonnes manquantes : {', '.join(sorted(manquantes))}")
    df["week_start"] = pd.to_datetime(df["week_start"])
    # Une date au dimanche désigne la fin de la semaine : on recule de 6 jours
    # pour obtenir le lundi, car l'API raisonne en « début de semaine ».
    est_dimanche = df["week_start"].dt.dayofweek == 6
    df.loc[est_dimanche, "week_start"] -= pd.Timedelta(days=6)
    return df.sort_values("week_start")
