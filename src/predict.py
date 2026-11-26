import pandas as pd
import joblib
import os

# -----------------------------------------------------------------------
# Noms des colonnes attendues dans le DataFrame historique passe en entree.
# A adapter ici si les noms changent cote API, sans toucher au reste.
# -----------------------------------------------------------------------
COL_DATE = "date" # colonne date (lundi de la semaine)
COL_CA   = "ca"             # colonne chiffre d'affaires en €

# Chemin vers le modele entraine et exporte par train_model.py
CHEMIN_MODELE = os.path.join(
    os.path.dirname(__file__),   # repertoire de ce fichier (src/)
    "..",                        # on remonte a la racine du projet
    "models",
    "ridge_model.joblib"
)


def predire_semaine_suivante(df_historique):
    """
    Predit le chiffre d'affaires de la prochaine semaine a partir
    de l'historique passe en parametre.

    Parametres
    ----------
    df_historique : pd.DataFrame
        DataFrame avec au moins les colonnes `date_semaine` (datetime ou str)
        et `ca` (float). Doit contenir au minimum 4 lignes consecutives.

    Retour
    ------
    dict avec :
        - "date_prevision" : date du lundi de la semaine suivante (str "YYYY-MM-DD")
        - "ca_predit"      : montant predit en euros (float arrondi a 2 decimales)
    """

    # --- Verification : il faut au moins 4 semaines pour calculer lag_4 ---
    if len(df_historique) < 4:
        raise ValueError(
            f"L'historique contient seulement {len(df_historique)} semaine(s). "
            "Il en faut au minimum 4 pour calculer les lags 1, 2 et 4."
        )

    # On travaille sur une copie pour ne pas modifier le DataFrame original
    df = df_historique.copy()

    # Conversion de la colonne date au bon format, tri chronologique
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = df.sort_values(COL_DATE).reset_index(drop=True)
    # --- Verification que les 4 dernieres semaines sont bien consecutives (7 jours d'ecart) ---
    ecarts_jours = df[COL_DATE].iloc[-4:].diff().dropna().dt.days
    if not (ecarts_jours == 7).all():
        raise ValueError(
            "Les 4 dernieres semaines de l'historique doivent etre consecutives (sans trou)."
        )

    # --- Calcul des 3 lags a partir des 4 dernieres semaines disponibles ---
    # On prend la derniere ligne comme point de reference.
    # lag_1 = CA de la derniere semaine connue
    # lag_2 = CA d'il y a 2 semaines
    # lag_4 = CA d'il y a 4 semaines
    lag_1 = df[COL_CA].iloc[-1]
    lag_2 = df[COL_CA].iloc[-2]
    lag_4 = df[COL_CA].iloc[-4]

    # On formate les features dans le meme ordre qu'a l'entrainement
    X = [[lag_1, lag_2, lag_4]]

    # --- Chargement du modele Ridge pre-entraine ---
    chemin_absolu = os.path.normpath(CHEMIN_MODELE)
    if not os.path.exists(chemin_absolu):
        raise FileNotFoundError(
            f"Modele introuvable : {chemin_absolu}\n"
            "Lance d'abord train_model.py pour generer le fichier .joblib."
        )

    modele = joblib.load(chemin_absolu)

    # --- Prediction ---
    ca_predit = modele.predict(X)[0]
    ca_predit = round(float(ca_predit), 2)

    # --- Date de la semaine suivante : lundi apres la derniere date connue ---
    derniere_date = df[COL_DATE].iloc[-1]
    date_prevision = derniere_date + pd.Timedelta(weeks=1)

    # On s'assure que la date tombe bien un lundi (securite si le CSV n'est pas parfait)
    # 0 = lundi dans pandas / Python
    if date_prevision.weekday() != 0:
        # On recale sur le lundi suivant
        jours_avant_lundi = (7 - date_prevision.weekday()) % 7
        date_prevision = date_prevision + pd.Timedelta(days=jours_avant_lundi)

    return {
        "date_prevision": date_prevision.strftime("%Y-%m-%d"),
        "ca_predit": ca_predit,
    }


# -----------------------------------------------------------------------
# Bloc de test rapide : execute uniquement si on lance ce fichier
# directement (python src/predict.py), jamais quand il est importe.
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Exemple minimal avec des donnees fictives pour verifier que ca tourne
    donnees_test = {
        COL_DATE: ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"],
        COL_CA:   [3200.0, 3450.0, 3100.0, 3600.0],
    }
    df_exemple = pd.DataFrame(donnees_test)

    resultat = predire_semaine_suivante(df_exemple)
    print(f"Semaine prevue : {resultat['date_prevision']}")
    print(f"CA predit      : {resultat['ca_predit']} euros")
