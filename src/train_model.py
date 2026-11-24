import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib
import os

# -----------------------------------------------------------------------
# Noms des colonnes du CSV produit par le camarade chargé du nettoyage.
# A adapter ici si les noms changent, sans toucher au reste du script.
# -----------------------------------------------------------------------
COL_DATE = "date_semaine"   # colonne contenant le lundi de chaque semaine
COL_CA   = "ca"             # colonne chiffre d'affaires hebdomadaire en €

# Chemins de sortie
CHEMIN_MODELE    = os.path.join("models", "ridge_model.joblib")
CHEMIN_GRAPHIQUE = os.path.join("models", "comparaison_predictions.png")

# Nombre de semaines réservées pour le test (jamais vues pendant l'entraînement)
NB_SEMAINES_TEST = 8

# -----------------------------------------------------------------------
# 1. Chargement des données
# -----------------------------------------------------------------------
print("Chargement du fichier CSV hebdomadaire...")
df = pd.read_csv(os.path.join("data", "ventes_hebdo.csv"))

# On s'assure que la colonne date est bien au format datetime
df[COL_DATE] = pd.to_datetime(df[COL_DATE])

# Tri chronologique : indispensable pour que les lags soient cohérents
df = df.sort_values(COL_DATE).reset_index(drop=True)

print(f"Donnees chargees : {len(df)} semaines, du {df[COL_DATE].min().date()} au {df[COL_DATE].max().date()}")

# -----------------------------------------------------------------------
# 2. Creation des variables de retard (lags)
# lag_1 : CA de la semaine precedente
# lag_2 : CA d'il y a 2 semaines
# lag_4 : CA d'il y a 4 semaines
# -----------------------------------------------------------------------
df["lag_1"] = df[COL_CA].shift(1)
df["lag_2"] = df[COL_CA].shift(2)
df["lag_4"] = df[COL_CA].shift(4)

# On supprime les lignes avec des NaN crees par le shift (les 4 premieres)
df = df.dropna().reset_index(drop=True)

print(f"Apres creation des lags : {len(df)} semaines utilisables")

# -----------------------------------------------------------------------
# 3. Separation train / test chronologique (pas de shuffle !)
# Les 8 dernieres semaines forment le jeu de test, le reste sert a l'entrainement.
# -----------------------------------------------------------------------
df_train = df.iloc[:-NB_SEMAINES_TEST]
df_test  = df.iloc[-NB_SEMAINES_TEST:]

print(f"Train : {len(df_train)} semaines | Test : {len(df_test)} semaines")

# -----------------------------------------------------------------------
# 4. Baseline : on predit que la prochaine semaine = semaine precedente (lag 1)
# C'est le modele de reference le plus simple : "rien ne change".
# -----------------------------------------------------------------------
y_test         = df_test[COL_CA].values
y_baseline     = df_test["lag_1"].values   # prediction = semaine d-1
mae_baseline   = mean_absolute_error(y_test, y_baseline)

print(f"\nBaseline MAE : {mae_baseline:.2f} euros")

# -----------------------------------------------------------------------
# 5. Entrainement du modele Ridge sur les lags 1, 2 et 4
# Ridge est une regression lineaire avec regularisation L2 :
# elle evite le surapprentissage quand on a peu de donnees.
# -----------------------------------------------------------------------
features = ["lag_1", "lag_2", "lag_4"]

X_train = df_train[features].values
y_train = df_train[COL_CA].values
X_test  = df_test[features].values

modele = Ridge(alpha=1.0)
modele.fit(X_train, y_train)

y_ridge = modele.predict(X_test)
mae_ridge = mean_absolute_error(y_test, y_ridge)

print(f"Ridge MAE    : {mae_ridge:.2f} euros")
print(f"Gain vs baseline : {mae_baseline - mae_ridge:.2f} euros")

# -----------------------------------------------------------------------
# 6. Graphique comparatif sur les 8 semaines de test
# On sauvegarde l'image dans le dossier models/.
# -----------------------------------------------------------------------
os.makedirs("models", exist_ok=True)

dates_test = df_test[COL_DATE].values

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(dates_test, y_test,      marker="o", label="CA reel",           color="steelblue",  linewidth=2)
ax.plot(dates_test, y_baseline,  marker="s", label=f"Baseline (lag 1) — MAE {mae_baseline:.0f} €", color="coral",      linewidth=2, linestyle="--")
ax.plot(dates_test, y_ridge,     marker="^", label=f"Ridge — MAE {mae_ridge:.0f} €",           color="seagreen",   linewidth=2, linestyle="-.")

# Formatage de l'axe des dates pour plus de lisibilite
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
fig.autofmt_xdate()

ax.set_title("Comparaison des predictions sur les 8 semaines de test", fontsize=14)
ax.set_xlabel("Semaine (lundi)")
ax.set_ylabel("Chiffre d'affaires (€)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(CHEMIN_GRAPHIQUE, dpi=150)
plt.close()

print(f"\nGraphique sauvegarde : {CHEMIN_GRAPHIQUE}")

# -----------------------------------------------------------------------
# 7. Re-entrainement final sur TOUTES les donnees disponibles
# On utilise maintenant l'integralite du dataset pour que le modele
# exporte soit le plus precis possible avant la mise en production.
# -----------------------------------------------------------------------
X_all = df[features].values
y_all = df[COL_CA].values

modele_final = Ridge(alpha=1.0)
modele_final.fit(X_all, y_all)

# Export du modele avec joblib (format compact, chargeable en 1 ligne)
joblib.dump(modele_final, CHEMIN_MODELE)

print(f"Modele exporte : {CHEMIN_MODELE}")
print("\nEntrainement termine.")
