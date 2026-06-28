import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# ============================================================
# VISUALISATION — Prédiction vs Réalité sur le jeu de test (données brutes)
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")

INPUT      = os.path.join(OUT_DIR, "dataset_train.csv")
MODEL_PATH = os.path.join(OUT_DIR, "modele_v2_optimise.joblib")

FEATURES_V2 = [
    "Conso_J1", "Conso_J7", "Conso_moy_7j",
    "Temperature", "Temperature_max",
    "Jour_semaine_sin", "Jour_semaine_cos",
    "Est_weekend", "Est_ferie",
]
TARGET = "Consommation"
ANNEE_SPLIT = 2022

# ------------------------------------------------------------
artefact = joblib.load(MODEL_PATH)
modele   = artefact["model"]

df = pd.read_csv(INPUT)
df['Date']  = pd.to_datetime(df['Date'])
df['Annee'] = df['Date'].dt.year
df = df.dropna(subset=FEATURES_V2 + [TARGET]).sort_values("Date")

# Réentraînement sur train uniquement (évite la fuite de données)
train = df[df["Annee"] < ANNEE_SPLIT]
test  = df[df["Annee"] >= ANNEE_SPLIT]
modele.fit(train[FEATURES_V2], train[TARGET])

dates   = test["Date"].values
reel_mw = test[TARGET].values
pred_mw = modele.predict(test[FEATURES_V2])

# Métriques
r2   = r2_score(reel_mw, pred_mw)
rmse = np.sqrt(mean_squared_error(reel_mw, pred_mw))
mape = mean_absolute_percentage_error(reel_mw, pred_mw) * 100
print(f"R²   : {r2:.4f}")
print(f"RMSE : {rmse:.1f} MW")
print(f"MAPE : {mape:.2f} %")

# ------------------------------------------------------------
# Graphique 1 : Série temporelle
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

axes[0].plot(dates, reel_mw, label="Réalité",    color="steelblue", linewidth=1.2)
axes[0].plot(dates, pred_mw, label="Prédiction", color="orangered", linewidth=1.0, alpha=0.8)
axes[0].set_title("Consommation électrique — Réalité vs Prédiction (jeu de test)", fontsize=14)
axes[0].set_ylabel("Consommation (MW)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].text(0.01, 0.97,
             f"R²={r2:.4f}  |  RMSE={rmse:.0f} MW  |  MAPE={mape:.2f}%",
             transform=axes[0].transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Graphique 2 : Résidus
erreur = pred_mw - reel_mw
axes[1].bar(dates, erreur, color=np.where(erreur >= 0, "orangered", "steelblue"), alpha=0.6, width=1)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Erreur journalière (Prédiction − Réalité)", fontsize=13)
axes[1].set_ylabel("Erreur (MW)")
axes[1].set_xlabel("Date")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
png_out = os.path.join(OUT_DIR, "prediction_vs_realite.png")
plt.savefig(png_out, dpi=150)
plt.show()
print(f"Graphique sauvegardé -> {png_out}")

# ------------------------------------------------------------
# Graphique 3 : Scatter
fig2, ax = plt.subplots(figsize=(7, 7))
ax.scatter(reel_mw, pred_mw, alpha=0.3, s=10, color="steelblue")
lims = [min(reel_mw.min(), pred_mw.min()), max(reel_mw.max(), pred_mw.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Prédiction parfaite")
ax.set_xlabel("Réalité (MW)")
ax.set_ylabel("Prédiction (MW)")
ax.set_title("Scatter — Réalité vs Prédiction", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95,
        f"R²={r2:.4f}\nRMSE={rmse:.0f} MW\nMAPE={mape:.2f}%",
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
scatter_out = os.path.join(OUT_DIR, "scatter_prediction_vs_realite.png")
plt.savefig(scatter_out, dpi=150)
plt.show()
print(f"Scatter sauvegardé -> {scatter_out}")