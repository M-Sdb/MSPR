# ============================================================
# perm_importance_barres.py
# Un graphe a barres de permutation importance PAR modele (6 au total),
# ET un CSV par modele (importance brute en MW + importance relative).
# Sorties : analyse_output/perm_importance_barres.png
#           analyse_output/perm_importance_<modele>.csv  (un par modele)
# ============================================================
import os, re, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

TEAL = "#2C7A73"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "analyse_output"); os.makedirs(OUT, exist_ok=True)

TARGET = "Consommation"
CANDIDATES = ["Conso_J1", "Conso_J7", "Conso_moy_3j", "Conso_moy_7j",
              "Temperature", "Temperature_min", "Temperature_max",
              "Mois_sin", "Mois_cos", "Jour_semaine_sin", "Jour_semaine_cos",
              "Est_weekend", "Est_ferie", "Tempo", "Is_Covid", "Annee"]
LABELS = {'Conso_J1':"Conso J-1", 'Conso_J7':"Conso J-7", 'Conso_moy_7j':"Moyenne 7j",
 'Conso_moy_3j':"Moyenne 3j", 'Temperature':"Temperature", 'Temperature_max':"Temperature max",
 'Temperature_min':"Temperature min", 'Mois_sin':"Mois (sin)", 'Mois_cos':"Mois (cos)",
 'Jour_semaine_sin':"Jour sem. (sin)", 'Jour_semaine_cos':"Jour sem. (cos)",
 'Est_weekend':"Week-end", 'Est_ferie':"Jour ferie", 'Tempo':"Tempo", 'Is_Covid':"Covid", 'Annee':"Annee"}
ANNEE_SPLIT = 2022

def slug(s):
    s = s.lower().replace("é", "e").replace("è", "e").replace("à", "a")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")

df = pd.read_csv(os.path.join(HERE, "output", "dataset_journalier_enrichi.csv")).dropna(subset=CANDIDATES + [TARGET])
train = df[df["Annee"] < ANNEE_SPLIT]; test = df[df["Annee"] >= ANNEE_SPLIT]
Xtr, ytr, Xte, yte = train[CANDIDATES], train[TARGET], test[CANDIDATES], test[TARGET]

modeles = {
    "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
    "Arbre de decision": DecisionTreeRegressor(random_state=42),
    "KNN":               make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)),
    "MLP":               make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)),
    "XGBoost":           XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Permutation importance par modele — sur quoi chaque modele s'appuie",
             fontsize=15, fontweight="bold", color=TEAL, y=0.98)

for ax, (nom, mdl) in zip(axes.flat, modeles.items()):
    t0 = time.time(); mdl.fit(Xtr, ytr)
    r = permutation_importance(mdl, Xte, yte, n_repeats=10, random_state=42, scoring="neg_root_mean_squared_error")
    raw = pd.Series(np.clip(r.importances_mean, 0, None), index=CANDIDATES)   # hausse de RMSE (MW)
    norm = raw / raw.max() if raw.max() > 0 else raw                          # importance relative (0..1)

    # --- CSV pour CE modele ---
    tab = pd.DataFrame({
        "variable": [LABELS[i] for i in CANDIDATES],
        "importance_RMSE_MW": raw.values.round(1),
        "importance_relative": norm.values.round(3),
    }).sort_values("importance_RMSE_MW", ascending=False)
    csv_path = os.path.join(OUT, f"perm_importance_{slug(nom)}.csv")
    tab.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # --- graphe ---
    imp = norm.sort_values()
    ax.barh([LABELS[i] for i in imp.index], imp.values, color=TEAL)
    ax.set_title(nom, fontsize=12, fontweight="bold", color="#333", loc="left")
    ax.set_xlim(0, 1.05); ax.tick_params(labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel("Influence (relative)", fontsize=8)
    print(f"{nom:18s} ok ({time.time()-t0:.1f}s) -> perm_importance_{slug(nom)}.csv")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, "perm_importance_barres.png"), dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print("\n-> perm_importance_barres.png + 6 CSV (un par modele)")