"""
=====================================================================
 PIPELINE — Prevision de la consommation electrique journaliere
=====================================================================
 4 etapes, avec des donnees DIFFERENTES pour regler et pour choisir
 (ce qui evite le biais d'optimisation / winner bias) :

   1. TRAIN (2014-2021)
        -> on REGLE les hyperparametres de chaque modele,
           par validation croisee temporelle (TimeSeriesSplit) DANS le train.

   2. VALIDATION (2022)
        -> on CHOISIT le meilleur modele : on compare les modeles regles
           sur cette annee, qui n'a PAS servi au reglage. (pas de winner bias)

   3. TEST (2023-2025)
        -> SCORE FINAL, une seule fois, sur des annees jamais vues.

   4. PRODUCTION (2014-2025)
        -> on reentraine le gagnant sur TOUT et on sauvegarde le .joblib.

 Reglage = sur le train | Choix = sur la validation | Note = sur le test

 Sorties (dossier output/) :
   resultats_validation.csv | resultats_test.csv
   modele_final.joblib      | realite_vs_prediction.png
=====================================================================
"""
import os
import time
import itertools
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

try:
    from xgboost import XGBRegressor
    XGB = True
except ImportError:
    XGB = False
    print("(XGBoost non installe -> ignore. pip install xgboost)\n")

# ------------------------------------------------------------------ #
#  CONFIGURATION                                                       #
# ------------------------------------------------------------------ #
HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "output"); os.makedirs(OUT, exist_ok=True)
INPUT = os.path.join(OUT, "dataset_train.csv")

TARGET   = "Consommation"
FEATURES = ["Conso_J1", "Conso_J7", "Conso_moy_7j", "Temperature", "Temperature_max",
            "Jour_semaine_sin", "Jour_semaine_cos", "Est_weekend", "Est_ferie"]

ANNEE_VAL       = 2022    # validation = cette annee (sert a CHOISIR)
ANNEE_TEST      = 2023    # test = cette annee et apres (sert a NOTER, 1 fois)
N_SPLITS        = 5       # plis de la validation croisee temporelle (dans le train)
FAIRE_GRAPHIQUE = True

# mois en francais (independant de la locale du PC -> marche partout)
MOIS_FR = ["janv.", "févr.", "mars", "avr.", "mai", "juin",
           "juil.", "août", "sept.", "oct.", "nov.", "déc."]

# ------------------------------------------------------------------ #
#  MODELES + grilles d'hyperparametres                                #
# ------------------------------------------------------------------ #
GRIDS = {
    "Random Forest":     {"n_estimators": [200, 400], "max_depth": [None, 15]},
    "Gradient Boosting": {"learning_rate": [0.05, 0.1], "max_iter": [300, 500], "max_depth": [4, 6]},
    "Arbre de decision": {"max_depth": [6, 10, None], "min_samples_leaf": [1, 5]},
    "KNN":               {"n_neighbors": [5, 7, 10], "weights": ["uniform", "distance"]},
    "MLP":               {"hidden_layer_sizes": [(64, 32), (128, 64)], "alpha": [1e-4, 1e-3]},
}
if XGB:
    GRIDS["XGBoost"] = {"n_estimators": [300, 500], "max_depth": [4, 6],
                        "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]}


def build(name, params):
    if name == "Random Forest":     return RandomForestRegressor(random_state=42, **params)
    if name == "Gradient Boosting": return HistGradientBoostingRegressor(random_state=42, **params)
    if name == "Arbre de decision": return DecisionTreeRegressor(random_state=42, **params)
    if name == "KNN":               return make_pipeline(StandardScaler(), KNeighborsRegressor(**params))
    if name == "MLP":               return make_pipeline(StandardScaler(), MLPRegressor(max_iter=1000, random_state=42, **params))
    if name == "XGBoost":           return XGBRegressor(random_state=42, verbosity=0, **params)


def mape(y, p):
    return mean_absolute_percentage_error(y, p) * 100


def cv_mape(name, params, X, y, tscv):
    """MAPE moyen en validation croisee temporelle (utilise pour le REGLAGE, dans le train)."""
    scores = []
    for tr, va in tscv.split(X):
        m = build(name, params)
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(mape(y.iloc[va], m.predict(X.iloc[va])))
    return float(np.mean(scores))


def regler(name, X, y, tscv):
    """Cherche les meilleurs hyperparametres -> (meilleur MAPE cv, meilleurs params)."""
    noms = list(GRIDS[name]); best = (1e9, None)
    for combo in itertools.product(*GRIDS[name].values()):
        params = dict(zip(noms, combo))
        s = cv_mape(name, params, X, y, tscv)
        if s < best[0]:
            best = (s, params)
    return best


# ================================================================== #
#  PIPELINE                                                            #
# ================================================================== #
def main():
    df = pd.read_csv(INPUT)
    df["Date"] = pd.to_datetime(df["Date"]); df["Annee"] = df["Date"].dt.year
    df = df.dropna(subset=FEATURES + [TARGET]).sort_values("Date").reset_index(drop=True)

    train = df[df["Annee"] < ANNEE_VAL].reset_index(drop=True)              # 2014-2021
    val   = df[df["Annee"] == ANNEE_VAL].reset_index(drop=True)             # 2022
    test  = df[df["Annee"] >= ANNEE_TEST].reset_index(drop=True)            # 2023-2025
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val,   y_val   = val[FEATURES],   val[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    print("=" * 62)
    print(f"  TRAIN      (2014-{ANNEE_VAL-1}) : {len(train):,} jours  ->  regler (tuning)")
    print(f"  VALIDATION ({ANNEE_VAL})        : {len(val):,} jours  ->  choisir le modele")
    print(f"  TEST       ({ANNEE_TEST}-...)   : {len(test):,} jours  ->  note finale (1 fois)")
    print("=" * 62)

    # --- 1) TRAIN : regler les hyperparametres de chaque modele ------
    print("\n[1] TRAIN - reglage des hyperparametres (TimeSeriesSplit)...\n")
    best_params = {}
    for name in GRIDS:
        t0 = time.time()
        s, params = regler(name, X_train, y_train, tscv)
        best_params[name] = params
        print(f"   {name:18s}  (MAPE cv train = {s:5.3f} %)   {params}   [{time.time()-t0:.0f}s]")

    # --- 2) VALIDATION : choisir le meilleur modele (sur 2022) -------
    print("\n[2] VALIDATION - choix du modele (sur 2022, non utilise pour le reglage)\n")
    lignes = []
    for name in GRIDS:
        m = build(name, best_params[name]); m.fit(X_train, y_train)
        lignes.append({"Modele": name, "MAPE_validation_%": round(mape(y_val, m.predict(X_val)), 3)})
    classement = pd.DataFrame(lignes).sort_values("MAPE_validation_%").reset_index(drop=True)
    classement.to_csv(os.path.join(OUT, "resultats_validation.csv"), index=False, encoding="utf-8-sig")
    print(classement.to_string(index=False))

    gagnant = classement.iloc[0]["Modele"]; params = best_params[gagnant]
    print(f"\n   --> Modele retenu : {gagnant}   {params}")

    # --- 3) TEST : score final unique (sur 2023-2025) ----------------
    print("\n[3] TEST - score final (sur des annees jamais vues)...")
    X_tv = pd.concat([X_train, X_val]); y_tv = pd.concat([y_train, y_val])   # train + validation
    modele = build(gagnant, params); modele.fit(X_tv, y_tv)
    pred = modele.predict(X_test)
    r2, rmse, mp = r2_score(y_test, pred), np.sqrt(mean_squared_error(y_test, pred)), mape(y_test, pred)
    pd.DataFrame([{"Modele": gagnant, "R2": round(r2, 3), "RMSE": round(rmse, 1),
                   "MAPE_%": round(mp, 3)}]).to_csv(
        os.path.join(OUT, "resultats_test.csv"), index=False, encoding="utf-8-sig")
    print(f"   R2 = {r2:.3f}   |   RMSE = {rmse:.0f} MW   |   MAPE = {mp:.3f} %")

    # --- 4) PRODUCTION : reentrainer sur TOUT (2014-2025) ------------
    print("\n[4] PRODUCTION - reentrainement sur toutes les donnees...")
    prod = build(gagnant, params); prod.fit(df[FEATURES], df[TARGET])
    path = os.path.join(OUT, "modele_final.joblib")
    joblib.dump({"model": prod, "features": FEATURES, "target": TARGET,
                 "modele": gagnant, "params": params}, path)
    print(f"   Sauvegarde -> {path}")

    # --- (option) courbe reel vs predit sur le test ------------------
    if FAIRE_GRAPHIQUE:
        TEAL, ORANGE = "#2C7A73", "#D2691E"

        # formate l'axe des dates en francais, sans dependre de la locale du PC
        def mois_fr(x, pos=None):
            d = mdates.num2date(x)
            return f"{MOIS_FR[d.month - 1]} {d.year}"

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(test["Date"], y_test, color=TEAL, lw=1.3, label="Réalité")
        ax.plot(test["Date"], pred,   color=ORANGE, lw=1.1, alpha=0.85, label="Prédiction")
        ax.set_title(f"Consommation réelle vs prédite sur le test ({gagnant})",
                     fontsize=13, fontweight="bold", color=TEAL)
        ax.set_xlabel("Date"); ax.set_ylabel("Consommation (MW)")
        ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(FuncFormatter(mois_fr))   # mois en francais, partout
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "realite_vs_prediction.png"), dpi=160, facecolor="white")
        plt.close()
        print("   Graphique -> realite_vs_prediction.png")

    print("\n" + "=" * 62)
    print(f"  TERMINE.  Modele final : {gagnant}   |   MAPE test : {mp:.2f} %")
    print("  (Note : le score a retenir est celui du TEST, pas de la validation.)")
    print("=" * 62)


if __name__ == "__main__":
    main()