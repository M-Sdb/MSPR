"""
=====================================================================
 PIPELINE DE RÉENTRAÎNEMENT — retrain.py
=====================================================================
 Réentraîne le modèle périodiquement sur les nouvelles données RTE
 et déploie automatiquement si le nouveau modèle est meilleur.

 Étapes :
   1. Réentraînement + optimisation des hyperparamètres (TimeSeriesSplit)
   2. Comparaison MAPE : model_v2 vs modele_final (prod)
   3. Si meilleur → swap automatique avec copie de sécurité
   4. Si moins bon → modèle actuel conservé

 Rotation à 3 slots (sans perte) :
   modele_final (prod) → model_backup
   model_v2 (nouveau)  → modele_final
   model_backup        → model_v2

 Lancer :
   python retrain.py
=====================================================================
"""
import os
import sys
import shutil
import itertools
import time
import warnings
import joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error

try:
    from xgboost import XGBRegressor
    XGB = True
except ImportError:
    XGB = False

# ----------------------------------------------------------------
# CHEMINS
# ----------------------------------------------------------------
HERE      = os.path.dirname(os.path.abspath(__file__))
OUT       = os.path.join(HERE, "output")
INPUT     = os.path.join(OUT, "dataset_journalier_enrichi.csv")

TARGET   = "Consommation"
FEATURES = [
    "Conso_J1", "Conso_J7", "Conso_moy_7j",
    "Temperature", "Temperature_max",
    "Jour_semaine_sin", "Jour_semaine_cos",
    "Est_weekend", "Est_ferie",
]

ANNEE_VAL = 2022
N_SPLITS  = 5

# ----------------------------------------------------------------
# GRILLES D'HYPERPARAMÈTRES
# ----------------------------------------------------------------
GRIDS = {
    "Random Forest": {
        "n_estimators":     [100, 200, 400, 600],
        "max_depth":        [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features":     ["sqrt", "log2", 0.8],
        "min_samples_split":[2, 5, 10],
    },
    "Gradient Boosting": {
        "max_iter":          [200, 400, 600],
        "max_depth":         [None, 5, 10, 15],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "min_samples_leaf":  [10, 20, 30],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    "Arbre de décision": {
        "max_depth":        [None, 5, 10, 20, 30],
        "min_samples_leaf": [1, 5, 10, 20],
        "min_samples_split":[2, 5, 10, 20],
        "max_features":     [None, "sqrt", "log2"],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 10, 15],
        "weights":     ["uniform", "distance"],
        "metric":      ["euclidean", "manhattan"],
    },
    "MLP": {
        "hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
        "alpha":              [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-3, 1e-2],
    },
}
if XGB:
    GRIDS["XGBoost"] = {
        "n_estimators":     [200, 400, 600, 800],
        "max_depth":        [3, 6, 8, 10],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha":        [0, 0.1, 1.0],
        "reg_lambda":       [1.0, 2.0, 5.0],
    }


# ----------------------------------------------------------------
# FONCTIONS UTILITAIRES
# ----------------------------------------------------------------
def build(name, params):
    if name == "Random Forest":
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    if name == "Gradient Boosting":
        return HistGradientBoostingRegressor(random_state=42, **params)
    if name == "Arbre de décision":
        return DecisionTreeRegressor(random_state=42, **params)
    if name == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(**params))
    if name == "MLP":
        return make_pipeline(StandardScaler(),
                             MLPRegressor(max_iter=1000, random_state=42, **params))
    if name == "XGBoost":
        return XGBRegressor(random_state=42, verbosity=0, **params)


def mape(y, p):
    return mean_absolute_percentage_error(y, p) * 100


def cv_mape(name, params, X, y, tscv):
    scores = []
    for tr, va in tscv.split(X):
        m = build(name, params)
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(mape(y.iloc[va], m.predict(X.iloc[va])))
    return float(np.mean(scores))


def regler(name, X, y, tscv):
    noms = list(GRIDS[name])
    best = (1e9, None)
    for combo in itertools.product(*GRIDS[name].values()):
        params = dict(zip(noms, combo))
        s = cv_mape(name, params, X, y, tscv)
        if s < best[0]:
            best = (s, params)
    return best


# ----------------------------------------------------------------
# ÉTAPE 1 : RÉENTRAÎNEMENT → model_v2.joblib
# ----------------------------------------------------------------
def retrain_and_save_v2():
    print("\n🔧 Réentraînement en cours...")

    df = pd.read_csv(INPUT)
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Annee"] = df["Date"].dt.year
    df = df.dropna(subset=FEATURES + [TARGET]).sort_values("Date").reset_index(drop=True)

    train = df[df["Annee"] <  ANNEE_VAL].reset_index(drop=True)
    val   = df[df["Annee"] == ANNEE_VAL].reset_index(drop=True)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val,   y_val   = val[FEATURES],   val[TARGET]
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # Réglage des hyperparamètres
    best_params = {}
    for name in GRIDS:
        t0 = time.time()
        s, params = regler(name, X_train, y_train, tscv)
        best_params[name] = params
        print(f"   {name:20s}  MAPE cv = {s:.3f} %   [{time.time()-t0:.0f}s]")

    # Choix sur validation
    scores_val = {}
    for name in GRIDS:
        m = build(name, best_params[name])
        m.fit(X_train, y_train)
        scores_val[name] = mape(y_val, m.predict(X_val))

    gagnant = min(scores_val, key=scores_val.get)
    params  = best_params[gagnant]
    print(f"\n   → Modèle retenu : {gagnant}  MAPE val = {scores_val[gagnant]:.3f} %")

    # Réentraînement sur train + val
    X_tv = pd.concat([X_train, X_val])
    y_tv = pd.concat([y_train, y_val])
    modele = build(gagnant, params)
    modele.fit(X_tv, y_tv)

    # Sauvegarde model_v2
    path_v2 = os.path.join(OUT, "model_v2.joblib")
    joblib.dump({
        "model":    modele,
        "features": FEATURES,
        "target":   TARGET,
        "modele":   gagnant,
        "params":   params,
        "mape_val": round(scores_val[gagnant], 4),
    }, path_v2)
    print(f"   model_v2 sauvegardé → {path_v2}")
    return scores_val[gagnant]


# ----------------------------------------------------------------
# ÉTAPE 2 : COMPARAISON MAPE prod vs v2
# ----------------------------------------------------------------
def get_mape_prod():
    path_prod = os.path.join(OUT, "modele_final.joblib")
    if not os.path.exists(path_prod):
        return 999.0
    bundle = joblib.load(path_prod)
    return bundle.get("mape_val", 999.0)


# ----------------------------------------------------------------
# ÉTAPE 3 : SWAP SÉCURISÉ (rotation à 3 slots sans perte)
# ----------------------------------------------------------------
def safe_swap_models():
    path_prod   = os.path.join(OUT, "modele_final.joblib")
    path_v2     = os.path.join(OUT, "model_v2.joblib")
    path_backup = os.path.join(OUT, "model_backup.joblib")

    tmp_prod   = path_prod   + ".tmp"
    tmp_v2     = path_v2     + ".tmp"
    tmp_backup = path_backup + ".tmp"

    if not os.path.exists(path_v2):
        raise FileNotFoundError(f"model_v2 introuvable : {path_v2}")

    # Copie horodatée de sécurité
    if os.path.exists(path_prod):
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        safety_copy = os.path.join(OUT, f"modele_final_{ts}.joblib")
        shutil.copy2(path_prod, safety_copy)
        print(f"   🛟 Copie de sécurité : {safety_copy}")

    # Phase 1 : vers .tmp
    if os.path.exists(path_prod):   os.replace(path_prod,   tmp_prod)
    if os.path.exists(path_v2):     os.replace(path_v2,     tmp_v2)
    if os.path.exists(path_backup): os.replace(path_backup, tmp_backup)

    # Phase 2 : vers destination finale
    if os.path.exists(tmp_prod):   os.replace(tmp_prod,   path_backup)
    if os.path.exists(tmp_v2):     os.replace(tmp_v2,     path_prod)
    if os.path.exists(tmp_backup): os.replace(tmp_backup, path_v2)

    print("   🔄 Swap terminé :")
    print("      modele_final (ancien) → model_backup")
    print("      model_v2 (nouveau)    → modele_final")
    print("      model_backup          → model_v2")


# ----------------------------------------------------------------
# PIPELINE PRINCIPAL
# ----------------------------------------------------------------
def run():
    print("\n" + "=" * 60)
    print("🚀 PIPELINE DE RÉENTRAÎNEMENT")
    print("=" * 60)

    # Step 1 : réentraînement
    mape_v2 = retrain_and_save_v2()

    # Step 2 : comparaison
    mape_prod = get_mape_prod()
    print(f"\n📊 Comparaison MAPE :")
    print(f"   modele_final (prod) : {mape_prod:.3f} %")
    print(f"   model_v2 (nouveau)  : {mape_v2:.3f} %")

    # Step 3 : déploiement
    if mape_v2 < mape_prod:
        print("\n✅ model_v2 est meilleur → déploiement...")
        try:
            safe_swap_models()
            print("\n🚀 DÉPLOIEMENT TERMINÉ")
        except Exception as e:
            print(f"\n⛔ DÉPLOIEMENT ANNULÉ : {e}")
    else:
        print("\n❌ model_v2 n'est pas meilleur → modèle actuel conservé")

    print("\n" + "=" * 60)
    print("🏁 PIPELINE TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    run()
