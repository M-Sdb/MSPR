"""
Test de monitoring en production.
Vérifie que les métriques du modèle restent dans les seuils acceptables.
Lancer :  pytest Tests/test_monitoring_prod.py -s
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(HERE, "output")

# Seuils d'alerte
SEUIL_MAPE = 3.0    # % — au-delà → réentraînement à déclencher
SEUIL_RMSE = 2000   # MW
SEUIL_R2   = 0.95   # en dessous → problème

def charger_donnees_test():
    bundle  = joblib.load(os.path.join(OUT, "modele_final.joblib"))
    model   = bundle["model"]
    features= bundle["features"]

    df = pd.read_csv(os.path.join(OUT, "dataset_journalier_enrichi.csv"))
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Annee"] = df["Date"].dt.year
    df = df.dropna(subset=features + ["Consommation"]).sort_values("Date")

    # Test sur 2023-2025 (données jamais vues)
    test = df[df["Annee"] >= 2023]
    X    = test[features]
    y    = test["Consommation"]
    pred = model.predict(X)
    return y.values, pred

def test_mape_en_production():
    y, pred = charger_donnees_test()
    valeur  = mean_absolute_percentage_error(y, pred) * 100
    print(f"\n   MAPE en production : {valeur:.3f} %  (seuil : {SEUIL_MAPE} %)")
    assert valeur < SEUIL_MAPE, f"⚠️  MAPE trop élevé : {valeur:.2f} % > {SEUIL_MAPE} %"

def test_rmse_en_production():
    y, pred = charger_donnees_test()
    valeur  = np.sqrt(mean_squared_error(y, pred))
    print(f"\n   RMSE en production : {valeur:.0f} MW  (seuil : {SEUIL_RMSE} MW)")
    assert valeur < SEUIL_RMSE, f"⚠️  RMSE trop élevé : {valeur:.0f} MW > {SEUIL_RMSE} MW"

def test_r2_en_production():
    y, pred = charger_donnees_test()
    valeur  = r2_score(y, pred)
    print(f"\n   R² en production : {valeur:.4f}  (seuil min : {SEUIL_R2})")
    assert valeur > SEUIL_R2, f"⚠️  R² trop bas : {valeur:.4f} < {SEUIL_R2}"

def test_rapport_monitoring():
    """Génère un rapport texte des métriques actuelles."""
    y, pred = charger_donnees_test()
    mape_v  = mean_absolute_percentage_error(y, pred) * 100
    rmse_v  = np.sqrt(mean_squared_error(y, pred))
    r2_v    = r2_score(y, pred)

    rapport = f"""
============================================================
 RAPPORT DE MONITORING — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
============================================================
 MAPE   : {mape_v:.3f} %   {'✅ OK' if mape_v < SEUIL_MAPE else '⚠️  ALERTE'}
 RMSE   : {rmse_v:.0f} MW  {'✅ OK' if rmse_v < SEUIL_RMSE else '⚠️  ALERTE'}
 R²     : {r2_v:.4f}       {'✅ OK' if r2_v > SEUIL_R2 else '⚠️  ALERTE'}
============================================================
"""
    print(rapport)

    rapport_path = os.path.join(OUT, "rapport_monitoring.txt")
    with open(rapport_path, "w", encoding="utf-8") as f:
        f.write(rapport)
    print(f"   Rapport sauvegardé → {rapport_path}")
