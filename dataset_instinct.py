import pandas as pd
import os

# ============================================================
# DATASET "INSTINCT" — uniquement les variables choisies à la main
# (hypothèses métier : historique de conso + température + type de jour)
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

INPUT  = os.path.join(OUT_DIR, "dataset_journalier_enrichi.csv")
OUTPUT = os.path.join(OUT_DIR, "dataset_instinct.csv")
print(f"Lecture du dataset enrichi : {INPUT}")

# Variables retenues "à l'instinct"  (modifie librement cette liste)
FEATURES_INSTINCT = [
    "Conso_J1", "Conso_J7",                        # historique de consommation
    "Conso_moy_3j", "Conso_moy_7j",                # tendance récente (moyennes mobiles)
    "Temperature", "Mois_sin", "Mois_cos",        # température du jour (proxy prévision météo J-1)
    "Jour_semaine_sin", "Jour_semaine_cos", "Est_weekend", "Est_ferie",  # nature de la période
    "Tempo",                                       # couleur Tempo (connue la veille)
    # "Prevision_J1",  # <- à activer si tu veux la version "avec prévision RTE"
    # "Mois_sin", "Mois_cos",
]
TARGET = "Consommation"

# ------------------------------------------------------------
df = pd.read_csv(INPUT)
# ---- Filtre : données à partir du 1er janvier 2014 ----
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] >= '2014-01-01'].reset_index(drop=True)
print(f"Après filtre 2014+ : {len(df):,} jours")

cols = ["Date"] + FEATURES_INSTINCT + [TARGET]
manquantes = [c for c in cols if c not in df.columns]
if manquantes:
    raise KeyError(f"Colonnes absentes du dataset enrichi : {manquantes}")

df_instinct = df[cols].copy()
df_instinct.to_csv(OUTPUT, index=False, encoding="utf-8-sig", float_format="%.2f")

print(f"Dataset instinct -> {OUTPUT}")
print(f"   Lignes   : {len(df_instinct):,}")
print(f"   Colonnes : {list(df_instinct.columns)}")