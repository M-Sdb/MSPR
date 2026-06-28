import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

INPUT       = os.path.join(OUT_DIR, "dataset_journalier_enrichi.csv")
OUTPUT_RAW  = os.path.join(OUT_DIR, "dataset_train.csv")
OUTPUT_NORM = os.path.join(OUT_DIR, "dataset_train_normalise.csv")
print(f"Lecture du dataset enrichi : {INPUT}")

FEATURES_V2 = [
    "Conso_J1", "Conso_J7", "Conso_moy_7j",
    "Temperature", "Temperature_max",
    "Jour_semaine_sin", "Jour_semaine_cos",
    "Est_weekend", "Est_ferie",
]
TARGET = "Consommation"

# ------------------------------------------------------------
df = pd.read_csv(INPUT)

# ---- Filtre : données à partir du 1er janvier 2014 ----
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] >= '2014-01-01'].reset_index(drop=True)
print(f"Après filtre 2014+ : {len(df):,} jours")

cols = ["Date"] + FEATURES_V2 + [TARGET]
manquantes = [c for c in cols if c not in df.columns]
if manquantes:
    raise KeyError(f"Colonnes absentes du dataset enrichi : {manquantes}")

df_train = df[cols].dropna().reset_index(drop=True)

# ---- Sauvegarde version brute ----
df_train.to_csv(OUTPUT_RAW, index=False, encoding="utf-8-sig", float_format="%.2f")
print(f"Dataset train (brut)       -> {OUTPUT_RAW}")

# ---- Normalisation MinMaxScaler (on ne touche pas la colonne Date) ----
cols_a_normaliser = FEATURES_V2 + [TARGET]
scaler = MinMaxScaler()
df_train_norm = df_train.copy()
df_train_norm[cols_a_normaliser] = scaler.fit_transform(df_train[cols_a_normaliser])
df_train_norm.to_csv(OUTPUT_NORM, index=False, encoding="utf-8-sig", float_format="%.6f")
print(f"Dataset train (normalisé)  -> {OUTPUT_NORM}")

print(f"\n   Lignes   : {len(df_train):,}")
print(f"   Colonnes : {list(df_train.columns)}")