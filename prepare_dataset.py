import pandas as pd
import numpy as np
import glob
import os
import time
import requests
import holidays

# ============================================================
# CHARGEMENT  (read_csv comme avant — on corrige juste le chemin)
# ============================================================
# On cherche le dossier 'data' quel que soit l'endroit d'où on lance le script
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATS = [
    os.path.join(HERE, "data"),        # data à côté du script (src/data)
    os.path.join(HERE, "..", "data"),  # data à la racine du projet
    "data",                            # data dans le dossier courant
]
data_dir = next(
    (d for d in CANDIDATS if os.path.isdir(d) and glob.glob(os.path.join(d, "*.xls"))),
    None,
)
if data_dir is None:
    raise FileNotFoundError(
        "Aucun dossier 'data' contenant des .xls n'a été trouvé. "
        "Place tes fichiers RTE dans un dossier 'data' (à côté du script ou à la racine du projet)."
    )

all_xls = glob.glob(os.path.join(data_dir, "*.xls"))
files       = sorted(f for f in all_xls if "tempo" not in os.path.basename(f).lower())
tempo_files = sorted(f for f in all_xls if "tempo" in os.path.basename(f).lower())
print(f"{len(files)} fichier(s) conso + {len(tempo_files)} fichier(s) Tempo dans : {data_dir}")

# Mapping positionnel des colonnes eCO2mix (comme dans le notebook V2)
RENAME = {
    1: 'Nature',         2: 'Date',          3: 'Heure',
    4: 'Consommation',   5: 'Prevision_J1',  6: 'Prevision_J',
    7: 'Fioul',          8: 'Charbon',       9: 'Gaz',
    10: 'Nucleaire',     11: 'Eolien',       12: 'Solaire',
    13: 'Hydraulique',   14: 'Pompage',      15: 'Bioenergies',
    16: 'Ech_physiques', 17: 'Taux_Co2',
    18: 'Ech_Angleterre', 19: 'Ech_Espagne',
    20: 'Ech_Italie',     21: 'Ech_Suisse',
    22: 'Ech_Allemagne_Belgique',
}

COLS = [
    'Nature', 'Date', 'Heure',
    'Consommation', 'Prevision_J1', 'Prevision_J',
    'Nucleaire', 'Eolien', 'Solaire', 'Hydraulique',
    'Gaz', 'Fioul', 'Charbon', 'Pompage', 'Bioenergies',
    'Taux_Co2', 'Ech_physiques',
    'Ech_Angleterre', 'Ech_Espagne', 'Ech_Italie',
    'Ech_Suisse', 'Ech_Allemagne_Belgique',
]

df_list = []
for file in files:
    df = pd.read_csv(file, sep="\t", encoding="latin1", low_memory=False, index_col=False)
    df.columns = range(df.shape[1])          # colonnes positionnelles (en-têtes accentués ignorés)
    df = df.rename(columns=RENAME)
    df = df[[c for c in COLS if c in df.columns]]
    df_list.append(df)

# fusion
df = pd.concat(df_list, ignore_index=True)
print("(nombre_lignes,nombre_colonnes) = ", df.shape)

# ============================================================
# NETTOYAGE + CRÉATION D'INDICATEURS  (repris du notebook MSPR_V2)
# ============================================================

# conversion numérique de toutes les variables (hors Nature/Date/Heure)
cols_num = [c for c in COLS if c not in ['Nature', 'Date', 'Heure']]
for col in cols_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# date + suppression des quarts d'heure vides
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date', 'Consommation'])
df = df.drop_duplicates(subset=['Date', 'Heure']).sort_values('Date').reset_index(drop=True)
print("Après nettoyage :", df.shape)

# ---- Agrégation journalière (moyenne de chaque variable) ----
df_daily = df.groupby('Date').agg(
    Consommation           = ('Consommation',           'mean'),
    Prevision_J1           = ('Prevision_J1',           'mean'),
    Nucleaire              = ('Nucleaire',              'mean'),
    Eolien                 = ('Eolien',                 'mean'),
    Solaire                = ('Solaire',                'mean'),
    Hydraulique            = ('Hydraulique',            'mean'),
    Gaz                    = ('Gaz',                    'mean'),
    Fioul                  = ('Fioul',                  'mean'),
    Charbon                = ('Charbon',                'mean'),
    Pompage                = ('Pompage',                'mean'),
    Bioenergies            = ('Bioenergies',            'mean'),
    Taux_Co2               = ('Taux_Co2',               'mean'),
    Ech_physiques          = ('Ech_physiques',          'mean'),
    Ech_Angleterre         = ('Ech_Angleterre',         'mean'),
    Ech_Espagne            = ('Ech_Espagne',            'mean'),
    Ech_Italie             = ('Ech_Italie',             'mean'),
    Ech_Suisse             = ('Ech_Suisse',             'mean'),
    Ech_Allemagne_Belgique = ('Ech_Allemagne_Belgique', 'mean'),
).reset_index()

# ---- Features temporelles ----
df_daily['Annee']        = df_daily['Date'].dt.year
df_daily['Mois']         = df_daily['Date'].dt.month
df_daily['Jour_semaine'] = df_daily['Date'].dt.dayofweek
df_daily['Est_weekend']  = (df_daily['Jour_semaine'] >= 5).astype(int)
df_daily['Jour_annee']   = df_daily['Date'].dt.dayofyear

# ---- Encodage cyclique des variables calendaires (sin/cos) ----
# respecte la continuité : décembre -> janvier, dimanche -> lundi
df_daily['Mois_sin'] = np.sin(2 * np.pi * df_daily['Mois'] / 12)
df_daily['Mois_cos'] = np.cos(2 * np.pi * df_daily['Mois'] / 12)
df_daily['Jour_semaine_sin'] = np.sin(2 * np.pi * df_daily['Jour_semaine'] / 7)
df_daily['Jour_semaine_cos'] = np.cos(2 * np.pi * df_daily['Jour_semaine'] / 7)

# ---- Flag COVID (restrictions sanitaires : 17 mars 2020 -> 30 juin 2021) ----
df_daily['Is_Covid'] = (
    (df_daily['Date'] >= '2020-03-17') &
    (df_daily['Date'] <= '2021-06-30')
).astype(int)

print(f"Dataset journalier : {len(df_daily):,} jours")
print(f"   Période         : {df_daily['Date'].min().date()} -> {df_daily['Date'].max().date()}")
print(f"   Jours COVID     : {df_daily['Is_Covid'].sum()}")

# ---- Température (Open-Meteo) : moyenne PONDÉRÉE PAR LA POPULATION ----
# Moyenne sur les grandes agglomérations françaises (pondérée par leur population),
# plus représentative de la demande qu'un point unique au centre du pays.
# On prend la température du JOUR cible, assumée comme proxy d'une prévision
# météo à J-1 (disponible la veille, donc sans fuite de données).
VILLES = [
    # (nom, latitude, longitude, poids ~ population de l'aire d'attraction, en millions)
    ("Paris",      48.8566,  2.3522, 13.0),
    ("Lyon",       45.7640,  4.8357,  2.3),
    ("Marseille",  43.2965,  5.3698,  1.9),
    ("Toulouse",   43.6047,  1.4442,  1.5),
    ("Lille",      50.6292,  3.0573,  1.5),
    ("Bordeaux",   44.8378, -0.5792,  1.4),
    ("Nantes",     47.2184, -1.5536,  1.0),
    ("Strasbourg", 48.5734,  7.7521,  0.85),
]
POIDS_TOTAL = sum(v[3] for v in VILLES)
COLS_TEMP = ['Temperature', 'Temperature_min', 'Temperature_max']


def _fetch_ville(lat, lon, date_min, date_max):
    """Une requête (une ville, toute la période) avec petit retry."""
    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': lat, 'longitude': lon,
        'start_date': date_min, 'end_date': date_max,
        'daily': 'temperature_2m_mean,temperature_2m_min,temperature_2m_max',
        'timezone': 'Europe/Paris',
    }
    derniere_err = None
    for tentative in range(3):
        try:
            return requests.get(url, params=params, timeout=60).json()['daily']
        except Exception as e:
            derniere_err = e
            time.sleep(2)
    raise derniere_err


def get_temperature(year, fin_max):
    """Moyenne journalière pondérée population pour UNE année (un appel par ville)."""
    debut = f'{year}-01-01'
    fin = min(f'{year}-12-31', fin_max)
    somme = None
    for nom, lat, lon, poids in VILLES:
        d = _fetch_ville(lat, lon, debut, fin)
        v = pd.DataFrame({
            'Temperature':     d['temperature_2m_mean'],
            'Temperature_min': d['temperature_2m_min'],
            'Temperature_max': d['temperature_2m_max'],
        }, index=pd.to_datetime(d['time'])) * poids
        somme = v if somme is None else somme + v   # alignement par Date, NaN propagé
        time.sleep(0.3)                              # petite pause -> évite le rate limit
    out = somme / POIDS_TOTAL
    out.index.name = 'Date'
    return out.reset_index()


print('Récupération des températures (moyenne pondérée population)...')
# l'archive Open-Meteo a quelques jours de retard -> on ne demande pas le tout récent
fin_max = (pd.Timestamp.today() - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
dfs_temp = []
for year in sorted(df_daily['Annee'].unique()):
    if f'{year}-01-01' > fin_max:          # année entièrement dans le futur non dispo
        continue
    try:
        dfs_temp.append(get_temperature(int(year), fin_max))
        print(f'  OK {year}')
    except Exception as e:
        print(f'  !! {year} — indisponible ({e})')

if dfs_temp:
    df_temperature = pd.concat(dfs_temp, ignore_index=True)
    df_daily = df_daily.merge(df_temperature, on='Date', how='left')
    nb_temp = int(df_daily['Temperature'].notna().sum())
    print(f'  -> {nb_temp} jours avec température')
else:
    for c in COLS_TEMP:
        df_daily[c] = np.nan

# ---- Jours fériés ----
fr_holidays = holidays.France()
df_daily['Est_ferie'] = df_daily['Date'].apply(lambda x: 1 if x in fr_holidays else 0)

# ---- Calendrier Tempo (couleur du jour : BLEU=0 / BLANC=1 / ROUGE=2) ----
# La couleur du lendemain est annoncée la veille -> connue au moment de prédire (pas de fuite).
TEMPO_MAP = {"BLEU": 0, "BLANC": 1, "ROUGE": 2}
if tempo_files:
    dfs_tempo = []
    for tf in tempo_files:
        t = pd.read_csv(tf, sep="\t", encoding="latin1", header=0, usecols=[0, 1],
                        names=["Date", "Tempo_couleur"], engine="python", on_bad_lines="skip")
        dfs_tempo.append(t)
    df_tempo = pd.concat(dfs_tempo, ignore_index=True)
    df_tempo["Date"] = pd.to_datetime(df_tempo["Date"], errors="coerce")
    df_tempo["Tempo"] = df_tempo["Tempo_couleur"].astype(str).str.strip().str.upper().map(TEMPO_MAP)
    df_tempo = df_tempo.dropna(subset=["Date", "Tempo"]).drop_duplicates(subset=["Date"])
    df_daily = df_daily.merge(df_tempo[["Date", "Tempo"]], on="Date", how="left")
    nb_inconnu = int(df_daily["Tempo"].isna().sum())
    df_daily["Tempo"] = df_daily["Tempo"].fillna(-1).astype(int)   # -1 = jour hors couverture Tempo
    print(f"Tempo : {(df_daily['Tempo'] >= 0).sum()} jours avec couleur, {nb_inconnu} sans (-1)")
else:
    df_daily["Tempo"] = -1
    print("(aucun fichier Tempo trouvé -> Tempo = -1)")

# ---- Variables retardées (lags) ----
df_daily['Conso_J1'] = df_daily['Consommation'].shift(1)   # consommation d'hier
df_daily['Conso_J7'] = df_daily['Consommation'].shift(7)   # même jour semaine dernière
df_daily['Temp_J1']  = df_daily['Temperature'].shift(1)    # température d'hier (connue au moment de prédire)

# ---- Moyennes mobiles (sur les jours PASSÉS uniquement -> shift(1) avant rolling) ----
df_daily['Conso_moy_3j'] = df_daily['Consommation'].shift(1).rolling(3).mean()
df_daily['Conso_moy_7j'] = df_daily['Consommation'].shift(1).rolling(7).mean()

# ---- Sélection finale + suppression des NaN (lags / température) ----
FEATURES = [
    'Prevision_J1',
    'Conso_J1', 'Conso_J7',
    'Conso_moy_3j', 'Conso_moy_7j',
    'Temperature', 'Temperature_min', 'Temperature_max',
    'Mois_sin', 'Mois_cos', 'Jour_semaine_sin', 'Jour_semaine_cos',
    'Est_weekend', 'Est_ferie',
    'Tempo',
    'Is_Covid', 'Annee',
]
TARGET = 'Consommation'

df_daily = df_daily.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

print(f"dataset final : {df_daily.shape}")
print(f"   Période    : {df_daily['Date'].min().date()} -> {df_daily['Date'].max().date()}")
print(f"   Features   : {len(FEATURES)}")

# ============================================================
# SAUVEGARDE  (chemin absolu : à côté du script)
# ============================================================
out_dir = os.path.join(HERE, "output")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "dataset_journalier_enrichi.csv")
df_daily.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.2f")

print(f"Dataset ready -> {out_path}")
print(f"Colonnes écrites : {list(df_daily.columns)}")