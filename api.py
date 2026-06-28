"""
=====================================================================
 API — Prevision de la consommation electrique journaliere (FastAPI)
=====================================================================
 Charge le modele entraine (output/modele_final.joblib) et expose :
   GET  /          -> etat du service
   POST /predict   -> prediction de la consommation (MW) pour un jour

 Lancer en local :  uvicorn api:app --reload
 Documentation interactive (pratique pour la demo) :  http://localhost:8000/docs
=====================================================================
"""
import os
import numpy as np
import pandas as pd
import joblib
from datetime import date as Date
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

# jours feries francais (optionnel : si la lib n'est pas installee, Est_ferie = 0)
try:
    import holidays
    FERIES_FR = holidays.France()
except Exception:
    FERIES_FR = None

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = joblib.load(os.path.join(HERE, "output", "modele_final.joblib"))
MODEL      = BUNDLE["model"]
FEATURES   = BUNDLE["features"]
NOM_MODELE = BUNDLE.get("modele", "modele")

app = FastAPI(
    title="Prévision de la consommation électrique",
    description="Prédit la consommation électrique journalière de la France (en MW).",
    version="1.0",
)


class Entree(BaseModel):
    date: str           = Field(..., examples=["2026-01-15"], description="Jour à prédire (AAAA-MM-JJ)")
    temperature: float  = Field(..., examples=[4.5],   description="Température moyenne du jour (°C)")
    temperature_max: float = Field(..., examples=[8.0], description="Température maximale du jour (°C)")
    conso_j1: float     = Field(..., examples=[72000], description="Consommation de la veille (MW)")
    conso_j7: float     = Field(..., examples=[70000], description="Consommation il y a 7 jours (MW)")
    conso_moy_7j: float = Field(..., examples=[68000], description="Moyenne des 7 derniers jours (MW)")

    @field_validator("date")
    @classmethod
    def valider_date(cls, v):
        try:
            Date.fromisoformat(v)
        except ValueError:
            raise ValueError("Date invalide : format attendu AAAA-MM-JJ")
        return v


def construire_features(e: Entree) -> pd.DataFrame:
    """A partir d'une date et de quelques valeurs, recree les 9 variables du modele."""
    d = Date.fromisoformat(e.date)
    jsem = d.weekday()  # 0 = lundi ... 6 = dimanche
    ligne = {
        "Conso_J1":         e.conso_j1,
        "Conso_J7":         e.conso_j7,
        "Conso_moy_7j":     e.conso_moy_7j,
        "Temperature":      e.temperature,
        "Temperature_max":  e.temperature_max,
        "Jour_semaine_sin": np.sin(2 * np.pi * jsem / 7),
        "Jour_semaine_cos": np.cos(2 * np.pi * jsem / 7),
        "Est_weekend":      int(jsem >= 5),
        "Est_ferie":        int(FERIES_FR is not None and d in FERIES_FR),
    }
    return pd.DataFrame([ligne])[FEATURES]   # meme ordre que l'entrainement


@app.get("/")
def sante():
    return {"status": "ok", "modele": NOM_MODELE}


@app.post("/predict")
def predict(e: Entree):
    X = construire_features(e)
    pred = float(MODEL.predict(X)[0])
    return {"date": e.date, "consommation_prevue_MW": round(pred, 1), "modele": NOM_MODELE}