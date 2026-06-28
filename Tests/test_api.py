"""
Tests fonctionnels et de robustesse (fuzz) de l'API.
Lancer :  pytest -v
(utilise TestClient : pas besoin de lancer le serveur)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

ENTREE_VALIDE = {
    "date": "2026-01-15", "temperature": 4.5, "temperature_max": 8.0,
    "conso_j1": 72000, "conso_j7": 70000, "conso_moy_7j": 68000,
}

# ---------- fonctionnels ----------
def test_sante():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_prediction_valide():
    r = client.post("/predict", json=ENTREE_VALIDE)
    assert r.status_code == 200
    pred = r.json()["consommation_prevue_MW"]
    assert isinstance(pred, (int, float))
    assert 20000 < pred < 120000          # prediction dans une plage realiste

# ---------- robustesse / fuzz ----------
def test_champ_manquant():
    mauvais = {k: v for k, v in ENTREE_VALIDE.items() if k != "temperature"}
    r = client.post("/predict", json=mauvais)
    assert r.status_code == 422            # FastAPI rejette proprement

def test_mauvais_type():
    mauvais = {**ENTREE_VALIDE, "temperature": "froid"}
    r = client.post("/predict", json=mauvais)
    assert r.status_code == 422

def test_date_invalide():
    mauvais = {**ENTREE_VALIDE, "date": "15-01-2026"}
    r = client.post("/predict", json=mauvais)
    assert r.status_code == 422            # grace a notre validateur

def test_valeurs_extremes():
    extreme = {**ENTREE_VALIDE, "temperature": -999, "conso_j1": 0}
    r = client.post("/predict", json=extreme)
    assert r.status_code == 200            # ne crashe pas, renvoie quand meme une prediction

def test_corps_vide():
    r = client.post("/predict", json={})
    assert r.status_code == 422
