"""
Test de charge — 50 requêtes simultanées sans erreur.
Lancer :  pytest Tests/test_charge.py -s
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

ENTREE = {
    "date": "2026-01-15", "temperature": 4.5, "temperature_max": 8.0,
    "conso_j1": 72000, "conso_j7": 70000, "conso_moy_7j": 68000,
}

def une_requete(_):
    r = client.post("/predict", json=ENTREE)
    return r.status_code, r.json().get("consommation_prevue_MW")

def test_charge_50_requetes():
    N = 50
    t0 = time.perf_counter()
    resultats = []

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(une_requete, i) for i in range(N)]
        for f in as_completed(futures):
            resultats.append(f.result())

    duree = time.perf_counter() - t0
    erreurs = [r for r in resultats if r[0] != 200]

    print(f"\n{N} requêtes en {duree:.2f}s — erreurs : {len(erreurs)}")
    assert len(erreurs) == 0, f"{len(erreurs)} erreur(s) sur {N} requêtes"
    assert all(20000 < r[1] < 120000 for r in resultats), "Prédiction hors plage"
