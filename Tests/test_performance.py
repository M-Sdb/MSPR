"""
Test de performance : mesure la latence d'une prediction.
Lancer :  pytest tests/test_performance.py -s
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)
ENTREE = {"date": "2026-01-15", "temperature": 4.5, "temperature_max": 8.0,
          "conso_j1": 72000, "conso_j7": 70000, "conso_moy_7j": 68000}

def test_latence_moyenne():
    N = 200
    client.post("/predict", json=ENTREE)            # chauffe
    t0 = time.perf_counter()
    for _ in range(N):
        r = client.post("/predict", json=ENTREE)
        assert r.status_code == 200
    duree = time.perf_counter() - t0
    moy_ms = duree / N * 1000
    print(f"\n{N} predictions en {duree:.2f}s  ->  {moy_ms:.1f} ms / prediction  ({N/duree:.0f} req/s)")
    assert moy_ms < 200          # objectif : moins de 200 ms par prediction
