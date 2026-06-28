"""
Test de charge avec Locust.
1) lancer l'API :        uvicorn api:app
2) lancer Locust :       locust -f tests/locustfile.py --host http://127.0.0.1:8000
3) ouvrir http://localhost:8089  (choisir nb d'utilisateurs et montee en charge)
Pour Render : remplacer --host par l'URL https de ton service.
"""
from locust import HttpUser, task, between

ENTREE = {"date": "2026-01-15", "temperature": 4.5, "temperature_max": 8.0,
          "conso_j1": 72000, "conso_j7": 70000, "conso_moy_7j": 68000}

class Utilisateur(HttpUser):
    wait_time = between(0.1, 0.5)      # temps d'attente entre 2 requetes

    @task(3)
    def predire(self):
        self.client.post("/predict", json=ENTREE)

    @task(1)
    def sante(self):
        self.client.get("/")
