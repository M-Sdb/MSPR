# Plan de tests — API de prévision de consommation

| Type de test | Fichier / outil | Ce qu'il vérifie | Critère de réussite | Résultat |
|---|---|---|---|---|
| Fonctionnel | `test_api.py` (pytest) | l'API répond et prédit correctement | `/` = 200, `/predict` renvoie une valeur réaliste | ✅ |
| Robustesse (fuzz) | `test_api.py` (pytest) | entrées invalides : champ manquant, mauvais type, date invalide, corps vide | l'API renvoie une erreur propre (422), ne crashe pas | ✅ |
| Performance | `test_performance.py` (pytest) | latence d'une prédiction | < 200 ms / prédiction | ✅ ~4 ms |
| Charge | `locustfile.py` (Locust) | tenue sous plusieurs utilisateurs simultanés | reste stable, peu d'erreurs | à lancer |
| Monitoring | route `/` + logs / health check Render | le service est vivant et surveillé | Render détecte l'état | en prod |
| Navigation | interface Streamlit (manuel / Playwright) | parcours saisie → prédiction → affichage | chaque étape fonctionne | manuel |

## Comment lancer

### Tests fonctionnels + fuzz + performance (sans serveur)
```bash
pip install pytest httpx
pytest -v -p no:warnings
```

### Test de charge (serveur lancé)
```bash
pip install locust
uvicorn api:app                      # terminal 1
locust -f tests/locustfile.py --host http://127.0.0.1:8000   # terminal 2
# puis ouvrir http://localhost:8089
```
Pour tester le service déployé sur Render : `--host https://<ton-service>.onrender.com`

## Note Render
- Déploiement de l'image Docker en **Web Service**.
- **Health check** configuré sur `/` → Render surveille l'état du service.
- Offre gratuite : le service **s'endort** après inactivité (premier appel plus lent = cold start) → visible sur le test de performance en ligne.
