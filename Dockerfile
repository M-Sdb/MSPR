# Image légère avec Python 3.11
# (3.11 évite au passage le bug de génération de schéma rencontré en Python 3.9)
FROM python:3.11-slim

WORKDIR /app

# Dépendance système requise par XGBoost (librairie OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python d'abord (cette couche est mise en cache tant que
# requirements.txt ne change pas -> reconstructions plus rapides)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'application + modèle entraîné
COPY api.py .
COPY app_streamlit.py .
COPY output/ ./output/

# Port exposé par l'API
EXPOSE 8000

# Lancement de l'API FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# VARIANTE : pour conteneuriser l'interface Streamlit à la place de l'API,
# remplace les deux dernières lignes (EXPOSE / CMD) par :
#
#   EXPOSE 8501
#   CMD ["streamlit", "run", "app_streamlit.py", \
#        "--server.address=0.0.0.0", "--server.port=8501"]
# ---------------------------------------------------------------------------
