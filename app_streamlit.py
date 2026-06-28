"""
Interface de demonstration (Streamlit) — Prevision de la consommation electrique.

Pages :
  ⚡ Prédiction  — formulaire + jauge résultat
  📊 Monitoring  — dashboard des métriques du modèle en production

Lancer (l'API doit tourner en parallele) :
    1) dans un terminal :  uvicorn api:app --reload
    2) dans un autre     :  streamlit run app_streamlit.py
"""
import datetime
import os
import requests
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

API_URL = "http://127.0.0.1:8000/predict"
JOURS   = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
TEAL    = "#2C7A73"
ORANGE  = "#D2691E"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "output")

try:
    import holidays
    FERIES_FR = holidays.France()
except Exception:
    FERIES_FR = None

st.set_page_config(
    page_title="Prévision consommation électrique",
    page_icon="⚡",
    layout="wide",
)

# ----------------------------------------------------------------
# NAVIGATION
# ----------------------------------------------------------------
page = st.sidebar.radio("Navigation", ["⚡ Prédiction", "📊 Monitoring"], index=0)

# ================================================================
# PAGE 1 — PRÉDICTION
# ================================================================
if page == "⚡ Prédiction":

    st.title("⚡ Prévision de la consommation électrique")
    st.caption("Prédiction de la consommation journalière de la France (en MW)")

    col1, col2 = st.columns(2)
    with col1:
        d               = st.date_input("Jour à prédire", datetime.date(2026, 1, 15))
        temperature     = st.number_input("Température moyenne (°C)", value=4.5, step=0.5)
        temperature_max = st.number_input("Température maximale (°C)", value=8.0, step=0.5)
    with col2:
        conso_j1     = st.number_input("Consommation de la veille (MW)", value=72000, step=500)
        conso_j7     = st.number_input("Consommation il y a 7 jours (MW)", value=70000, step=500)
        conso_moy_7j = st.number_input("Moyenne des 7 derniers jours (MW)", value=68000, step=500)

    jour_nom    = JOURS[d.weekday()]
    est_weekend = d.weekday() >= 5
    est_ferie   = FERIES_FR is not None and d in FERIES_FR
    tags        = []
    if est_weekend: tags.append("week-end")
    if est_ferie:   tags.append("jour férié")
    st.write(
        f"📅 **{jour_nom.capitalize()}** {d.strftime('%d/%m/%Y')}"
        + (f" — {', '.join(tags)}" if tags else "")
    )

    if st.button("Prédire la consommation", type="primary"):
        payload = {
            "date": d.isoformat(), "temperature": temperature,
            "temperature_max": temperature_max, "conso_j1": conso_j1,
            "conso_j7": conso_j7, "conso_moy_7j": conso_moy_7j,
        }
        try:
            r = requests.post(API_URL, json=payload, timeout=10)
            r.raise_for_status()
            res  = r.json()
            pred = res["consommation_prevue_MW"]

            st.metric("Consommation prévue", f"{pred:,.0f} MW".replace(",", " "))

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number={"suffix": " MW", "font": {"size": 36}},
                gauge={
                    "axis":  {"range": [30000, 95000]},
                    "bar":   {"color": TEAL},
                    "steps": [
                        {"range": [30000, 50000], "color": "#EAF3F1"},
                        {"range": [50000, 70000], "color": "#CFE6E2"},
                        {"range": [70000, 95000], "color": "#AFD6D0"},
                    ],
                },
            ))
            fig.update_layout(height=300, margin=dict(t=20, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Modèle utilisé : {res.get('modele', '—')}")

        except requests.exceptions.ConnectionError:
            st.error("Impossible de joindre l'API. Lance-la d'abord : `uvicorn api:app --reload`")
        except Exception as ex:
            st.error(f"Erreur : {ex}")

# ================================================================
# PAGE 2 — MONITORING
# ================================================================
else:
    st.title("📊 Monitoring du modèle en production")

    # Chargement modèle
    model_path = os.path.join(OUT, "modele_final.joblib")
    if not os.path.exists(model_path):
        st.error("modele_final.joblib introuvable. Lance d'abord train_models.py.")
        st.stop()

    bundle     = joblib.load(model_path)
    NOM_MODELE = bundle.get("modele", "—")
    st.caption(f"Modèle actif : **{NOM_MODELE}** | Fichier : `modele_final.joblib`")

    # Chargement CSV — normalisation des noms de colonnes
    csv_val  = os.path.join(OUT, "resultats_validation.csv")
    csv_test = os.path.join(OUT, "resultats_test.csv")

    df_val  = None
    df_test = None

    if os.path.exists(csv_val):
        df_val = pd.read_csv(csv_val, encoding="utf-8-sig")
        df_val.columns = df_val.columns.str.strip()

    if os.path.exists(csv_test):
        df_test = pd.read_csv(csv_test, encoding="utf-8-sig")
        df_test.columns = df_test.columns.str.strip()

    # ---- KPI principaux ----
    if df_test is not None:
        st.subheader("Performance finale sur le jeu de test (2023-2025)")
        row = df_test.iloc[0]

        # Noms de colonnes flexibles
        mape_val = float(row.get("MAPE_%",     row.get("MAPE_test_%", 0)))
        rmse_val = float(row.get("RMSE",       row.get("RMSE_test",   0)))
        r2_val   = float(row.get("R2",         row.get("R2_test",     0)))

        k1, k2, k3 = st.columns(3)
        k1.metric("MAPE", f"{mape_val:.2f} %",
                  delta="< 2 % ✅" if mape_val < 2 else "> 2 % ⚠️",
                  delta_color="normal" if mape_val < 2 else "inverse")
        k2.metric("R²",   f"{r2_val:.4f}")
        k3.metric("RMSE", f"{rmse_val:,.0f} MW".replace(",", " "))

    st.divider()

    # ---- Graphiques comparaison modèles ----
    if df_val is not None:
        st.subheader("Comparaison des modèles sur la validation (2022)")

        # Colonne modèle flexible (avec ou sans accent)
        col_modele = "Modele" if "Modele" in df_val.columns else "Modèle"
        col_mape   = "MAPE_validation_%" if "MAPE_validation_%" in df_val.columns else "MAPE_val_%"

        # Colonnes RMSE et R2 optionnelles
        has_rmse = "RMSE_val" in df_val.columns
        has_r2   = "R2_val"   in df_val.columns

        if has_rmse and has_r2:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(6, 4))
            axes = [axes]

        fig.patch.set_facecolor("white")

        df_plot = df_val.sort_values(col_mape)
        colors  = [ORANGE if i == 0 else TEAL for i in range(len(df_plot))]

        axes[0].barh(df_plot[col_modele], df_plot[col_mape], color=colors)
        axes[0].set_xlabel("MAPE (%)", fontsize=11)
        axes[0].set_title("MAPE — plus bas = meilleur", fontsize=11, color=TEAL)
        axes[0].spines[["top", "right"]].set_visible(False)

        if has_rmse and has_r2:
            df_r  = df_val.sort_values("RMSE_val")
            c_r   = [ORANGE if i == 0 else TEAL for i in range(len(df_r))]
            axes[1].barh(df_r[col_modele], df_r["RMSE_val"], color=c_r)
            axes[1].set_xlabel("RMSE (MW)", fontsize=11)
            axes[1].set_title("RMSE — plus bas = meilleur", fontsize=11, color=TEAL)
            axes[1].spines[["top", "right"]].set_visible(False)

            df_r2 = df_val.sort_values("R2_val", ascending=False)
            c_r2  = [ORANGE if i == 0 else TEAL for i in range(len(df_r2))]
            axes[2].barh(df_r2[col_modele], df_r2["R2_val"], color=c_r2)
            axes[2].set_xlabel("R²", fontsize=11)
            axes[2].set_title("R² — plus haut = meilleur", fontsize=11, color=TEAL)
            axes[2].spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ---- Tableau seuils ----
    st.subheader("Seuils de surveillance en production")
    seuils = pd.DataFrame({
        "Métrique":          ["MAPE", "RMSE", "R²", "Temps de réponse"],
        "Seuil d'alerte":    ["> 3 %", "> 2 000 MW", "< 0.95", "> 500 ms"],
        "Action si dépassé": ["Déclencher retrain.py", "Déclencher retrain.py",
                              "Vérifier les données",  "Vérifier le serveur"],
    })
    st.dataframe(seuils, use_container_width=True, hide_index=True)

    st.divider()

    # ---- Infos modèle ----
    st.subheader("Informations sur le modèle chargé")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Nom du modèle :** {NOM_MODELE}")
    c1.markdown(f"**Nombre de features :** {len(bundle['features'])}")
    c2.markdown("**Features utilisées :**")
    c2.code("\n".join(bundle["features"]))