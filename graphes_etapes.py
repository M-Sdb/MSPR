"""
graphes_etapes.py
Genere 3 visuels (pour le PPT) a partir des resultats du pipeline :
  - etape1_reglage.png    : erreur par modele apres reglage (validation croisee sur le train)
  - etape2_validation.png : erreur par modele sur la validation (choix du modele)
  - etape3_test.png       : les 3 metriques finales sur le test
Remplace les chiffres ci-dessous par ceux de TON run.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---- couleurs ----
TEAL = "#2C7A73"; INK = "#404040"; LIGHT = "#EAF3F1"; MUTE = "#7A7A7A"
OUT = os.path.dirname(os.path.abspath(__file__))

# ================== CHIFFRES (a adapter a ton run) ==================
# Etape 1 : MAPE en validation croisee sur le train (sortie [1] du pipeline)
REGLAGE = {
    "XGBoost": 1.575, "Gradient Boosting": 1.766, "Random Forest": 1.878,
    "KNN": 2.186, "Arbre de decision": 2.599, "MLP": 3.107,
}
# Etape 2 : MAPE sur la validation 2022 (sortie [2] / resultats_validation.csv)
VALIDATION = {
    "XGBoost": 1.714, "Gradient Boosting": 1.738, "MLP": 1.753,
    "Random Forest": 1.834, "KNN": 2.308, "Arbre de decision": 2.432,
}
# Etape 3 : score final sur le test (sortie [3] / resultats_test.csv)
TEST = [("R2", "0,98", "proche de 1 = tres bon"),
        ("RMSE", "1 310 MW", "erreur moyenne"),
        ("MAPE", "1,83 %", "metrique principale")]
# ===================================================================


def barres(data, titre, fichier):
    """Graphe a barres horizontales, meilleur (plus petit MAPE) en haut."""
    items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    noms = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    barres = ax.barh(noms, vals, color=TEAL)
    for b, v in zip(barres, vals):
        ax.text(v + 0.03, b.get_y() + b.get_height() / 2, f"{v:.3f} %",
                va="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel("MAPE (%)", fontsize=10)
    ax.set_title(titre, fontsize=13, fontweight="bold", color=TEAL, pad=10)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fichier), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()


def cartes_test(metriques, fichier):
    """3 cartes : R2, RMSE, MAPE."""
    fig, ax = plt.subplots(figsize=(11, 3.2)); ax.axis("off")
    fig.suptitle("Etape 3 - Score final sur le test (2023-2025)",
                 fontsize=14, fontweight="bold", color=TEAL, y=0.98)
    for i, (nom, val, sous) in enumerate(metriques):
        x = 0.04 + i * 0.32
        ax.add_patch(FancyBboxPatch((x, 0.12), 0.28, 0.66,
                     boxstyle="round,pad=0.02,rounding_size=0.04",
                     transform=ax.transAxes, facecolor=LIGHT, edgecolor="none"))
        ax.text(x + 0.14, 0.66, nom, transform=ax.transAxes, ha="center",
                fontsize=15, fontweight="bold", color=TEAL)
        ax.text(x + 0.14, 0.42, val, transform=ax.transAxes, ha="center",
                fontsize=26, fontweight="bold", color=INK)
        ax.text(x + 0.14, 0.22, sous, transform=ax.transAxes, ha="center",
                fontsize=9.5, color=MUTE)
    plt.savefig(os.path.join(OUT, fichier), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    barres(REGLAGE, "Etape 1 - Entrainement + Optimisation des hyperparametres",
           "etape1_reglage.png")
    barres(VALIDATION, "Etape 2 - Choix du modele : erreur sur la validation 2022",
           "etape2_validation.png")
    cartes_test(TEST, "etape3_test.png")
    print("3 visuels generes : etape1_reglage.png, etape2_validation.png, etape3_test.png")