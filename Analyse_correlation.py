# ============================================================
# analyse_correlation.py
# Matrice de correlation des variables du dataset final + la cible.
# Sert a justifier le choix des variables (lien avec la cible, redondances).
# Sortie : analyse_output/matrice_correlation.png (+ .csv)
# Lancer :  python analyse_correlation.py
# ============================================================
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

TEAL = "#2C7A73"; ORANGE = "#D2691E"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "analyse_output"); os.makedirs(OUT, exist_ok=True)

CAND = [os.path.join(HERE, "output", "dataset_journalier_enrichi.csv"),
        os.path.join(HERE, "dataset_journalier_enrichi.csv"),
        os.path.join(HERE, "..", "output", "dataset_journalier_enrichi.csv")]
path_csv = next((p for p in CAND if os.path.isfile(p)), None)
assert path_csv, "dataset_journalier_enrichi.csv introuvable."
df = pd.read_csv(path_csv)

# variables affichees (cible + 9 variables du modele) avec libelles lisibles
LABELS = {
 'Consommation':"Consommation (cible)",
 'Conso_J1':"Conso J-1", 'Conso_J7':"Conso J-7", 'Conso_moy_7j':"Moyenne 7j",
 'Temperature':"Temperature", 'Temperature_max':"Temperature max",
 'Jour_semaine_sin':"Jour sem. (sin)", 'Jour_semaine_cos':"Jour sem. (cos)",
 'Est_weekend':"Week-end", 'Est_ferie':"Jour ferie",
}
cols = list(LABELS.keys())
corr = df[cols].corr()

corr.rename(index=LABELS, columns=LABELS).round(2).to_csv(os.path.join(OUT, "matrice_correlation.csv"))

# colormap teal (negatif) -> blanc (0) -> orange (positif)
cmap = LinearSegmentedColormap.from_list("teal_orange", [TEAL, "#FFFFFF", ORANGE])

labels = [LABELS[c] for c in cols]
n = len(cols)
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
# valeurs dans les cases
for i in range(n):
    for j in range(n):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=8, color="white" if abs(v) > 0.55 else "#333333")
ax.set_title("Matrice de correlation des variables", fontsize=13, fontweight="bold", color=TEAL, pad=12)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Correlation (-1 a +1)", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "matrice_correlation.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close()
print("-> analyse_output/matrice_correlation.png")
print("\nCorrelations avec la cible (Consommation) :")
print(corr['Consommation'].drop('Consommation').sort_values(key=abs, ascending=False).round(2).to_string())