import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
R = json.load(open("sorties/resultats.json"))
cv = R["cv_summary"]; dc = R["detected_counts"]
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(13.28, 5.31), dpi=300); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 40)
def panel(x, w, title, fc):
    ax.add_patch(FancyBboxPatch((x, 3), w, 31, boxstyle="round,pad=0.3,rounding_size=1.5", fc=fc, ec="#555555", lw=1))
    ax.text(x + w / 2, 31.2, title, ha="center", va="center", fontsize=15, weight="bold")
def arrow(x0, x1):
    ax.add_patch(FancyArrowPatch((x0, 18.5), (x1, 18.5), arrowstyle="-|>", mutation_scale=28, lw=2, color="#444444"))
ax.text(50, 37.6, "Causes of PV/wind load shedding on the SENELEC grid: from dispatching text to prediction",
        ha="center", va="center", fontsize=17, weight="bold")
# panneau 1 : journal
panel(1, 22, "Dispatching log", "#F2F2F2")
lines = ["baisse PV Mekhe et Sakal", "fluctuations des centrales PV", "perturbations SOMELEC", "  et baisse PV",
         "declenchement G07 Malicounda", "..."]
for i, l in enumerate(lines):
    ax.text(3, 26.5 - i * 3.1, l, fontsize=10.5, family="DejaVu Sans Mono", color="#333333")
ax.text(12, 6.2, f"{R['n_events']} events, 2019 - May 2023", ha="center", fontsize=11.5, color="#333333")
arrow(23.6, 27.4)
# panneau 2 : etape 1
panel(28, 22, "Stage 1: rules", "#DCEAF5")
cls = [("Output reduction", dc["Output reduction"], "#0072B2"), ("Fluctuation", dc["Fluctuation"], "#E69F00"),
       ("Disconnection", dc["Renewable plant trip"] + dc["Non-renewable origin"], "#009E73"), ("Unspecified", dc["Unspecified"], "#8C8C8C")]
mx = max(c[1] for c in cls)
for i, (name, v, col) in enumerate(cls):
    y = 25.5 - i * 4.2
    ax.add_patch(plt.Rectangle((30, y), 12 * v / mx, 2.6, color=col))
    ax.text(30, y + 3.3, f"{name}  {v}", fontsize=11, va="center")
ax.text(39, 6.2, f"kappa vs legacy = {R['agreement_legacy']['kappa']:.2f}\n{R['detected_counts']['Non-renewable origin']} thermal/network events found",
        ha="center", fontsize=11, color="#333333", linespacing=1.4)
arrow(50.6, 54.4)
# panneau 3 : etape 2
panel(55, 44, "Stage 2: prediction", "#FCEBD2")
ax.text(77, 28, "random forest with power and duration, macro-F1", ha="center", fontsize=11.5, color="#333333")
bars = [("Chance\nbaseline", cv["grouped|S2|Stratified (dummy)"]["macro_F1"], "#BBBBBB"),
        ("Date-grouped\nCV", cv["grouped|S2|Random forest"]["macro_F1"], "#0072B2"),
        ("Random split\n(leaky)", cv["random|S2|Random forest"]["macro_F1"], "#D55E00"),
        ("Best learner,\n2022-2023 test", max(r["macro_F1"] for r in R["temporal"] if "dummy" not in r["model"]), "#56B4E9")]
for i, (name, v, col) in enumerate(bars):
    x = 59 + i * 9.8
    ax.add_patch(plt.Rectangle((x, 9.5), 6, 30 * v * 0.62, color=col))
    ax.text(x + 3, 9.5 + 30 * v * 0.62 + 1.0, f"{v:.2f}", ha="center", fontsize=13, weight="bold")
    ax.text(x + 3, 7.6, name, ha="center", va="top", fontsize=10.5, linespacing=1.2)
fig.savefig("sorties/fig/graphical_abstract_v2.png", dpi=300, facecolor="white")
Image.open("sorties/fig/graphical_abstract_v2.png").convert("RGB").save("sorties/fig/graphical_abstract_v2.tif", compression="tiff_lzw", dpi=(300, 300))
print(Image.open("sorties/fig/graphical_abstract_v2.tif").size)
