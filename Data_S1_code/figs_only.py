import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT="sorties"; R=json.load(open(f"{OUT}/resultats.json"))
OKABE = {"Output reduction": "#0072B2", "Fluctuation": "#E69F00", "Disconnection": "#009E73", "Unspecified": "#8C8C8C"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight"})
CLASSES=["Output reduction","Fluctuation","Disconnection"]; N_PERM=1000
lab_df=pd.read_excel(f"{OUT}/tables/donnees_etiquetees.xlsx"); lab_df=lab_df[lab_df.classe.isin(CLASSES)]
cv=pd.read_csv(f"{OUT}/tables/cv_par_pli.csv"); perm=R["permutation_test"]; per_class=R["per_class"]; imp=R["group_importance"]; best=R["best_model"]
learners=["Logistic regression","k-nearest neighbours","SVM (RBF)","Random forest","Gradient boosting"]
# Fig. 1 : schema du pipeline (sert aussi d'abstract graphique)
fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 46)
def box(x, y0, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec="#4D4D4D", lw=0.6))
    ax.text(x + w / 2, y0 + h - 2.6, title, ha="center", va="top", fontsize=8.2, weight="bold")
    ax.text(x + w / 2, y0 + h - 7.2, body, ha="center", va="top", fontsize=6.9, linespacing=1.35)
def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#4D4D4D"))
nE = R["n_events"]
box(1, 24, 22, 20, "Dispatching log", f"{nE} PV/wind load-\nshedding events\n2019 - May 2023\nfree-text cause +\nP, D, timestamps", "#F2F2F2")
box(27, 24, 22, 20, "Stage 1: detection", "normalisation\nspelling map\nmechanism lexicon\nobject binding\nplant gazetteer", "#DCEAF5")
box(53, 24, 22, 20, "Detected causes", f"reduction {R['detected_counts'].get('Output reduction',0)}\nfluctuation {R['detected_counts'].get('Fluctuation',0)}\ndisconnection {R['class_counts']['Disconnection']}\nunspecified {R['detected_counts'].get('Unspecified',0)}\nplants named {R['attribution_rate']*100:.1f}%", "#DCEAF5")
box(79, 24, 20, 20, "Audit", f"kappa vs legacy\n{R['agreement_legacy']['kappa']:.2f}\nrule ablation\nblind double-coding\nsample (n={R['audit_sample_size']})", "#F2F2F2")
box(14, 1, 26, 19, "Stage 2: features", "S1 ex ante: hour, month,\nweekday, year, same-day\nhistory\nS2 ex post: S1 + power,\nduration, technology", "#FCEBD2")
box(44, 1, 26, 19, "Nested repeated CV", "5 outer folds x 10 repeats\n3 inner folds (grid search)\n5 learners + 2 baselines\nmacro-F1, bal. acc., MCC", "#FCEBD2")
box(74, 1, 25, 19, "Validation", f"permutation tests\n(events, day blocks)\nNadeau-Bengio t-test\ntemporal hold-out\ngrouped importance", "#FCEBD2")
arrow(23.8, 34, 26.4, 34); arrow(49.8, 34, 52.4, 34); arrow(75.8, 34, 78.4, 34)
arrow(64, 23.4, 30, 20.6); arrow(40.6, 10.5, 43.4, 10.5); arrow(70.6, 10.5, 73.4, 10.5)
fig.savefig(f"{OUT}/fig/Fig1_pipeline.png"); fig.savefig(f"{OUT}/fig/graphical_abstract.png", dpi=300); plt.close(fig)

# Fig. 3 : profils des classes detectees (heure, saison, puissance)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
hrs = np.arange(0, 24)
for c in CLASSES:
    hh = (lab_df.loc[lab_df.classe == c, "t_debut"] // 60).value_counts(normalize=True).reindex(hrs, fill_value=0)
    axs[0].plot(hrs, hh.values * 100, color=OKABE[c], lw=2, label={"Output reduction": "Reduction", "Fluctuation": "Fluctuation", "Disconnection": "Disconn."}[c])
axs[0].set_xlim(6, 22); axs[0].set_xlabel("Hour of day"); axs[0].set_ylabel("Share of class events (%)")
axs[0].set_ylim(-1, 36); axs[0].legend(frameon=False, fontsize=6.5, loc="upper right", ncol=1, handlelength=1.2, borderaxespad=0.1)
rain = lab_df.groupby("classe")["saison"].apply(lambda s: s.str.startswith("Pluies").mean() * 100).reindex(CLASSES)
axs[1].bar(range(3), rain.values, color=[OKABE[c] for c in CLASSES], width=0.6)
for i, v in enumerate(rain.values): axs[1].text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7.5)
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["Reduction", "Fluctuation", "Disconn."], fontsize=7.5)
axs[1].set_ylabel("Rainy-season share (%)"); axs[1].set_ylim(0, 100)
data = [lab_df.loc[lab_df.classe == c, "puissance_MW"] for c in CLASSES]
bp = axs[2].boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", ms=2.5, mfc="#8C8C8C", mec="none"),
                    medianprops=dict(color="black", lw=1.2))
for patch, c in zip(bp["boxes"], CLASSES): patch.set_facecolor(OKABE[c]); patch.set_alpha(.85); patch.set_edgecolor("#4D4D4D")
axs[2].set_yscale("log"); axs[2].set_xticks([1, 2, 3]); axs[2].set_xticklabels(["Reduction", "Fluctuation", "Disconn."], fontsize=7.5)
axs[2].set_ylabel("Interrupted power (MW, log)")
for a, l in zip(axs, "abc"): a.text(-0.2, 1.04, f"({l})", transform=a.transAxes, fontsize=9, weight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/fig/Fig3_profils_classes.png"); plt.close(fig)

# Fig. 2 : attribution aux centrales
pm = pd.Series(R["plant_mentions"]).sort_values()
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.barh(pm.index, pm.values, color="#0072B2", height=0.62)
for i, v in enumerate(pm.values): ax.text(v + 0.8, i, str(v), va="center", fontsize=7.5)
ax.set_xlabel("Events in which the plant is named"); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig/Fig2_attribution_centrales.png"); plt.close(fig)

# Fig. 4 : performance comparee (macro-F1 par pli, S1 vs S2)
order = ["Majority (dummy)", "Stratified (dummy)"] + learners
short = {"Majority (dummy)": "Majority", "Stratified (dummy)": "Stratified", "Logistic regression": "LogReg",
         "k-nearest neighbours": "kNN", "SVM (RBF)": "SVM", "Random forest": "RF", "Gradient boosting": "GBM"}
fig, ax = plt.subplots(figsize=(7.2, 2.9))
for j, fs in enumerate(["S1", "S2"]):
    g_ = cv[cv.scheme == "grouped"]; r_ = cv[cv.scheme == "random"]
    vals = [g_[(g_.features == fs) & (g_.model == m)]["macro_F1"].values for m in order]
    pos = np.arange(len(order)) + (j - 0.5) * 0.36
    b = ax.boxplot(vals, positions=pos, widths=0.3, patch_artist=True, showfliers=False,
                   medianprops=dict(color="black", lw=1))
    for p_ in b["boxes"]: p_.set_facecolor("#56B4E9" if fs == "S1" else "#0072B2"); p_.set_edgecolor("#4D4D4D")
    rm = [r_[(r_.features == fs) & (r_.model == m)]["macro_F1"].mean() for m in order]
    ax.scatter(pos, rm, marker="D", s=16, facecolor="white", edgecolor="#D55E00", lw=1.1, zorder=5)
ax.set_xticks(range(len(order))); ax.set_xticklabels([short[m] for m in order])
ax.set_ylabel("Macro-F1 (50 date-grouped folds)")
ax.axhline(perm["S1"]["null_p95"], color="#8C8C8C", ls="--", lw=0.8)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[Patch(fc="#56B4E9", ec="#4D4D4D", label="S1 ex ante"), Patch(fc="#0072B2", ec="#4D4D4D", label="S2 ex post"),
                   Line2D([], [], marker="D", ls="", mfc="white", mec="#D55E00", label="mean, random split (leaky)")],
          frameon=False, loc="upper left", fontsize=7, ncol=3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig/Fig4_performance.png"); plt.close(fig)

# Fig. 5 : matrices de confusion normalisees (S1 et S2, meilleur modele)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
for a, fs in zip(axs, ["S1", "S2"]):
    cm = np.array(per_class[fs]["confusion_mean"]); cmn = cm / cm.sum(1, keepdims=True)
    a.imshow(cmn, cmap="Blues", vmin=0, vmax=1); a.grid(False)
    for i in range(3):
        for k in range(3):
            a.text(k, i, f"{cmn[i, k]*100:.0f}%", ha="center", va="center", fontsize=8,
                   color="white" if cmn[i, k] > .55 else "black")
    a.set_xticks(range(3)); a.set_yticks(range(3))
    a.set_xticklabels(["Reduc.", "Fluct.", "Disc."]); a.set_yticklabels(["Reduc.", "Fluct.", "Disc."])
    a.set_xlabel("Predicted"); a.set_ylabel("Detected (stage 1)")
    a.set_title(f"{fs}: {short[best[fs]]}", fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig/Fig5_confusion.png"); plt.close(fig)

# Fig. 6 : distributions nulles (evenements et journees) et importance groupee
import os
fig, axs4 = plt.subplots(1, 4, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1, 1, 0.62, 1.05]})
axs4[2].axis("off"); axs = [axs4[0], axs4[1], axs4[3]]
for a, fs in zip(axs[:2], ["S1", "S2"]):
    ev = np.load(f"{OUT}/tables/null_{fs}.npy")
    bins = np.linspace(0.22, 0.56, 35)
    pb = f"{OUT}/tables/null_blocs_{fs}.npy"
    if os.path.exists(pb):
        a.hist(np.load(pb), bins=bins, color="#0072B2", alpha=.8, label="day blocks")
    a.hist(ev, bins=bins, histtype="step", color="#333333", lw=1.0, label="events")
    a.axvline(perm[fs]["observed_macroF1"], color="#D55E00", lw=1.6, label="observed")
    a.set_title(f"{fs}: {'ex ante' if fs == 'S1' else 'ex post'}", fontsize=8.5)
    a.set_xlabel("Macro-F1"); a.set_xlim(0.22, 0.56)
axs[0].set_ylabel("Permutations")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.33, 0.0))
g = pd.DataFrame(imp["S2"]).T.sort_values("mean")
axs[2].barh(g.index, g["mean"], xerr=g["sd"], color="#0072B2", height=0.6, error_kw=dict(lw=0.6, capsize=2))
axs[2].axvline(0, color="#4D4D4D", lw=0.6); axs[2].set_xlabel("Drop in macro-F1 (S2)")
axs[2].grid(axis="y", visible=False); axs[2].tick_params(axis="y", labelsize=7)
for a, l, dx in zip(axs, "abc", [-0.3, -0.18, -0.9]): a.text(dx, 1.06, f"({l})", transform=a.transAxes, fontsize=9, weight="bold")
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.27, top=0.88, wspace=0.3)
fig.savefig(f"{OUT}/fig/Fig6_null_importance.png"); plt.close(fig)

