# %% [markdown]
# # Detection et prediction des causes des delestages lies aux ENR intermittentes, reseau SENELEC
#
# Pipeline unique et re-executable qui produit **tous** les chiffres, tableaux et figures de l'article
# MethodsX. Etape 1 : detection des causes par fouille de texte a regles explicites du champ libre
# du journal de dispatching. Etape 2 : prediction supervisee de la classe de cause a partir de
# variables operationnelles, evaluee par validation croisee imbriquee repetee, modele nul par
# permutation, comparaison corrigee de Nadeau-Bengio et validation temporelle.

# %%
import os, json, re, time, platform, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
import sklearn
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, StratifiedGroupKFold,
                                     permutation_test_score)
from sklearn.metrics import (f1_score, balanced_accuracy_score, matthews_corrcoef,
                             confusion_matrix, cohen_kappa_score, recall_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, SMOTENC
import stage1_text as S1

warnings.filterwarnings("ignore")
SEED = 42
N_OUTER_SPLITS, N_REPEATS, N_INNER = 5, 10, 3
N_PERM = 1000
if os.environ.get("QUICK"):
    N_REPEATS, N_PERM, N_PERM_IMP_Q = 1, 20, 3
N_PERM_IMP = 30
OUT = "sorties"
for sub in ("fig", "tables"):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
R = {}  # registre de tous les nombres cites dans l'article
OKABE = {"Output reduction": "#0072B2", "Fluctuation": "#E69F00",
         "Disconnection": "#009E73", "Unspecified": "#8C8C8C"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E6",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})
t_start = time.time()

# %% [markdown]
# ## 1. Chargement et controle qualite

# %%
df = S1.run()
R["n_events"] = len(df)
R["period"] = [str(df["date"].min().date()), str(df["date"].max().date())]
R["events_per_year"] = df.groupby("annee").size().to_dict()

def to_minutes(x):
    s = re.sub(r"\s+", "", str(x)).replace("hh", "h")
    m = re.fullmatch(r"(\d{1,2})h(\d{2})?", s)
    return int(m.group(1)) * 60 + int(m.group(2) or 0) if m else np.nan

df["t_debut"] = df["heure debut"].map(to_minutes)
df["t_fin"] = df["heure fin"].map(to_minutes)
qc = {"heures_malformees_brutes": int(sum(not re.fullmatch(r"\s*\d{1,2}h\d{0,2}\s*", str(v))
                                          for v in pd.concat([df["heure debut"], df["heure fin"]])))}
# reparation : un horodatage illisible est reconstruit a partir de l'autre borne et de la duree
miss_d, miss_f = df["t_debut"].isna(), df["t_fin"].isna()
df.loc[miss_d, "t_debut"] = (df.loc[miss_d, "t_fin"] - df.loc[miss_d, "duree_min"]) % 1440
df.loc[miss_f, "t_fin"] = (df.loc[miss_f, "t_debut"] + df.loc[miss_f, "duree_min"]) % 1440
qc["horodatages_reconstruits"] = int(miss_d.sum() + miss_f.sum())
dur_rec = (df["t_fin"] - df["t_debut"]) % 1440
qc["durees_incoherentes"] = int((np.abs(dur_rec - df["duree_min"]) > 0.5).sum())
qc["energie_deterministe_ecart_max"] = float(np.abs(df["energie_MWh"] - df["puissance_MW"] * df["duree_min"] / 60).max())
qc["heure_colonne_vs_horodatage_ecarts"] = int((df["heure"] != (df["t_debut"] // 60)).sum())
R["qc"] = qc
print(qc)
assert qc["horodatages_reconstruits"] <= qc["heures_malformees_brutes"]

# %% [markdown]
# ## 2. Etape 1 - detection des causes par regles

# %%
R["detected_counts"] = df["cause_detectee"].value_counts().to_dict()
R["coverage_specified"] = float((df["cause_detectee"] != "Unspecified").mean())
R["n_compound"] = int(df["co_cause_non_renouvelable"].sum())
R["n_multi_mechanism"] = int(df["multi_mecanisme"].sum())
R["attribution_rate"] = float((df["n_centrales_citees"] > 0).mean())
R["n_attributed"] = int((df["n_centrales_citees"] > 0).sum())
R["n_multi_plant"] = int((df["n_centrales_citees"] > 1).sum())
pl = df["centrales_citees"].str.split("; ").explode().replace("", np.nan).dropna()
R["plant_mentions"] = pl.value_counts().to_dict()

# accord avec l'etiquetage herite (colonne cause_enr), sur une taxonomie commune a 4 modalites
legacy_map = {"Baisse production": "Output reduction", "Fluctuation (nuage/vent)": "Fluctuation",
              "Déclenchement PV": "Disconnection", "Autres": "Other/unspecified"}
new_map = {"Output reduction": "Output reduction", "Fluctuation": "Fluctuation",
           "Renewable plant trip": "Disconnection", "Non-renewable origin": "Disconnection",
           "Unspecified": "Other/unspecified"}
leg, new = df["cause_enr"].map(legacy_map), df["cause_detectee"].map(new_map)
R["agreement_legacy"] = {"kappa": float(cohen_kappa_score(leg, new)), "raw": float((leg == new).mean()),
                         "n_disagree": int((leg != new).sum())}
ct_leg = pd.crosstab(df["cause_detectee"], df["cause_enr"], margins=True)
ct_leg.to_csv(f"{OUT}/tables/T_accord_heritage.csv")
dis = df.loc[leg != new, ["causes", "cause_enr", "cause_detectee"]]
R["disagreement_breakdown"] = dis.groupby(["cause_enr", "cause_detectee"]).size().rename("n").reset_index().to_dict("records")
print(ct_leg); print(R["agreement_legacy"])

# Ablation des regles : contribution de chaque composant a l'etiquette finale
def ablate(component):
    saved = {}
    if component == "spelling":
        saved["SPELLING"] = S1.SPELLING.copy(); S1.SPELLING.clear()
    elif component == "object_binding":
        saved["re_sub"] = S1.re.sub
        orig_sub = S1.re.sub
        def no_bind(p, r, s, *a, **k):
            if isinstance(p, str) and "somelec|edm|nawec" in p:
                return s
            return orig_sub(p, r, s, *a, **k)
        S1.re.sub = no_bind
    elif component == "non_renewable_gazetteer":
        saved["NON_RE_OBJ"] = {k: v for k, v in S1.NON_RE_OBJ.items()}
        for k in S1.NON_RE_OBJ: S1.NON_RE_OBJ[k] = r"(?!x)x"
    lab = S1.run()["cause_detectee"]
    if "SPELLING" in saved: S1.SPELLING.update(saved["SPELLING"])
    if "re_sub" in saved: S1.re.sub = saved["re_sub"]
    if "NON_RE_OBJ" in saved: S1.NON_RE_OBJ.update(saved["NON_RE_OBJ"])
    return lab

abl = {}
for comp in ["spelling", "object_binding", "non_renewable_gazetteer"]:
    lab = ablate(comp)
    abl[comp] = {"n_changed": int((lab != df["cause_detectee"]).sum()),
                 "changes": (pd.crosstab(df["cause_detectee"][lab != df["cause_detectee"]],
                                         lab[lab != df["cause_detectee"]]).stack().loc[lambda s: s > 0]
                             .rename_axis(["full", "ablated"]).reset_index(name="n").to_dict("records"))}
assert (S1.run()["cause_detectee"] == df["cause_detectee"]).all(), "ablation non restauree"
R["rule_ablation"] = abl
print(json.dumps(abl, indent=1, ensure_ascii=False))

# Echantillon d'audit stratifie pour double codage humain (protocole, fichier a remplir)
rng = np.random.default_rng(SEED)
audit = pd.concat([g.sample(min(len(g), 20), random_state=SEED) for _, g in df.groupby("cause_detectee")])
audit = audit[["date", "heure debut", "centrales", "causes", "cause_detectee", "centrales_citees"]]
audit = audit.sample(frac=1, random_state=SEED).reset_index().rename(columns={"index": "id_evenement"})
audit["code_annotateur_1"] = ""; audit["code_annotateur_2"] = ""
audit.drop(columns=["cause_detectee", "centrales_citees"]).to_excel(f"{OUT}/tables/audit_double_codage_aveugle.xlsx", index=False)
audit.to_excel(f"{OUT}/tables/audit_cle_de_correction.xlsx", index=False)
R["audit_sample_size"] = len(audit)

# %% [markdown]
# ## 3. Classes cibles et coherence physique de la taxonomie

# %%
df["classe"] = df["cause_detectee"].replace({"Renewable plant trip": "Disconnection",
                                             "Non-renewable origin": "Disconnection"})
CLASSES = ["Output reduction", "Fluctuation", "Disconnection"]
lab_df = df[df["classe"].isin(CLASSES)].copy()
R["class_counts"] = lab_df["classe"].value_counts().reindex(CLASSES).to_dict()
R["n_supervised"] = len(lab_df)

desc = (lab_df.groupby("classe")
        .agg(n=("classe", "size"), P_median=("puissance_MW", "median"),
             P_p90=("puissance_MW", lambda x: x.quantile(.9)),
             D_median=("duree_min", "median"), D_p90=("duree_min", lambda x: x.quantile(.9)),
             E_total=("energie_MWh", "sum"), hour_median=("t_debut", lambda x: np.median(x) / 60),
             rainy_share=("saison", lambda x: (x.str.startswith("Pluies")).mean()))
        .reindex(CLASSES))
desc["E_share"] = desc["E_total"] / desc["E_total"].sum()
desc.round(3).to_csv(f"{OUT}/tables/T_descriptif_classes.csv")
R["class_descriptives"] = desc.round(4).to_dict("index")
tests = {}
for v in ["puissance_MW", "duree_min", "t_debut"]:
    H, p = stats.kruskal(*[lab_df.loc[lab_df.classe == c, v] for c in CLASSES])
    n = len(lab_df); k = len(CLASSES)
    tests[v] = {"H": float(H), "p": float(p), "epsilon2": float((H - k + 1) / (n - k))}
for v in ["saison", "annee"]:
    tab = pd.crosstab(lab_df["classe"], lab_df[v])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    tests[v] = {"chi2": float(chi2), "p": float(p), "dof": int(dof),
                "cramer_v": float(np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1))))}
R["taxonomy_tests"] = tests
print(desc.round(2)); print(json.dumps(tests, indent=1))

# %% [markdown]
# ## 4. Variables explicatives
# S1 (ex ante) : contexte temporel connu avant l'evenement. S2 (ex post) : S1 + grandeurs mesurees
# par le SCADA a la fin de l'evenement. L'energie, produit deterministe P x D, est exclue.

# %%
df["dt"] = df["date"] + pd.to_timedelta(df["t_debut"], unit="m")
df = df.sort_values(["dt", "t_debut"]).reset_index(drop=True)
df["n_prior_same_day"] = df.groupby("date").cumcount()
gap = df.groupby("date")["dt"].diff().dt.total_seconds() / 60
df["log_gap_prev_min"] = np.log1p(gap.fillna(1440).clip(lower=0, upper=1440))
h = df["t_debut"] / 60
df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * (df["mois"] - 1) / 12), np.cos(2 * np.pi * (df["mois"] - 1) / 12)
df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * df["jour_sem"] / 7), np.cos(2 * np.pi * df["jour_sem"] / 7)
df["year"] = df["annee"].astype(float)
df["log_power"] = np.log(df["puissance_MW"])
df["log_duration"] = np.log(df["duree_min"])
df["is_wind"] = (df["type_enr"] != "PV").astype(int)

GROUPS_S1 = {"Hour of day": ["hour_sin", "hour_cos"], "Month": ["month_sin", "month_cos"],
             "Day of week": ["dow_sin", "dow_cos"], "Year": ["year"],
             "Same-day history": ["n_prior_same_day", "log_gap_prev_min"]}
GROUPS_S2 = {**GROUPS_S1, "Interrupted power": ["log_power"], "Duration": ["log_duration"],
             "Technology (wind)": ["is_wind"]}
FEATS = {"S1": sum(GROUPS_S1.values(), []), "S2": sum(GROUPS_S2.values(), [])}
GROUPS = {"S1": GROUPS_S1, "S2": GROUPS_S2}
lab_df = df[df["classe"].isin(CLASSES)].reset_index(drop=True)
y = lab_df["classe"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
R["features"] = FEATS
R["same_day_multi_event_share"] = float((df.groupby("date").size() > 1).mean())

# %% [markdown]
# ## 5. Modeles et grilles (declarees a priori)

# %%
def sc(m): return Pipeline([("sc", StandardScaler()), ("clf", m)])
MODELS = {
    "Majority (dummy)":   (DummyClassifier(strategy="most_frequent"), {}),
    "Stratified (dummy)": (DummyClassifier(strategy="stratified", random_state=SEED), {}),
    "Logistic regression": (sc(LogisticRegression(class_weight="balanced", max_iter=5000)),
                            {"clf__C": [0.01, 0.1, 1, 10]}),
    "k-nearest neighbours": (sc(KNeighborsClassifier()),
                             {"clf__n_neighbors": [5, 11, 21], "clf__weights": ["uniform", "distance"]}),
    "SVM (RBF)": (sc(SVC(class_weight="balanced", random_state=SEED)),
                  {"clf__C": [0.3, 1, 3, 10], "clf__gamma": ["scale", 0.1]}),
    "Random forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                             random_state=SEED, n_jobs=1),
                      {"max_depth": [None, 6], "min_samples_leaf": [1, 5]}),
    "Gradient boosting": (HistGradientBoostingClassifier(class_weight="balanced", max_iter=200,
                                                         random_state=SEED),
                          {"learning_rate": [0.05, 0.1], "max_depth": [3, None]}),
}
R["grids"] = {k: {p: [str(x) for x in v] for p, v in g.items()} for k, (_, g) in MODELS.items()}

def metrics(yt, yp):
    return {"macro_F1": f1_score(yt, yp, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(yt, yp),
            "MCC": matthews_corrcoef(yt, yp)}

def fit_tuned(model, grid, X, y, seed, groups=None):
    if not grid:
        return clone(model).fit(X, y), {}
    inner = (StratifiedGroupKFold(N_INNER, shuffle=True, random_state=seed) if groups is not None
             else StratifiedKFold(N_INNER, shuffle=True, random_state=seed))
    gs = GridSearchCV(clone(model), grid, scoring="f1_macro", cv=inner, n_jobs=2)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.best_params_

# %% [markdown]
# ## 6. Validation croisee imbriquee repetee (5 plis x 10 repetitions)

# %%
groups_all = lab_df["date"].dt.strftime("%Y-%m-%d").to_numpy()
R["n_days"] = int(pd.Series(groups_all).nunique())
R["share_events_on_multi_event_days"] = float(pd.Series(groups_all).map(pd.Series(groups_all).value_counts()).gt(1).mean())
# Concordance des classes entre evenements d'une meme journee (dependance intra-jour)
same = [(a, b) for _, g in lab_df.groupby("date") for a, b in zip(g["classe"].iloc[:-1], g["classe"].iloc[1:])]
p_marg = lab_df["classe"].value_counts(normalize=True)
R["within_day_label_concordance"] = {"observed": float(np.mean([a == b for a, b in same])),
                                     "expected_if_independent": float((p_marg ** 2).sum()),
                                     "n_consecutive_pairs": len(same)}
print(R["within_day_label_concordance"], R["n_days"])

def make_splits(scheme):
    if scheme == "random":
        o = RepeatedStratifiedKFold(n_splits=N_OUTER_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
        return list(o.split(np.zeros(len(y)), y))
    sp = []
    for r in range(N_REPEATS):
        o = StratifiedGroupKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=SEED + r)
        sp += list(o.split(np.zeros(len(y)), y, groups_all))
    return sp

SCHEMES = {"grouped": make_splits("grouped"), "random": make_splits("random")}
splits = SCHEMES["grouped"]  # protocole principal
assert all(set(groups_all[tr]).isdisjoint(groups_all[te]) for tr, te in splits)
rows, oof, chosen = [], {}, {}
for scheme, spl in SCHEMES.items():
    for fs in ["S1", "S2"]:
        X = lab_df[FEATS[fs]].to_numpy(float)
        for name, (model, grid) in MODELS.items():
            pred_all = np.full((N_REPEATS, len(y)), -1)
            chosen[(scheme, fs, name)] = []
            for i, (tr, te) in enumerate(spl):
                est, bp = fit_tuned(model, grid, X[tr], y[tr], seed=SEED + i,
                                    groups=groups_all[tr] if scheme == "grouped" else None)
                yp = est.predict(X[te])
                pred_all[i // N_OUTER_SPLITS, te] = yp
                rows.append({"scheme": scheme, "features": fs, "model": name, "fold": i, "n_test": len(te),
                             **metrics(y[te], yp)})
                chosen[(scheme, fs, name)].append(json.dumps(bp, sort_keys=True, default=str))
            oof[(scheme, fs, name)] = pred_all
            print(f"{scheme:8s} {fs} {name:22s} done  {time.time() - t_start:6.0f}s")
cv = pd.DataFrame(rows)
cv.to_csv(f"{OUT}/tables/cv_par_pli.csv", index=False)
summ = cv.groupby(["scheme", "features", "model"])[["macro_F1", "balanced_accuracy", "MCC"]].agg(["mean", "std"])
summ.to_csv(f"{OUT}/tables/T_performance_cv.csv")
print(summ.round(3))
R["cv_summary"] = {f"{sc_}|{fs}|{m}": {k: float(summ.loc[(sc_, fs, m), (k, "mean")]) for k in ["macro_F1", "balanced_accuracy", "MCC"]}
                   for sc_, fs, m in summ.index}
R["cv_summary_sd"] = {f"{sc_}|{fs}|{m}": {k: float(summ.loc[(sc_, fs, m), (k, "std")]) for k in ["macro_F1", "balanced_accuracy", "MCC"]}
                      for sc_, fs, m in summ.index}
R["chosen_hyperparams"] = {"|".join(k): pd.Series(v).value_counts().to_dict() for k, v in chosen.items()}
# optimisme du decoupage aleatoire par rapport au decoupage par journee
opt = (summ.xs("random")[("macro_F1", "mean")] - summ.xs("grouped")[("macro_F1", "mean")]).rename("optimism_macroF1")
opt.to_csv(f"{OUT}/tables/T_optimisme_decoupage.csv")
R["split_optimism"] = {f"{fs}|{m}": float(v) for (fs, m), v in opt.items()}

# %% [markdown]
# ## 7. Comparaisons appariees : test t corrige de Nadeau et Bengio

# %%
def corrected_t(a, b, n_train, n_test):
    d = np.asarray(a) - np.asarray(b); J = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var) if var > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return float(d.mean()), float(t), float(p)

n_test = len(y) // N_OUTER_SPLITS; n_train = len(y) - n_test
learners = [m for m in MODELS if "dummy" not in m]
best = {}
comp_rows = []
for fs in ["S1", "S2"]:
    sub = cv[(cv.features == fs) & (cv.scheme == "grouped")]
    means = sub.groupby("model")["macro_F1"].mean()
    best[fs] = means[learners].idxmax()
    for m in MODELS:
        if m == best[fs]: continue
        a = sub[sub.model == best[fs]].sort_values("fold")["macro_F1"]
        b = sub[sub.model == m].sort_values("fold")["macro_F1"]
        dm, t, p = corrected_t(a.values, b.values, n_train, n_test)
        comp_rows.append({"features": fs, "best": best[fs], "vs": m, "delta_macroF1": dm, "t": t, "p": p})
# S2 vs S1 pour chaque apprenant
for m in learners:
    g_ = cv[cv.scheme == "grouped"]
    a = g_[(g_.features == "S2") & (g_.model == m)].sort_values("fold")["macro_F1"].values
    b = g_[(g_.features == "S1") & (g_.model == m)].sort_values("fold")["macro_F1"].values
    dm, t, p = corrected_t(a, b, n_train, n_test)
    comp_rows.append({"features": "S2 vs S1", "best": m, "vs": m, "delta_macroF1": dm, "t": t, "p": p})
comp = pd.DataFrame(comp_rows)
comp.to_csv(f"{OUT}/tables/T_comparaisons_nadeau_bengio.csv", index=False)
R["best_model"] = best
R["comparisons"] = comp.round(5).to_dict("records")
print(best); print(comp.round(4))

# %% [markdown]
# ## 8. Matrice de confusion et rappel par classe (predictions hors pli, 10 repetitions)

# %%
per_class = {}
for fs in ["S1", "S2"]:
    P = oof[("grouped", fs, best[fs])]
    cms = np.array([confusion_matrix(y, P[r], labels=range(3)) for r in range(N_REPEATS)])
    cm = cms.mean(0)
    np.savetxt(f"{OUT}/tables/confusion_{fs}.csv", cm, delimiter=",", fmt="%.1f")
    rec = np.array([recall_score(y, P[r], average=None, labels=range(3)) for r in range(N_REPEATS)])
    per_class[fs] = {c: {"recall_mean": float(rec[:, i].mean()), "recall_sd": float(rec[:, i].std(ddof=1))}
                     for i, c in enumerate(CLASSES)}
    per_class[fs]["confusion_mean"] = cm.round(2).tolist()
R["per_class"] = per_class
print(json.dumps(per_class, indent=1))

# %% [markdown]
# ## 9. Modele nul par permutation des etiquettes (1000 permutations)

# %%
def majority_params(fs, m):
    s = pd.Series(chosen[("grouped", fs, m)]).value_counts()
    return json.loads(s.index[0])

# Decoupage par journee fige, puis permutation GLOBALE des etiquettes (test 1 d'Ojala et Garriga).
# Ne pas passer `groups` a permutation_test_score : scikit-learn permuterait alors les etiquettes a
# l'interieur de chaque journee, ce qui conserve la dependance intra-jour et gonfle la distribution nulle.
perm_splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, groups_all))
perm = {}
for fs in ["S1", "S2"]:
    m = best[fs]; model, _ = MODELS[m]
    est = clone(model).set_params(**majority_params(fs, m))
    if hasattr(est, "n_jobs"): est.set_params(n_jobs=1)
    X = lab_df[FEATS[fs]].to_numpy(float)
    t0 = time.time()
    score, null, p = permutation_test_score(est, X, y, scoring="f1_macro",
                                            cv=perm_splits,
                                            n_permutations=N_PERM, random_state=SEED, n_jobs=2)
    perm[fs] = {"model": m, "params": majority_params(fs, m), "observed_macroF1": float(score),
                "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, .95)),
                "null_max": float(null.max()), "p_value": float(p), "n_perm": N_PERM,
                "p_floor": 1 / (N_PERM + 1), "seconds": round(time.time() - t0)}
    np.save(f"{OUT}/tables/null_{fs}.npy", null)
    print(fs, perm[fs])
R["permutation_test"] = perm

# %% [markdown]
# ## 10. Validation temporelle (apprentissage 2019-2021, test 2022-mai 2023)

# %%
tr_mask = lab_df["annee"] <= 2021
R["temporal_split"] = {"n_train": int(tr_mask.sum()), "n_test": int((~tr_mask).sum()),
                       "train_counts": lab_df.loc[tr_mask, "classe"].value_counts().reindex(CLASSES).to_dict(),
                       "test_counts": lab_df.loc[~tr_mask, "classe"].value_counts().reindex(CLASSES).to_dict()}
temporal = []
for fs in ["S1", "S2"]:
    X = lab_df[FEATS[fs]].to_numpy(float)
    for m, (model, grid) in MODELS.items():
        est, bp = fit_tuned(model, grid, X[tr_mask], y[tr_mask], seed=SEED, groups=groups_all[tr_mask.to_numpy()])
        yp = est.predict(X[~tr_mask])
        temporal.append({"features": fs, "model": m, **metrics(y[~tr_mask], yp),
                         "recall_disconnection": recall_score(y[~tr_mask], yp, labels=[2], average="macro")})
temporal = pd.DataFrame(temporal)
temporal.to_csv(f"{OUT}/tables/T_validation_temporelle.csv", index=False)
R["temporal"] = temporal.round(4).to_dict("records")
print(temporal.round(3))

# %% [markdown]
# ## 11. Sensibilite : sur-echantillonnage SMOTE / SMOTE-NC dans les plis d'apprentissage

# %%
smote_rows = []
for fs in ["S1", "S2"]:
    m = best[fs]; model, _ = MODELS[m]
    est = clone(model).set_params(**majority_params(fs, m))
    if "class_weight" in est.get_params():
        est.set_params(class_weight=None)
    elif "clf__class_weight" in est.get_params():
        est.set_params(clf__class_weight=None)
    X = lab_df[FEATS[fs]].to_numpy(float)
    sampler = (SMOTENC(categorical_features=[FEATS[fs].index("is_wind")], random_state=SEED, k_neighbors=5)
               if "is_wind" in FEATS[fs] else SMOTE(random_state=SEED, k_neighbors=5))
    pipe = ImbPipeline([("smote", sampler), ("model", est)])
    sc_ = [f1_score(y[te], clone(pipe).fit(X[tr], y[tr]).predict(X[te]), average="macro") for tr, te in splits]
    base = cv[(cv.scheme == "grouped") & (cv.features == fs) & (cv.model == m)].sort_values("fold")["macro_F1"].values
    dm, t, p = corrected_t(np.array(sc_), base, n_train, n_test)
    smote_rows.append({"features": fs, "model": m, "macroF1_smote": float(np.mean(sc_)),
                       "macroF1_classweight_tuned": float(base.mean()), "delta": dm, "p": p})
R["smote_sensitivity"] = smote_rows
print(pd.DataFrame(smote_rows).round(4))

# %% [markdown]
# ## 12. Importance des variables par permutation groupee, sur plis tenus a l'ecart

# %%
imp = {}
for fs in ["S1", "S2"]:
    m = best[fs]; model, _ = MODELS[m]
    X = lab_df[FEATS[fs]].to_numpy(float)
    rng = np.random.default_rng(SEED)
    drops = {g: [] for g in GROUPS[fs]}
    for i, (tr, te) in enumerate(splits[:N_OUTER_SPLITS]):  # premiere repetition : 5 plis
        est = clone(model).set_params(**majority_params(fs, m)).fit(X[tr], y[tr])
        base = f1_score(y[te], est.predict(X[te]), average="macro")
        for g, cols in GROUPS[fs].items():
            idx = [FEATS[fs].index(c) for c in cols]
            for _ in range(N_PERM_IMP):
                Xp = X[te].copy(); Xp[:, idx] = Xp[rng.permutation(len(te))][:, idx]
                drops[g].append(base - f1_score(y[te], est.predict(Xp), average="macro"))
    imp[fs] = {g: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for g, v in drops.items()}
R["group_importance"] = imp
print(json.dumps(imp, indent=1))

# %% [markdown]
# ## 13. Application : attribution probabiliste des evenements non specifies

# %%
fs = "S2"; m = best[fs]; model, _ = MODELS[m]
X_all = lab_df[FEATS[fs]].to_numpy(float)
final = clone(model).set_params(**majority_params(fs, m)).fit(X_all, y)
unspec = df[df["classe"] == "Unspecified"].copy()
if hasattr(final, "predict_proba"):
    proba = final.predict_proba(unspec[FEATS[fs]].to_numpy(float))
    for i, c in enumerate(CLASSES): unspec[f"p_{c}"] = proba[:, i]
unspec["prediction"] = [CLASSES[k] for k in final.predict(unspec[FEATS[fs]].to_numpy(float))]
cols = ["date", "heure debut", "centrales", "causes", "puissance_MW", "duree_min", "prediction"] + \
       [c for c in unspec.columns if c.startswith("p_")]
unspec[cols].to_csv(f"{OUT}/tables/T_application_non_specifies.csv", index=False)
R["unspecified_predictions"] = unspec[cols].astype(str).to_dict("records")
print(unspec[cols])

# %% [markdown]
# ## 14. Figures

# %%
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

# Fig. 6 : distribution nulle par permutation et importance groupee
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
for fs, col in [("S1", "#56B4E9"), ("S2", "#0072B2")]:
    null = np.load(f"{OUT}/tables/null_{fs}.npy")
    axs[0].hist(null, bins=30, color=col, alpha=.75, label=f"{fs} null")
    axs[0].axvline(perm[fs]["observed_macroF1"], color=col, lw=1.6)
axs[0].set_xlabel("Macro-F1"); axs[0].set_ylabel("Permutations"); axs[0].legend(frameon=False, fontsize=7, loc="upper left")
for fs, col in [("S1", "#56B4E9"), ("S2", "#0072B2")]:
    axs[0].text(perm[fs]["observed_macroF1"] - 0.004, axs[0].get_ylim()[1] * 0.97, f"{fs} observed", rotation=90, ha="right", va="top", fontsize=6.5, color="#333333")
g = pd.DataFrame(imp["S2"]).T.sort_values("mean")
axs[1].barh(g.index, g["mean"], xerr=g["sd"], color="#0072B2", height=0.6, error_kw=dict(lw=0.6, capsize=2))
axs[1].axvline(0, color="#4D4D4D", lw=0.6); axs[1].set_xlabel("Drop in macro-F1 when permuted (S2)")
axs[1].grid(axis="y", visible=False)
for a, l in zip(axs, "ab"): a.text(-0.12 if l == "a" else -0.45, 1.04, f"({l})", transform=a.transAxes, fontsize=9, weight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/fig/Fig6_null_importance.png"); plt.close(fig)

# %% [markdown]
# ## 15. Assertions de coherence et export

# %%
assert sum(R["detected_counts"].values()) == R["n_events"]
assert sum(R["class_counts"].values()) + R["detected_counts"].get("Unspecified", 0) == R["n_events"]
assert all(0 <= v["macro_F1"] <= 1 for v in R["cv_summary"].values())
assert len(cv) == 2 * 2 * len(MODELS) * N_OUTER_SPLITS * N_REPEATS
assert all(perm[f]["p_value"] >= 1 / (N_PERM + 1) for f in perm)
R["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__, "scipy": stats.__name__ and __import__("scipy").__version__,
                    "imbalanced_learn": __import__("imblearn").__version__, "seed": SEED,
                    "runtime_s": round(time.time() - t_start)}
json.dump(R, open(f"{OUT}/resultats.json", "w"), indent=1, ensure_ascii=False, default=str)
with pd.ExcelWriter(f"{OUT}/tables/donnees_etiquetees.xlsx") as w:
    df.drop(columns=["dt"]).to_excel(w, sheet_name="evenements_etiquetes", index=False)
    ct_leg.to_excel(w, sheet_name="accord_heritage")
    summ.to_excel(w, sheet_name="performance_cv")
    temporal.to_excel(w, sheet_name="validation_temporelle", index=False)
print("Termine en", R["environment"]["runtime_s"], "s")
