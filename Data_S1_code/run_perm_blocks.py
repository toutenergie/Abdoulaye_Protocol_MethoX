import json, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from perm_blocks_core import day_block_permutation
SEED = 42; N_PERM = 1000
R = json.load(open("sorties/resultats.json"))
d = pd.read_excel("sorties/tables/donnees_etiquetees.xlsx")
CLASSES = ["Output reduction", "Fluctuation", "Disconnection"]
lab = d[d.classe.isin(CLASSES)].reset_index(drop=True)
y = lab.classe.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
groups = pd.to_datetime(lab["date"]).dt.strftime("%Y-%m-%d").to_numpy()
splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, groups))
out = {}
for fs in ["S1", "S2"]:
    assert R["best_model"][fs] == "Random forest"
    est = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=SEED, n_jobs=1,
                                 **R["permutation_test"][fs]["params"])
    X = lab[R["features"][fs]].to_numpy(float)
    t0 = time.time()
    obs, null, p, sizes = day_block_permutation(est, X, y, groups, splits, N_PERM, SEED, n_jobs=2, verbose=5)
    assert abs(obs - R["permutation_test"][fs]["observed_macroF1"]) < 1e-9, (obs, R["permutation_test"][fs]["observed_macroF1"])
    out[fs] = {"observed_macroF1": obs, "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, .95)),
               "null_max": float(null.max()), "p_value": p, "n_perm": N_PERM, "day_size_classes": sizes,
               "seconds": round(time.time() - t0)}
    np.save(f"sorties/tables/null_blocs_{fs}.npy", null)
    print(fs, out[fs], flush=True)
json.dump(out, open("sorties/resultats_permutation_blocs.json", "w"), indent=1)
