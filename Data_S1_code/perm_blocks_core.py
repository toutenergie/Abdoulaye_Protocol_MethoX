# Test de permutation par blocs-journees (Winkler et al. 2014) : les vecteurs d'etiquettes sont
# echanges entre journees de meme effectif, ce qui conserve la dependance intra-jour sous H0.
def day_block_permutation(est, X, y, groups, splits, n_perm, seed, n_jobs=2, verbose=0):
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.base import clone
    from sklearn.metrics import f1_score

    def score(yy):
        return float(np.mean([f1_score(yy[te], clone(est).fit(X[tr], yy[tr]).predict(X[te]), average="macro")
                              for tr, te in splits]))

    rng = np.random.default_rng(seed)
    days = {}
    for i, g in enumerate(groups):
        days.setdefault(g, []).append(i)
    by_size = {}
    for g, idx in days.items():
        by_size.setdefault(len(idx), []).append(idx)
    perms = []
    for _ in range(n_perm):  # tirages generes sequentiellement : resultat reproductible
        yp = y.copy()
        for size, blocks in by_size.items():
            order = rng.permutation(len(blocks))
            for dst, src in zip(blocks, order):
                yp[dst] = y[blocks[src]]
        perms.append(yp)
    observed = score(y)
    null = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(score)(yp) for yp in perms))
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    sizes = {int(k): len(v) for k, v in sorted(by_size.items())}
    return observed, null, float(p), sizes
