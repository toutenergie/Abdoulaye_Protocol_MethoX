"""Stage 1 : detection des causes par fouille de texte du journal de dispatching (regles explicites)."""
import re, unicodedata
import numpy as np
import pandas as pd

DATA = "/root/.claude/uploads/7f974846-0d95-5994-81c7-2da5b3d49b9b/0666734d-1791242392980_defauts_senelec_enr.xlsx"


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --- 1. Correction orthographique deterministe (variantes observees dans le journal) -------------
SPELLING = {
    r"\bfluctation\b": "fluctuation", r"\bfuctuation\b": "fluctuation",
    r"\bcenrales\b": "centrales", r"\bbiasse\b": "baisse", r"\bbaise\b": "baisse",
    r"\bmeckhe\b": "mekhe", r"\bsclaing\b": "scaling", r"\bscalig\b": "scaling",
    r"\ber\b": "et", r"\bcpv\b": "centrales pv", r"\bsmk\b": "santhiou mekhe",
    r"\bagg\b": "aggreko",
}

# --- 2. Lexique des mecanismes ------------------------------------------------------------------
LEX = {
    "reduction":   r"\b(baisses?|perte de production|diminution|chute|baisse de charges?)\b",
    "fluctuation": r"\b(fluctuations?|variations?|perturbations?|instabilite)\b",
    "trip":        r"\b(declenchement|perte interconnection|surcharge)\b",
    "pickup":      r"\breprise( de)? charge\b",
}
# Objets non renouvelables (thermique, reseau, interconnexions) : gazetteer
NON_RE_OBJ = {
    "thermal_unit": r"\b(aggreko|tobene|kps|petn|tp|g ?0?\d+|groupes?)\b",
    "network":      r"\b(ligne|225 kv|disjoncteur|tr ?\d|transfos?|poste|interconnection|dagana)\b",
    "interconnected_utility": r"\b(nawec|somelec|edm)\b",
}
# Gazetteer des centrales renouvelables (alias -> centrale canonique)
PLANTS = {
    "Kahone (ERS / Scaling)": r"\b(kahone|kaolack)\b",
    "Santhiou Mekhe":         r"\bmekhe\b",
    "Sakal":                  r"\bsakal\b",
    "Bokhol":                 r"\bbokhol\b",
    "Diass":                  r"\bdiass\b",
    "Malicounda / Mbour":     r"\b(malicounda|mbour)\b",
    "Ten Merina":             r"\bten merina\b",
    "Kael":                   r"\bkael\b",
    "Touba":                  r"\btouba\b",
    "Taiba Ndiaye (wind)":    r"\b(taiba|eolien|eolienne)\b",
}


def detect(row):
    centr, cause = norm(row["centrales"]), norm(row["causes"])
    t = f"{cause} {centr}" if centr not in ("pv", "enr") else cause
    for a, b in SPELLING.items():
        t = re.sub(a, b, t)
    # Liaison mecanisme-objet : un terme de mecanisme qui porte sur un reseau interconnecte ou une
    # unite thermique n'est pas un mecanisme renouvelable (ex. "perturbations EDM", "baisse de PETN")
    t_mech = re.sub(r"\b(perturbations?|depassement|limitation|perte)( de| des)?"
                    r"( \d+ mw)?( de)? (somelec|edm|nawec|tp|petn|kps)\b", " ", t)
    mech = {k: bool(re.search(p, t_mech)) for k, p in LEX.items()}
    nonre = {k: bool(re.search(p, t)) for k, p in NON_RE_OBJ.items()}
    plants = [p for p, rx in PLANTS.items() if re.search(rx, t)]
    # Malicounda : centrale PV (22 MW) ET centrale thermique ; "groupes"/"G07" designent le thermique
    trip_obj_thermal = mech["trip"] and (nonre["thermal_unit"] or nonre["network"])
    trip_obj_renew = mech["trip"] and not trip_obj_thermal and bool(re.search(r"\b(pv|centrales? pv|centrale pv)\b", t))
    renew_mech = mech["reduction"] or mech["fluctuation"] or trip_obj_renew
    # --- Regle de decision hierarchique ---------------------------------------------------------
    if trip_obj_thermal and not (mech["reduction"] or mech["fluctuation"]):
        label = "Non-renewable origin"
    elif mech["trip"] and not renew_mech and not trip_obj_thermal:
        label = "Non-renewable origin"          # declenchement sans objet renouvelable (ex. surcharge)
    elif mech["pickup"] and not renew_mech:
        label = "Non-renewable origin"          # reprise de charge sur poste
    elif re.search(r"\b(dagana sakal)\b", t) and not (mech["fluctuation"] or re.search(r"\bpv\b", t)):
        label = "Non-renewable origin"          # baisse de transit sur la ligne Dagana-Sakal
    elif trip_obj_renew:
        label = "Renewable plant trip"
    elif mech["fluctuation"]:
        label = "Fluctuation"
    elif mech["reduction"]:
        label = "Output reduction"
    else:
        label = "Unspecified"
    compound = renew_mech and (nonre["thermal_unit"] or nonre["interconnected_utility"]
                               or (mech["trip"] and trip_obj_thermal))
    return pd.Series({
        "texte_normalise": t, "cause_detectee": label, "co_cause_non_renouvelable": bool(compound),
        "multi_mecanisme": int(mech["reduction"]) + int(mech["fluctuation"]) + int(mech["trip"]) > 1,
        "centrales_citees": "; ".join(plants), "n_centrales_citees": len(plants),
    })


def run(path=DATA):
    df = pd.read_excel(path)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    out = pd.concat([df, df.apply(detect, axis=1)], axis=1)
    return out


if __name__ == "__main__":
    d = run()
    print(d["cause_detectee"].value_counts())
    print(pd.crosstab(d["cause_detectee"], d["cause_enr"], margins=True))
    print("compound", d.co_cause_non_renouvelable.sum(), "multi", d.multi_mecanisme.sum())
    print("attribution", (d.n_centrales_citees > 0).mean())
    for lab in ["Non-renewable origin", "Renewable plant trip", "Unspecified"]:
        print("---", lab); print(d.loc[d.cause_detectee == lab, "texte_normalise"].tolist())
    print(d.loc[d.co_cause_non_renouvelable, ["texte_normalise", "cause_detectee"]].to_string())
    pl = d["centrales_citees"].str.split("; ").explode().replace("", np.nan).dropna().value_counts()
    print(pl)
