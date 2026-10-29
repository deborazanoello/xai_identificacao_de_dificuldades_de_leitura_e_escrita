"""ATV04 — LOFO nos pinguins de Palmer (mesmo exemplo das aulas): prever P(sexo = fêmea)."""
import json
import time
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEMENTE = 42
OUT = Path("saidas_pinguins")
OUT.mkdir(exist_ok=True)

COR = {"sup": "#fcfcfb", "tinta": "#0b0b0b", "tinta2": "#52514e", "suave": "#898781",
       "grade": "#e1e0d9", "eixo": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834", "s4": "#eda100"}
mpl.rcParams.update({
    "figure.facecolor": COR["sup"], "axes.facecolor": COR["sup"], "savefig.facecolor": COR["sup"],
    "axes.edgecolor": COR["eixo"], "axes.labelcolor": COR["tinta2"], "axes.titlecolor": COR["tinta"],
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "axes.axisbelow": True, "grid.color": COR["grade"], "grid.linewidth": 0.8,
    "xtick.color": COR["suave"], "ytick.color": COR["suave"], "legend.frameon": False,
    "font.size": 10.5, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False})

d = load_penguins().dropna(subset=["sex", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])
d = d.reset_index(drop=True)
NUM = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CAT = ["species", "island"]
FEATS = NUM + CAT
X_all = d[FEATS]
y_all = (d["sex"] == "female").astype(int)
print("n =", len(d), " fêmeas =", y_all.mean().round(3))
ROT = {"bill_length_mm": "comprimento do bico", "bill_depth_mm": "profundidade do bico",
       "flipper_length_mm": "comprimento da nadadeira", "body_mass_g": "massa corporal",
       "species": "espécie", "island": "ilha", "{species,island}": "grupo {espécie, ilha}",
       "{bico}": "grupo {bico: compr. + prof.}", "{4 medidas}": "grupo {4 medidas}"}


def preproc(feats):
    num = [f for f in feats if f in NUM]
    cat = [f for f in feats if f in CAT]
    tr = []
    if num:
        tr.append(("z", StandardScaler(), num))
    if cat:
        tr.append(("oh", OneHotEncoder(handle_unknown="ignore", drop=None), cat))
    return ColumnTransformer(tr)


def rf(feats):
    return make_pipeline(preproc(feats), RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                                max_features=0.5, random_state=SEMENTE, n_jobs=-1))


def logit(feats):
    return make_pipeline(preproc(feats), LogisticRegression(max_iter=2000, C=1.0))


def teste_nb(dd, razao=0.25):
    dd = np.asarray(dd); k = len(dd)
    t = dd.mean() / np.sqrt((1 / k + razao) * dd.var(ddof=1))
    return 2 * stats.t.sf(abs(t), k - 1)


DOBRAS = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEMENTE).split(X_all, y_all))
GRUPOS = {f: [f] for f in FEATS}
GRUPOS.update({"{species,island}": ["species", "island"], "{bico}": ["bill_length_mm", "bill_depth_mm"],
               "{4 medidas}": NUM})


def lofo(fab):
    base_ll, base_ac, sem_ll, sem_ac = [], [], {g: [] for g in GRUPOS}, {g: [] for g in GRUPOS}
    for tr, te in DOBRAS:
        Xa, Xb, ya, yb = X_all.iloc[tr], X_all.iloc[te], y_all.iloc[tr], y_all.iloc[te]
        p = fab(FEATS).fit(Xa, ya).predict_proba(Xb)[:, 1]
        base_ll.append(log_loss(yb, p, labels=[0, 1])); base_ac.append(accuracy_score(yb, p > 0.5))
        for g, cols in GRUPOS.items():
            fs = [f for f in FEATS if f not in cols]
            p = fab(fs).fit(Xa[fs], ya).predict_proba(Xb[fs])[:, 1]
            sem_ll[g].append(log_loss(yb, p, labels=[0, 1])); sem_ac[g].append(accuracy_score(yb, p > 0.5))
    base_ll, base_ac = np.array(base_ll), np.array(base_ac)
    lin = []
    for g in GRUPOS:
        dd = np.array(sem_ll[g]) - base_ll
        da = base_ac - np.array(sem_ac[g])
        lin.append({"variavel": g, "lofo": dd.mean(), "dp": dd.std(ddof=1), "frac_pos": (dd > 0).mean(),
                    "p_nb": teste_nb(dd), "queda_acuracia_pp": 100 * da.mean()})
    return pd.DataFrame(lin).sort_values("lofo", ascending=False).reset_index(drop=True), base_ll, base_ac


res = {"n": len(d), "frac_femeas": float(y_all.mean())}
t0 = time.time()
tab_rf, ll_rf, ac_rf = lofo(rf)
print(f"\nLOFO RF (CV estrat. 5x10) — logloss {ll_rf.mean():.3f}  acurácia {ac_rf.mean():.3f} ({time.time()-t0:.0f}s)")
print(tab_rf.round(4).to_string())
tab_lg, ll_lg, ac_lg = lofo(logit)
print(f"\nLOFO Logística — logloss {ll_lg.mean():.3f}  acurácia {ac_lg.mean():.3f}")
print(tab_lg.round(4).to_string())
res.update({"rf_logloss": ll_rf.mean(), "rf_acc": ac_rf.mean(), "lg_logloss": ll_lg.mean(), "lg_acc": ac_lg.mean(),
            "logloss_chance": float(log_loss(y_all, np.full(len(y_all), y_all.mean())))})

# PFI no hold-out (mesmo modelo RF), sobre as colunas ORIGINAIS (espécie permutada inteira)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.3, random_state=SEMENTE, stratify=y_all)
m_rf = rf(FEATS).fit(X_tr, y_tr)
m_lg = logit(FEATS).fit(X_tr, y_tr)
pfi = {}
for nome, m in [("rf", m_rf), ("lg", m_lg)]:
    r = permutation_importance(m, X_te, y_te, scoring="neg_log_loss", n_repeats=50, random_state=SEMENTE)
    pfi[nome] = pd.Series(r.importances_mean, index=FEATS)
print("\nPFI hold-out (Δ logloss):\n", pd.DataFrame(pfi).round(4).to_string())

# quão bem as medidas recuperam a espécie? (explica a redundância)
from sklearn.model_selection import cross_val_score
acc_sp = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), d[NUM], d["species"],
                         cv=5).mean()
res["acc_especie_pelas_medidas"] = float(acc_sp)
print("\nAcurácia para prever a espécie só pelas 4 medidas:", round(acc_sp, 3))
print(pd.crosstab(d["species"], d["island"]))

tab_rf.to_csv(OUT / "lofo_rf.csv", index=False)
tab_lg.to_csv(OUT / "lofo_logistica.csv", index=False)
pd.DataFrame(pfi).to_csv(OUT / "pfi.csv")
json.dump(res, open(OUT / "resumo.json", "w"), indent=2, default=float)

# ── figuras ──
ind_rf = tab_rf[tab_rf["variavel"].isin(FEATS)]
ordem = ind_rf.sort_values("lofo")["variavel"].tolist()


def barras(ax, serie, titulo, xl, cor_fixa=None):
    s = serie.loc[ordem]
    ax.barh([ROT[v] for v in ordem], s, color=cor_fixa or [COR["s2"] if v > 0 else COR["s1"] for v in s], height=0.6)
    ax.axvline(0, color=COR["tinta"], lw=1); ax.set_title(titulo); ax.set_xlabel(xl); ax.grid(axis="y", visible=False)


fig, ax = plt.subplots(figsize=(8, 4.2))
t = ind_rf.set_index("variavel").loc[ordem]
ax.barh([ROT[v] for v in ordem], t["lofo"], xerr=t["dp"] / np.sqrt(len(DOBRAS)),
        color=[COR["s2"] if v > 0 else COR["s1"] for v in t["lofo"]], height=0.6,
        error_kw={"ecolor": COR["tinta2"], "capsize": 3, "lw": 1})
ax.axvline(0, color=COR["tinta"], lw=1); ax.grid(axis="y", visible=False)
ax.set_title("LOFO — Random Forest, P(fêmea), CV estratificada 5×10"); ax.set_xlabel("aumento da log-loss ao remover")
plt.savefig(OUT / "fig1_lofo_rf.png"); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
barras(axes[0], pfi["rf"], "PFI (hold-out): permutar", "aumento da log-loss ao permutar", COR["s1"])
barras(axes[1], tab_rf.set_index("variavel")["lofo"], "LOFO: remover e re-treinar", "aumento da log-loss ao remover")
plt.tight_layout(); plt.savefig(OUT / "fig2_pfi_vs_lofo.png"); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
barras(axes[0], tab_rf.set_index("variavel")["lofo"], "LOFO — Random Forest", "Δ log-loss")
barras(axes[1], tab_lg.set_index("variavel")["lofo"], "LOFO — Regressão logística", "Δ log-loss")
plt.tight_layout(); plt.savefig(OUT / "fig3_lofo_modelos.png"); plt.close()

fig, ax = plt.subplots(figsize=(7.5, 3.6))
gs = ["species", "island", "{species,island}", "bill_length_mm", "bill_depth_mm", "{bico}"]
t = tab_rf.set_index("variavel").loc[gs]
ax.bar([ROT[v].replace(" ", "\n", 1) if not v.startswith("{") else ROT[v].replace(": ", ":\n").replace(", ", ",\n") for v in gs],
       t["lofo"], color=[COR["s2"], COR["s2"], COR["s4"], COR["s2"], COR["s2"], COR["s4"]])
ax.axhline(0, color=COR["tinta"], lw=1); ax.grid(axis="x", visible=False)
ax.set_title("LOFO: sozinhas × em grupo (Random Forest)"); ax.set_ylabel("Δ log-loss")
plt.savefig(OUT / "fig4_grupos.png"); plt.close()
print("\nFiguras salvas.")
