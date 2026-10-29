"""ATV04 — Importância das Variáveis por Remoção (LOFO) no KeyRecs.

Mesmo problema da ATV02/ATV03: prever hesitacao_livre (mediana do flight time no
texto livre) a partir de 4 features motoras do texto fixo + 3 demográficas.
"""
import json
import time
import warnings
import os
TABELA = os.environ.get("TABELA_ESCRITA", "tabelas/tabela_analitica.csv")   # tabela da escrita (ou a corrigida)
SUF = os.environ.get("SUFIXO", "")                                  # sufixo das pastas de saída
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEMENTE = 42
OUT = Path("resultados/saidas" + SUF)
OUT.mkdir(parents=True, exist_ok=True)

COR = {"sup": "#fcfcfb", "tinta": "#0b0b0b", "tinta2": "#52514e", "suave": "#898781",
       "grade": "#e1e0d9", "eixo": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834",
       "s3": "#1baf7a", "s4": "#eda100", "neg": "#2a78d6", "pos": "#e34948"}
mpl.rcParams.update({
    "figure.facecolor": COR["sup"], "axes.facecolor": COR["sup"], "savefig.facecolor": COR["sup"],
    "axes.edgecolor": COR["eixo"], "axes.labelcolor": COR["tinta2"], "axes.titlecolor": COR["tinta"],
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "axes.axisbelow": True, "grid.color": COR["grade"], "grid.linewidth": 0.8,
    "xtick.color": COR["suave"], "ytick.color": COR["suave"], "legend.frameon": False,
    "font.size": 10.5, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False})

ALVO = "hesitacao_livre"
FEATS = ["fix_dwell_med", "fix_flight_med", "fix_total_cv", "fix_ganho_pratica",
         "dem_idade", "dem_genero", "dem_lateralidade"]
CATS = {"dem_genero": "Male", "dem_lateralidade": "Right-Handed"}   # referência = mais frequente

dados = pd.read_csv(TABELA, index_col="participant")
print("n =", len(dados))


def desenho(X):
    """Dummies com categoria de referência fixa; devolve DataFrame numérico."""
    X = X.copy()
    for c, ref in CATS.items():
        if c in X:
            X[c] = (X[c] != ref).astype(float)
    return X.astype(float)


def novo_rf(**hp):
    return RandomForestRegressor(n_estimators=300, random_state=SEMENTE, n_jobs=-1, **hp)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


# ── hiperparâmetros da floresta: CV interna no treino (mesmo protocolo da ATV03) ──
X_all = desenho(dados[FEATS + ["fix_total_med"]])
y_all = dados[ALVO]
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.30, random_state=SEMENTE)
t0 = time.time()
busca = GridSearchCV(novo_rf(), {"min_samples_leaf": [3, 5, 10, 15, 20], "max_features": [0.4, 0.7, 1.0]},
                     cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEMENTE),
                     scoring="neg_root_mean_squared_error", n_jobs=1).fit(X_tr[FEATS], y_tr)
HP = busca.best_params_
print("HP RF:", HP, f"({time.time()-t0:.0f}s)")

CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEMENTE)
DOBRAS = list(CV.split(X_all))


def teste_nb(d, razao=0.25):
    """t corrigido de Nadeau & Bengio (2003) para diferenças pareadas entre dobras de CV."""
    d = np.asarray(d)
    k = len(d)
    t = d.mean() / np.sqrt((1 / k + razao) * d.var(ddof=1))
    return t, 2 * stats.t.sf(abs(t), k - 1)


def lofo(fabrica, feats, grupos=None, dobras=DOBRAS):
    """LOFO por CV: para cada dobra, RMSE(sem j) − RMSE(com todas). Re-treina o modelo."""
    grupos = grupos or {f: [f] for f in feats}
    base, sem = [], {g: [] for g in grupos}
    for tr, te in dobras:
        Xa, Xb, ya, yb = X_all.iloc[tr], X_all.iloc[te], y_all.iloc[tr], y_all.iloc[te]
        base.append(rmse(yb, fabrica().fit(Xa[feats], ya).predict(Xb[feats])))
        for g, cols in grupos.items():
            fs = [f for f in feats if f not in cols]
            sem[g].append(rmse(yb, fabrica().fit(Xa[fs], ya).predict(Xb[fs])))
    base = np.array(base)
    linhas = []
    for g in grupos:
        d = np.array(sem[g]) - base
        t, p = teste_nb(d)
        linhas.append({"variavel": g, "lofo_ms": 1000 * d.mean(), "dp_ms": 1000 * d.std(ddof=1),
                       "pct": 100 * d.mean() / base.mean(), "frac_pos": (d > 0).mean(), "p_nb": p})
    tab = pd.DataFrame(linhas).sort_values("lofo_ms", ascending=False).reset_index(drop=True)
    return tab, base, {g: np.array(sem[g]) - base for g in grupos}


res = {}
t0 = time.time()
tab_rf, base_rf, d_rf = lofo(lambda: novo_rf(**HP), FEATS)
print(f"\nLOFO RF (CV 5x10) — RMSE base {1000*base_rf.mean():.1f} ms  ({time.time()-t0:.0f}s)")
print(tab_rf.round(3).to_string())
res["rf_base_rmse_ms"] = 1000 * base_rf.mean()

# baseline (mediana) para contexto
base_med = np.array([rmse(y_all.iloc[te], np.full(len(te), y_all.iloc[tr].median())) for tr, te in DOBRAS])
res["baseline_rmse_ms"] = 1000 * base_med.mean()


def novo_lin():
    cont = [c for c in ["fix_dwell_med", "fix_flight_med", "fix_total_cv", "fix_ganho_pratica",
                         "dem_idade", "fix_total_med"]]
    ct = ColumnTransformer([("z", StandardScaler(), lambda X: [c for c in X.columns if c in cont])],
                           remainder="passthrough")
    return make_pipeline(ct, LinearRegression())


t0 = time.time()
tab_lin, base_lin, _ = lofo(novo_lin, FEATS)
print(f"\nLOFO Linear — RMSE base {1000*base_lin.mean():.1f} ms ({time.time()-t0:.0f}s)")
print(tab_lin.round(3).to_string())
res["lin_base_rmse_ms"] = 1000 * base_lin.mean()

# ── Experimento de redundância: adicionar fix_total_med (≈ dwell + flight, ρ alto com flight) ──
FEATS_R = FEATS + ["fix_total_med"]
print("\nCorrelação flight × total_med:", round(dados[["fix_flight_med", "fix_total_med"]].corr().iloc[0, 1], 3))
res["rho_flight_total"] = float(dados[["fix_flight_med", "fix_total_med"]].corr().iloc[0, 1])
t0 = time.time()
grupos_r = {f: [f] for f in FEATS_R}
grupos_r["{flight + total_med}"] = ["fix_flight_med", "fix_total_med"]
tab_r, base_r, _ = lofo(lambda: novo_rf(**HP), FEATS_R, grupos_r)
print(f"\nLOFO RF com variável redundante ({time.time()-t0:.0f}s)")
print(tab_r.round(3).to_string())

# ── PFI no hold-out, mesmo modelo, para contraste (com e sem redundante) ──
def pfi(feats):
    m = novo_rf(**HP).fit(X_tr[feats], y_tr)
    r = permutation_importance(m, X_te[feats], y_te, scoring="neg_root_mean_squared_error",
                               n_repeats=50, random_state=SEMENTE)
    return pd.Series(1000 * r.importances_mean, index=feats)


pfi_7 = pfi(FEATS)
pfi_8 = pfi(FEATS_R)
print("\nPFI hold-out (ms) — 7 feats:\n", pfi_7.round(2).to_string())
print("\nPFI hold-out (ms) — 8 feats:\n", pfi_8.round(2).to_string())

# ── Estabilidade: ranking LOFO em 5 sementes diferentes de partição (5x2 cada) ──
ranks = []
for s in [1, 7, 21, 99, 2026]:
    dob = list(RepeatedKFold(n_splits=5, n_repeats=2, random_state=s).split(X_all))
    t, _, _ = lofo(lambda: novo_rf(**HP), FEATS, dobras=dob)
    ranks.append(t.set_index("variavel")["lofo_ms"].rank(ascending=False).rename(s))
rank_tab = pd.concat(ranks, axis=1)
print("\nRanking LOFO em 5 sementes:\n", rank_tab.to_string())
res["top1_estavel"] = int((rank_tab.loc["fix_flight_med"] == 1).sum())

# ── salvar tabelas ──
tab_rf.to_csv(OUT / "lofo_rf.csv", index=False)
tab_lin.to_csv(OUT / "lofo_linear.csv", index=False)
tab_r.to_csv(OUT / "lofo_redundancia.csv", index=False)
pd.DataFrame({"pfi_7": pfi_7, "pfi_8": pfi_8}).to_csv(OUT / "pfi.csv")
rank_tab.to_csv(OUT / "lofo_ranks_sementes.csv")
res["HP"] = HP
res["n"] = len(dados)
json.dump(res, open(OUT / "resumo.json", "w"), indent=2, default=float)

# ═════════════════════════ FIGURAS ═════════════════════════
ROT = {"fix_flight_med": "flight (pausa entre teclas)", "fix_total_cv": "CV do tempo total",
       "fix_dwell_med": "dwell (tecla pressionada)", "fix_ganho_pratica": "ganho com prática",
       "dem_idade": "idade", "dem_genero": "gênero", "dem_lateralidade": "lateralidade",
       "fix_total_med": "tempo total (redundante)", "{flight + total_med}": "grupo {flight + total}"}


def barras(ax, tab, titulo, destaque=None):
    t = tab.sort_values("lofo_ms")
    cores = [COR["s2"] if v > 0 else COR["s1"] for v in t["lofo_ms"]]
    if destaque:
        cores = [COR["s4"] if n in destaque else c for n, c in zip(t["variavel"], cores)]
    ax.barh([ROT[v] for v in t["variavel"]], t["lofo_ms"], xerr=t["dp_ms"] / np.sqrt(50),
            color=cores, height=0.6, error_kw={"ecolor": COR["tinta2"], "capsize": 3, "lw": 1})
    ax.axvline(0, color=COR["tinta"], lw=1)
    ax.set_title(titulo)
    ax.set_xlabel("aumento do RMSE ao remover a variável (ms)")
    ax.grid(axis="y", visible=False)


# fig1: LOFO RF
fig, ax = plt.subplots(figsize=(8, 4.2))
barras(ax, tab_rf, "LOFO — Random Forest (CV 5×10)")
for i, (v, p) in enumerate(zip(tab_rf.sort_values("lofo_ms")["lofo_ms"], tab_rf.sort_values("lofo_ms")["p_nb"])):
    ax.text(max(v, 0) + 0.4, i, f"{v:+.1f} ms" + ("  *" if p < 0.05 else ""), va="center",
            fontsize=9, color=COR["tinta2"])
plt.savefig(OUT / "fig1_lofo_rf.png"); plt.close()

# fig2: RF vs linear
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
ordem = tab_rf.sort_values("lofo_ms")["variavel"].tolist()
for ax, tab, tt in [(axes[0], tab_rf, "Random Forest"), (axes[1], tab_lin, "Regressão linear")]:
    t = tab.set_index("variavel").loc[ordem]
    ax.barh([ROT[v] for v in ordem], t["lofo_ms"], color=[COR["s2"] if v > 0 else COR["s1"] for v in t["lofo_ms"]],
            height=0.6)
    ax.axvline(0, color=COR["tinta"], lw=1); ax.set_title(f"LOFO — {tt}"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ RMSE (ms)")
plt.tight_layout(); plt.savefig(OUT / "fig2_lofo_modelos.png"); plt.close()

# fig3: redundância — LOFO vs PFI
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
alvo3 = ["fix_flight_med", "fix_total_med", "{flight + total_med}"]
l7 = tab_rf.set_index("variavel")["lofo_ms"]
l8 = tab_r.set_index("variavel")["lofo_ms"]
ax = axes[0]
x = np.arange(3); w = 0.38
ax.bar(x - w/2, [l7["fix_flight_med"], 0, 0], w, color=COR["suave"], label="sem total_med (7 var.)")
ax.bar(x + w/2, [l8[a] for a in alvo3], w, color=[COR["s2"], COR["s2"], COR["s4"]], label="com total_med (8 var.)")
ax.set_xticks(x, ["flight", "total_med", "grupo\n{flight+total}"])
ax.set_title("LOFO: a importância se divide"); ax.set_ylabel("Δ RMSE (ms)"); ax.legend(fontsize=8.5)
ax.grid(axis="x", visible=False)
ax = axes[1]
ax.bar(x[:2] - w/2, [pfi_7["fix_flight_med"], 0], w, color=COR["suave"], label="sem total_med")
ax.bar(x[:2] + w/2, [pfi_8["fix_flight_med"], pfi_8["fix_total_med"]], w, color=COR["s1"], label="com total_med")
ax.set_xticks(x[:2], ["flight", "total_med"]); ax.set_title("PFI (hold-out): mesma situação")
ax.set_ylabel("Δ RMSE ao permutar (ms)"); ax.legend(fontsize=8.5); ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(OUT / "fig3_redundancia.png"); plt.close()

# fig4: distribuição por dobra (estabilidade)
fig, ax = plt.subplots(figsize=(8, 4))
ordem = tab_rf["variavel"].tolist()[::-1]
ax.boxplot([1000 * d_rf[v] for v in ordem], vert=False, widths=0.55, patch_artist=True,
           boxprops={"facecolor": "#dbe8f8", "edgecolor": COR["s1"]}, medianprops={"color": COR["s2"], "lw": 2},
           whiskerprops={"color": COR["suave"]}, capprops={"color": COR["suave"]},
           flierprops={"markeredgecolor": COR["suave"], "markersize": 4})
ax.set_yticks(range(1, len(ordem) + 1), [ROT[v] for v in ordem])
ax.axvline(0, color=COR["tinta"], lw=1)
ax.set_xlabel("Δ RMSE por dobra (ms) — 50 dobras"); ax.set_title("LOFO dobra a dobra: sinal × ruído")
ax.grid(axis="y", visible=False)
plt.savefig(OUT / "fig4_lofo_dobras.png"); plt.close()
print("\nFiguras salvas.")
