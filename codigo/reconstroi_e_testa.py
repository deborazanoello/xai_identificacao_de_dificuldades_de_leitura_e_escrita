"""Reconstroi a tabela a partir da KeyRecs bruta e testa a robustez do LOFO."""
import sys, json, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
warnings.filterwarnings("ignore")
B = sys.argv[1] if len(sys.argv) > 1 else "../dados/keyrecs"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "resultados/saidas_comp"
import os; os.makedirs(OUTDIR, exist_ok=True)
dem = pd.read_csv(f"{B}/demographics.csv").set_index("participant")
fx = pd.read_csv(f"{B}/fixed-text.csv")
fr = pd.read_csv(f"{B}/free-text.csv", low_memory=False, dtype=str); fr.columns = [c.strip() for c in fr.columns]
tab = pd.read_csv("tabelas/tabela_analitica.csv", index_col="participant")
cols = fx.columns.tolist()
dw = [c for c in cols if c.startswith("DU.") and c.split(".")[1] == c.split(".")[2]]
pausa = [c for c in cols if c.startswith("DU.") and c.split(".")[1] != c.split(".")[2]]

# --- 1. definicao do tempo total
soma_int = fx[dw[1:-1]].sum(axis=1) + fx[pausa].sum(axis=1)
ok = np.isclose(soma_int, fx["total time"], atol=0.002)
print(f"tempo total = soltar a 1a tecla -> apertar a ultima (pausas + dwell internos): {100*ok.mean():.2f}% das linhas")
print("   linhas que nao batem:", fx.loc[~ok, ["participant","session","repetition","total time"]].assign(esperado=soma_int[~ok].round(3)).values.tolist()[:6])

# --- 2. linhas quebradas do texto livre
num = ["DU.key1.key1","DD.key1.key2","DU.key1.key2","UD.key1.key2","UU.key1.key2"]
bad = fr[pd.to_numeric(fr["UD.key1.key2"], errors="coerce").isna() | pd.to_numeric(fr["DU.key1.key1"], errors="coerce").isna()]
print(f"texto livre: {len(bad)} linhas com valor nao numerico; teclas envolvidas:", pd.concat([bad.key1, bad.key2]).value_counts().head(6).to_dict())
for c in num: fr[c] = pd.to_numeric(fr[c], errors="coerce")

# --- 3. tabela reconstruida (definicoes documentadas)
g = fx.groupby("participant")
def cv_limpo(s):
    s = s[s > 0]; q1, q3 = s.quantile([.25, .75]); s = s[s <= q3 + 3 * (q3 - q1)]
    return s.std() / s.mean()
def ganho(d):
    a = d.loc[d.session == 1, "total time"].median(); b = d.loc[d.session == 2, "total time"].median(); return (a - b) / a
rec = pd.DataFrame({
    "dem_idade": dem.age, "dem_genero": dem.gender, "dem_lateralidade": dem.handedness,
    "fix_dwell_med": g.apply(lambda d: np.nanmedian(d[dw].values)),
    "fix_flight_med": g.apply(lambda d: np.nanmedian(d[pausa].values)),
    "fix_total_med": g["total time"].median(),
    "fix_total_cv": g["total time"].apply(cv_limpo),
    "fix_ganho_pratica": g.apply(ganho),
    "hesitacao_livre": fr.groupby("participant")["UD.key1.key2"].median()})
rec.to_csv(f"{OUTDIR}/tabela_reconstruida_99.csv")
r96 = rec.loc[tab.index]
print("\nreconstruida x tabela do grupo (96):")
for c in ["fix_dwell_med","fix_flight_med","fix_total_med","fix_total_cv","fix_ganho_pratica","hesitacao_livre"]:
    d = (r96[c] - tab[c]).abs()
    print(f"   {c:18s} corr={np.corrcoef(r96[c], tab[c])[0,1]:.4f}  |dif| mediana={d.median():.4f} max={d.max():.4f}  iguais(<1e-3)={int((d<1e-3).sum())}/96")
print("participantes com hesitacao <= 0 na base bruta:", rec.index[rec.hesitacao_livre <= 0].tolist())

# --- 4. LOFO em tres versoes da tabela
FEATS = ["fix_dwell_med","fix_flight_med","fix_total_cv","fix_ganho_pratica","dem_idade","dem_genero","dem_lateralidade"]
HP = {"max_features": 1.0, "min_samples_leaf": 10}
def desenho(X):
    X = X.copy(); X["dem_genero"] = (X["dem_genero"] != "Male").astype(float)
    X["dem_lateralidade"] = (X["dem_lateralidade"] != "Right-Handed").astype(float); return X.astype(float)
def lofo(df, rotulo):
    X, y = desenho(df[FEATS]), df["hesitacao_livre"]
    dob = list(RepeatedKFold(n_splits=5, n_repeats=10, random_state=42).split(X))
    def uma(k, fs):
        tr, te = dob[k]
        if not fs: p = np.full(len(te), y.iloc[tr].median())
        else: p = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1, **HP).fit(X.iloc[tr][fs], y.iloc[tr]).predict(X.iloc[te][fs])
        return k, tuple(fs), np.sqrt(np.mean((y.iloc[te].values - p) ** 2))
    confs = [FEATS, []] + [[f for f in FEATS if f != v] for v in FEATS] + [["fix_flight_med"]]
    res = Parallel(n_jobs=2)(delayed(uma)(k, fs) for k in range(50) for fs in confs)
    R = {}
    for k, fs, r in res: R.setdefault(fs, np.zeros(50))[k] = r
    base = R[tuple(FEATS)]
    linhas = []
    for v in FEATS:
        d = 1000 * (R[tuple(f for f in FEATS if f != v)] - base)
        ep = np.sqrt((1/50 + 0.25) * d.var(ddof=1)); t = d.mean() / ep if ep > 0 else 0
        linhas.append({"versao": rotulo, "variavel": v, "lofo_ms": d.mean(), "p": 2 * stats.t.sf(abs(t), 49)})
    out = pd.DataFrame(linhas)
    resumo = {"versao": rotulo, "n": len(df), "rmse_ms": 1000 * base.mean(), "mediana_ms": 1000 * R[()].mean(),
              "so_voo_ms": 1000 * R[("fix_flight_med",)].mean(), "r_voo_hes": df[["fix_flight_med","hesitacao_livre"]].corr().iloc[0,1]}
    return out, resumo
todas, resumos = [], []
for df, rot in [(tab, "tabela do grupo (96)"), (rec.loc[tab.index], "reconstruida (96)"), (rec, "reconstruida (99)")]:
    o, r = lofo(df, rot); todas.append(o); resumos.append(r)
todas = pd.concat(todas); todas.to_csv(f"{OUTDIR}/lofo_robustez_tabelas.csv", index=False)
pd.DataFrame(resumos).to_csv(f"{OUTDIR}/lofo_robustez_resumo.csv", index=False)
print("\n", pd.DataFrame(resumos).round(3).to_string())
print("\n", todas.pivot(index="variavel", columns="versao", values="lofo_ms").round(2).to_string())
print("\n p:\n", todas.pivot(index="variavel", columns="versao", values="p").round(3).to_string())
