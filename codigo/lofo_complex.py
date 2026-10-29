"""ATV04 - LOFO na leitura (CompLex), com o mesmo protocolo usado na escrita (KeyRecs).

Tabela: tabelas/tabela_complex_reconstruida.csv (monta_tabela_complex.py), 8.579 termos de uma palavra.
Alvo: termo dificil (complexidade >= 0,5). Validacao: 5 dobras agrupadas por palavra x 10 repeticoes.
Perda do LOFO: queda da AUC (em pontos de AUC, x100) e da precisao nos 10%.
p-valores e IC: teste t corrigido de Nadeau & Bengio (2003), razao teste/treino 0,25.
Uso: python lofo_complex.py          (completo: 50 dobras, ~25 min; gera resultados/saidas_complex/)
     python lofo_complex.py rapido   (teste: 10 dobras, sem ruido e sem alvo continuo, ~5 min; gera resultados/saidas_complex_rapido/)
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RAPIDO = "rapido" in sys.argv
REPS = 2 if RAPIDO else 10
OUT = Path("resultados/saidas_complex_rapido" if RAPIDO else "resultados/saidas_complex"); OUT.mkdir(parents=True, exist_ok=True)
SEMENTE, N_JOBS, K, RAZAO = 42, 2, 5 * REPS, 0.25
F = ["frequencia", "comprimento", "silabas", "sigla", "nome_proprio", "tam_frase", "posicao", "forma_pt"]
NOMES = {"frequencia": "frequência", "comprimento": "comprimento", "silabas": "sílabas", "sigla": "sigla",
         "nome_proprio": "nome próprio", "tam_frase": "tamanho da frase", "posicao": "posição",
         "forma_pt": "forma em português"}
t = pd.read_csv("tabelas/tabela_complex_reconstruida.csv", keep_default_na=False, dtype={"termo": str, "grupo": str})
X, y, g, yc = t[F].astype(float), t["dificil"].values, t["grupo"].values, t["complexidade"].values
DOBRAS = [(tr, te) for rep in range(REPS) for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=rep).split(X, y, g)]
REP = np.repeat(np.arange(REPS), 5)


def prec10(yt, p):
    k = int(np.ceil(0.10 * len(p))); return yt[np.argsort(-p)[:k]].mean()


def nb(d):
    d = np.asarray(d, float); k = len(d)
    ep = np.sqrt((1 / k + RAZAO) * d.var(ddof=1)); tt = d.mean() / ep if ep > 0 else 0.0
    m = stats.t.ppf(0.975, k - 1) * ep
    return {"media": d.mean(), "dp": d.std(ddof=1), "ic_inf": d.mean() - m, "ic_sup": d.mean() + m,
            "p": 2 * stats.t.sf(abs(tt), k - 1), "frac_pos": float((d > 0).mean())}


t0 = time.time()
# ------------------------------------------------------------ hiperparametros (busca em 70% dos grupos)
tr70, _ = next(GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=SEMENTE).split(X, y, g))
busca = GridSearchCV(RandomForestClassifier(n_estimators=300, random_state=SEMENTE, n_jobs=1),
                     {"min_samples_leaf": [3, 5, 10, 20], "max_features": [0.3, 0.5, 1.0]},
                     cv=GroupKFold(n_splits=5), scoring="roc_auc", n_jobs=N_JOBS)
busca.fit(X.iloc[tr70], y[tr70], groups=g[tr70])
HP = busca.best_params_
print("HP floresta:", HP, f"({time.time()-t0:.0f}s)")


def rf(semente=SEMENTE): return RandomForestClassifier(n_estimators=300, random_state=semente, n_jobs=1, **HP)
def logit(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


BLOCOS = {"{comprimento, sílabas}": ["comprimento", "silabas"], "{sigla, nome próprio}": ["sigla", "nome_proprio"],
          "{tamanho da frase, posição}": ["tam_frase", "posicao"]}
CONF_RF = {"todas": F, **{f"sem {v}": [f for f in F if f != v] for v in F},
           **{f"sem {b}": [f for f in F if f not in c] for b, c in BLOCOS.items()},
           **{f"só {v}": [v] for v in F}, "só {frequência, comprimento}": ["frequencia", "comprimento"]}
CONF_LG = {"todas": F, **{f"sem {v}": [f for f in F if f != v] for v in F}}


def uma_dobra(k):
    tr, te = DOBRAS[k]; yt = y[te]; out = {}
    for nome, fs in CONF_RF.items():
        p = rf().fit(X.iloc[tr][fs], y[tr]).predict_proba(X.iloc[te][fs])[:, 1]
        out[("rf", nome)] = (roc_auc_score(yt, p), prec10(yt, p))
        if nome == "todas":                                   # PFI no mesmo modelo e na mesma dobra
            m = rf().fit(X.iloc[tr], y[tr]); rng = np.random.default_rng(1000 + k)
            a0 = roc_auc_score(yt, m.predict_proba(X.iloc[te])[:, 1])
            for f in F:
                q = []
                for _ in range(10):
                    Xp = X.iloc[te].copy(); Xp[f] = rng.permutation(Xp[f].values)
                    q.append(a0 - roc_auc_score(yt, m.predict_proba(Xp)[:, 1]))
                out[("pfi", f)] = (np.mean(q), np.nan)
    for nome, fs in CONF_LG.items():
        p = logit().fit(X.iloc[tr][fs], y[tr]).predict_proba(X.iloc[te][fs])[:, 1]
        out[("lg", nome)] = (roc_auc_score(yt, p), prec10(yt, p))
    p = -X.iloc[te]["frequencia"].values
    out[("regra", "frequência")] = (roc_auc_score(yt, p), prec10(yt, p))
    return k, out


res = Parallel(n_jobs=N_JOBS)(delayed(uma_dobra)(k) for k in range(K))
print(f"LOFO/PFI: {time.time()-t0:.0f}s")
A = {}
for k, out in res:
    for chave, v in out.items():
        A.setdefault(chave, np.full((K, 2), np.nan))[k] = v
pd.DataFrame([{"modelo": m, "config": c, "dobra": k, "auc": v[k, 0], "prec10": v[k, 1]}
              for (m, c), v in A.items() for k in range(K)]).to_csv(OUT / "por_dobra.csv", index=False)
resumo = {"HP": HP, "n": len(t), "dificeis_pct": 100 * y.mean()}
for mod in ["rf", "lg"]:
    resumo[f"auc_{mod}"] = A[(mod, "todas")][:, 0].mean(); resumo[f"prec10_{mod}"] = A[(mod, "todas")][:, 1].mean()
resumo["auc_regra"] = A[("regra", "frequência")][:, 0].mean(); resumo["prec10_regra"] = A[("regra", "frequência")][:, 1].mean()


def tabela_lofo(mod, nomes, rotulos, col=0, escala=100):
    base = A[(mod, "todas")][:, col]
    return pd.DataFrame([{"modelo": mod, "variavel": r, "metrica": ["auc", "prec10"][col],
                          **nb(escala * (base - A[(mod, n)][:, col]))} for n, r in zip(nomes, rotulos)])


ind = pd.concat([tabela_lofo("rf", [f"sem {v}" for v in F], [NOMES[v] for v in F], c) for c in (0, 1)] +
                [tabela_lofo("lg", [f"sem {v}" for v in F], [NOMES[v] for v in F], c) for c in (0, 1)])
ind.to_csv(OUT / "A_lofo_individual.csv", index=False)
blo = tabela_lofo("rf", [f"sem {b}" for b in BLOCOS], list(BLOCOS))
ia = ind[(ind.modelo == "rf") & (ind.metrica == "auc")].set_index("variavel")["media"]
blo["soma_individuais"] = [ia[[NOMES[v] for v in c]].sum() for c in BLOCOS.values()]
blo.to_csv(OUT / "B_lofo_blocos.csv", index=False)
suf = pd.DataFrame([{"config": n, "auc": A[("rf", n)][:, 0].mean(), "prec10": A[("rf", n)][:, 1].mean(),
                     **{f"dif_{a}": b for a, b in nb(100 * (A[("rf", n)][:, 0] - A[("rf", "todas")][:, 0])).items()}}
                    for n in [f"só {v}" for v in F] + ["só {frequência, comprimento}", "todas"]])
suf.to_csv(OUT / "C_suficiencia.csv", index=False)
pfi = pd.DataFrame([{"variavel": NOMES[f], **nb(100 * A[("pfi", f)][:, 0])} for f in F])
pfi.to_csv(OUT / "D_pfi_mesmas_dobras.csv", index=False)
# estabilidade: posicao no ranking em cada uma das 10 repeticoes
base = A[("rf", "todas")][:, 0]
rk = pd.DataFrame({NOMES[v]: [np.mean(100 * (base - A[("rf", f"sem {v}")][:, 0])[REP == r]) for r in range(REPS)] for v in F})
ranks = rk.rank(axis=1, ascending=False)
pd.DataFrame({"min": ranks.min(), "max": ranks.max(), "primeiro_em": (ranks == 1).sum()}).to_csv(OUT / "E_estabilidade.csv")
rho = stats.spearmanr(ia.loc[[NOMES[v] for v in F]], pfi.set_index("variavel").loc[[NOMES[v] for v in F], "media"]).statistic
resumo["rho_lofo_pfi"] = rho
print(f"AUC RF {resumo['auc_rf']:.3f} | logistica {resumo['auc_lg']:.3f} | regra {resumo['auc_regra']:.3f}; rho LOFO x PFI = {rho:.2f}")
print(ind[ind.metrica == "auc"].pivot(index="variavel", columns="modelo", values="media").round(3).to_string())
print(ind[(ind.metrica == "auc") & (ind.modelo == "rf")][["variavel", "media", "ic_inf", "ic_sup", "p", "frac_pos"]].round(3).to_string())
print(blo[["variavel", "media", "ic_inf", "ic_sup", "p", "soma_individuais"]].round(3).to_string())
print(suf[["config", "auc", "prec10", "dif_media", "dif_p"]].round(3).to_string())
print(pfi[["variavel", "media", "ic_inf", "ic_sup", "p"]].round(3).to_string())
print(pd.read_csv(OUT / "E_estabilidade.csv").to_string())

if RAPIDO:
    json.dump(resumo, open(OUT / "resumo.json", "w"), indent=2, default=float)
    print(f"Modo rapido: pulei ruido e alvo continuo. Tempo total: {time.time()-t0:.0f}s"); sys.exit(0)
# ------------------------------------------------------------ ruido: sementes da floresta x dobras
def ruido(k, s):
    tr, te = DOBRAS[k]; yt = y[te]
    a0 = roc_auc_score(yt, rf(s).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1])
    o = {}
    for v in ["frequencia", "comprimento"]:
        fs = [f for f in F if f != v]
        o[v] = 100 * (a0 - roc_auc_score(yt, rf(s).fit(X.iloc[tr][fs], y[tr]).predict_proba(X.iloc[te][fs])[:, 1]))
    return k, s, o
rr = Parallel(n_jobs=N_JOBS)(delayed(ruido)(k, s) for k in range(10) for s in [1, 2, 3, 4, 5])
lin = []
for v in ["frequencia", "comprimento"]:
    M = np.array([[next(o[v] for k2, s2, o in rr if k2 == k and s2 == s) for s in [1, 2, 3, 4, 5]] for k in range(10)])
    vs, vd = M.var(axis=1, ddof=1).mean(), M.mean(axis=1).var(ddof=1)
    lin.append({"variavel": NOMES[v], "media": M.mean(), "dp_semente": np.sqrt(vs), "dp_dobra": np.sqrt(vd), "frac_var_semente": vs / (vs + vd)})
pd.DataFrame(lin).to_csv(OUT / "F_ruido.csv", index=False)
print(pd.DataFrame(lin).round(3).to_string())

# ------------------------------------------------------------ outro alvo: complexidade continua (regressao)
def reg(k):
    tr, te = DOBRAS[k]; o = {}
    for nome, fs in {"todas": F, **{f"sem {v}": [f for f in F if f != v] for v in F}}.items():
        p = RandomForestRegressor(n_estimators=300, random_state=SEMENTE, n_jobs=1, min_samples_leaf=HP["min_samples_leaf"],
                                  max_features=HP["max_features"]).fit(X.iloc[tr][fs], yc[tr]).predict(X.iloc[te][fs])
        o[nome] = np.sqrt(np.mean((yc[te] - p) ** 2))
    return k, o
rg = Parallel(n_jobs=N_JOBS)(delayed(reg)(k) for k in range(K))
R = pd.DataFrame({k: o for k, o in rg}).T.sort_index()
cont = pd.DataFrame([{"variavel": NOMES[v], "rmse_base": R["todas"].mean(), **nb(100 * (R[f"sem {v}"] - R["todas"]).values / R["todas"].mean())} for v in F])
cont.to_csv(OUT / "G_alvo_continuo.csv", index=False)
resumo["rmse_continuo"] = R["todas"].mean()
print("Alvo continuo (Δ RMSE em % do RMSE base):\n", cont[["variavel", "media", "p"]].round(3).to_string())
json.dump(resumo, open(OUT / "resumo.json", "w"), indent=2, default=float)
print(f"Tempo total: {time.time()-t0:.0f}s")
