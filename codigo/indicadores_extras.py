"""Indicadores complementares do artigo: quanto cada resultado é firme.

1. Concordância LOFO x permutação: intervalo de 95% para a correlação de Spearman, reamostrando as 10
   repetições da validação cruzada (cada repetição é uma partição completa em 5 dobras). O intervalo mede a
   variação devida à partição dos dados, não a variação de amostragem dos participantes ou termos.
2. Estabilidade do H de Friedman: bootstrap da amostra de treino (na leitura, por palavra), com a floresta
   treinada de novo em cada reamostra e o H avaliado sempre nos mesmos pontos.
3. Calibração na leitura: curva de confiabilidade com probabilidades fora da amostra (5 dobras agrupadas por
   palavra), erro de calibração esperado (ECE, 10 faixas de mesmo tamanho) e inclinação e intercepto de
   calibração (regressão logística de y no logito da probabilidade; o ideal é 1 e 0).

Na escrita usa a tabela corrigida, como no artigo. Rode depois de lofo_complementar.py (com SUFIXO=_corrigida),
lofo_complex.py e metodos_globais.py.
Uso: python indicadores_extras.py      (~15 min em 2 núcleos; gera resultados/saidas_extras/)
"""
import json, os, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RepeatedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from interpret.glassbox import ExplainableBoostingClassifier

warnings.filterwarnings("ignore")
TABELA = os.environ.get("TABELA_ESCRITA", "tabelas/tabela_analitica_corrigida.csv")
SUF = os.environ.get("SUFIXO", "_corrigida")
OUT = Path("resultados/saidas_extras"); OUT.mkdir(parents=True, exist_ok=True)
SEMENTE, N_JOBS = 42, 2
rng = np.random.default_rng(SEMENTE)
t0 = time.time()
res = {"tabela_escrita": TABELA}


def ic(amostras):
    a = np.asarray(amostras, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def rho_boot(lofo_dobra, pfi_dobra, rep, B=2000):
    """lofo_dobra, pfi_dobra: matrizes dobras x variáveis; rep: repetição de cada dobra."""
    obs = stats.spearmanr(lofo_dobra.mean(0), pfi_dobra.mean(0)).statistic
    reps = np.unique(rep); bs = []
    for _ in range(B):
        sel = rng.choice(reps, len(reps), replace=True)
        idx = np.concatenate([np.where(rep == r)[0] for r in sel])
        bs.append(stats.spearmanr(lofo_dobra[idx].mean(0), pfi_dobra[idx].mean(0)).statistic)
    return float(obs), ic(bs)


# ════════════════════════════════════════════════════════════ 1a. escrita: LOFO por dobra
FEATS = ["fix_dwell_med", "fix_flight_med", "fix_total_cv", "fix_ganho_pratica", "dem_idade", "dem_genero", "dem_lateralidade"]
CATS = {"dem_genero": "Male", "dem_lateralidade": "Right-Handed"}
d = pd.read_csv(TABELA, index_col="participant")
X = d[FEATS].copy()
for c, ref in CATS.items():
    X[c] = (X[c] != ref).astype(float)
X, y = X.astype(float), d["hesitacao_livre"].values
HP = json.load(open(f"resultados/saidas{SUF}/resumo.json"))["HP"]
DOBRAS = list(RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEMENTE).split(X))
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
L = np.zeros((len(DOBRAS), len(FEATS)))
for k, (tr, te) in enumerate(DOBRAS):
    rf = lambda: RandomForestRegressor(n_estimators=300, random_state=SEMENTE, n_jobs=N_JOBS, **HP)
    base = rmse(y[te], rf().fit(X.iloc[tr], y[tr]).predict(X.iloc[te]))
    for j, f in enumerate(FEATS):
        fs = [g for g in FEATS if g != f]
        L[k, j] = rmse(y[te], rf().fit(X.iloc[tr][fs], y[tr]).predict(X.iloc[te][fs])) - base
confere = pd.read_csv(f"resultados/saidas{SUF}/lofo_rf.csv").set_index("variavel")["lofo_ms"]
print("LOFO médio refeito × saída original (ms):",
      {f: (round(1000 * L[:, j].mean(), 3), round(confere[f], 3)) for j, f in enumerate(FEATS)})
P = pd.read_csv(f"resultados/saidas_comp{SUF}/C_pfi_por_dobra_7vars.csv", index_col=0)[FEATS].values
rep_e = np.arange(len(DOBRAS)) // 5
res["escrita_rho"], res["escrita_rho_ic95"] = rho_boot(L, P, rep_e)
pd.DataFrame(1000 * L, columns=FEATS).to_csv(OUT / "escrita_lofo_por_dobra_ms.csv", index_label="dobra")
print(f"escrita: rho = {res['escrita_rho']:.2f}, IC {res['escrita_rho_ic95']} ({time.time()-t0:.0f} s)")

# ════════════════════════════════════════════════════════════ 1b. leitura: LOFO e permutação por dobra
pdob = pd.read_csv("resultados/saidas_complex/por_dobra.csv")
F = ["frequencia", "comprimento", "silabas", "sigla", "nome_proprio", "tam_frase", "posicao", "forma_pt"]
auc = lambda m, c: pdob.query("modelo == @m and config == @c").sort_values("dobra")["auc"].values
base_l = auc("rf", "todas")
Ll = np.column_stack([base_l - auc("rf", f"sem {f}") for f in F])
Pl = np.column_stack([auc("pfi", f) for f in F])
rep_l = np.arange(len(base_l)) // 5
res["leitura_rho"], res["leitura_rho_ic95"] = rho_boot(Ll, Pl, rep_l)
print(f"leitura: rho = {res['leitura_rho']:.2f}, IC {res['leitura_rho_ic95']}")


# ════════════════════════════════════════════════════════════ 2. H de Friedman por bootstrap
def pd_grade(f, Xp, Xf, cols):
    n, m = len(Xp), len(Xf)
    g = pd.concat([Xf] * n, ignore_index=True)
    for c in cols:
        g[c] = np.repeat(Xp[c].values, m)
    return f(g).reshape(n, m).mean(axis=1)


def centra(v): return v - v.mean()


def H_var(f, A, c):
    fx = centra(f(A)); pj = centra(pd_grade(f, A, A, [c]))
    pr = centra(pd_grade(f, A, A, [o for o in A.columns if o != c]))
    return float(np.sqrt(max(np.sum((fx - pj - pr) ** 2), 0) / np.sum(fx ** 2)))


def H_par(f, A, a, b):
    pa, pb = centra(pd_grade(f, A, A, [a])), centra(pd_grade(f, A, A, [b]))
    pab = centra(pd_grade(f, A, A, [a, b]))
    return float(np.sqrt(max(np.sum((pab - pa - pb) ** 2), 0) / np.sum(pab ** 2)))


# escrita: mesma floresta e mesmo treino de metodos_globais.py (dummies como na ATV03)
Xd = d[FEATS].copy()
for c in ["dem_genero", "dem_lateralidade"]:
    Xd[c] = pd.Categorical(Xd[c], categories=list(Xd[c].value_counts().index))
Xd = pd.get_dummies(Xd, drop_first=True, dtype=float)
Xe_tr, _, ye_tr, _ = train_test_split(Xd, y, test_size=0.30, random_state=SEMENTE)
Xe_tr = Xe_tr.reset_index(drop=True)
CV_, VOO = "fix_total_cv", "fix_flight_med"


def h_escrita(Xa, ya, semente):
    m = RandomForestRegressor(n_estimators=500, max_features=1.0, min_samples_leaf=10, random_state=semente,
                              n_jobs=N_JOBS).fit(Xa, ya)
    return [H_var(m.predict, Xe_tr, VOO), H_var(m.predict, Xe_tr, CV_), H_par(m.predict, Xe_tr, VOO, CV_)]


obs_e = h_escrita(Xe_tr, ye_tr, SEMENTE)
boot_e = []
for b in range(100):
    i = rng.choice(len(Xe_tr), len(Xe_tr), replace=True)
    boot_e.append(h_escrita(Xe_tr.iloc[i], ye_tr[i], b))
boot_e = np.array(boot_e)
res["H_escrita"] = {n: {"H": o, "ic95": ic(boot_e[:, j])} for j, (n, o) in
                    enumerate(zip(["tempo de voo", "CV do tempo total", "par tempo de voo e CV"], obs_e))}
print(f"H escrita: {res['H_escrita']} ({time.time()-t0:.0f} s)")

# leitura: floresta da leitura, treino em 70% das palavras, H avaliado em 200 termos fixos do treino
t = pd.read_csv("tabelas/tabela_complex_reconstruida.csv", keep_default_na=False, dtype={"termo": str, "grupo": str})
Xl, yl, gl = t[F].astype(float), t["dificil"].values, t["grupo"].values
HP_L = json.load(open("resultados/saidas_complex/resumo.json"))["HP"]
tr70, _ = next(GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=SEMENTE).split(Xl, yl, gl))
aval = Xl.iloc[np.random.default_rng(SEMENTE).choice(tr70, 200, replace=False)].reset_index(drop=True)


def h_leitura(idx, semente):
    m = RandomForestClassifier(n_estimators=500, random_state=semente, n_jobs=N_JOBS, **HP_L).fit(Xl.iloc[idx], yl[idx])
    f = lambda Z: m.predict_proba(Z)[:, 1]
    return [H_var(f, aval, "frequencia"), H_var(f, aval, "comprimento"), H_par(f, aval, "comprimento", "sigla")]


obs_l = h_leitura(tr70, SEMENTE)
grupos_tr = np.unique(gl[tr70]); por_grupo = pd.Series(tr70).groupby(gl[tr70]).apply(list)
boot_l = []
for b in range(40):
    gs = rng.choice(grupos_tr, len(grupos_tr), replace=True)
    idx = np.concatenate([por_grupo[g] for g in gs])
    boot_l.append(h_leitura(idx, b))
boot_l = np.array(boot_l)
res["H_leitura"] = {n: {"H": o, "ic95": ic(boot_l[:, j])} for j, (n, o) in
                    enumerate(zip(["frequência", "comprimento", "par comprimento e sigla"], obs_l))}
res["H_leitura_nota"] = "avaliado em 200 termos fixos do treino; o artigo usa 300 termos na estimativa pontual"
print(f"H leitura: {res['H_leitura']} ({time.time()-t0:.0f} s)")


# ════════════════════════════════════════════════════════════ 3. calibração na leitura
def calibracao(yv, p, nome):
    faixas = pd.qcut(p, 10, labels=False, duplicates="drop")
    curva = pd.DataFrame({"y": yv, "p": p, "faixa": faixas}).groupby("faixa").agg(prev=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    ece = float(np.sum(curva["n"] * np.abs(curva["prev"] - curva["obs"])) / len(p))
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    cal = LogisticRegression(penalty=None, max_iter=1000).fit(lg.reshape(-1, 1), yv)
    curva.assign(modelo=nome).to_csv(OUT / f"calibracao_{nome}.csv")
    return {"ece": ece, "inclinacao": float(cal.coef_[0, 0]), "intercepto": float(cal.intercept_[0]),
            "brier": float(np.mean((p - yv) ** 2)), "curva_prev": curva["prev"].round(4).tolist(),
            "curva_obs": curva["obs"].round(4).tolist()}


MODELOS = {"Random Forest": lambda: RandomForestClassifier(n_estimators=500, random_state=SEMENTE, n_jobs=N_JOBS, **HP_L),
           "EBM": lambda: ExplainableBoostingClassifier(interactions=0, random_state=SEMENTE, n_jobs=N_JOBS),
           "Regressão logística": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))}
oof = {n: np.empty(len(Xl)) for n in MODELOS}
for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=0).split(Xl, yl, gl):
    for n, fab in MODELOS.items():
        oof[n][te] = fab().fit(Xl.iloc[tr], yl[tr]).predict_proba(Xl.iloc[te])[:, 1]
ARQ = {"Regressão logística": "logistica", "EBM": "ebm", "Random Forest": "floresta"}
res["calibracao_leitura"] = {n: calibracao(yl, p, ARQ.get(n, n.split()[0].lower())) for n, p in oof.items()}
for n, r in res["calibracao_leitura"].items():
    print(f"{n:22s} ECE {r['ece']:.3f}  inclinação {r['inclinacao']:.2f}  intercepto {r['intercepto']:+.2f}  Brier {r['brier']:.3f}")

json.dump(res, open(OUT / "resumo.json", "w"), indent=2, ensure_ascii=False)
print(f"ok ({time.time()-t0:.0f} s)")
