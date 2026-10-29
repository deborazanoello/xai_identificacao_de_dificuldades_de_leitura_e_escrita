"""Leitura (CompLex): desempenho dos modelos interpretáveis e caixa-preta (Tabela III do artigo).

Tabela: tabelas/tabela_complex_reconstruida.csv (monta_tabela_complex.py), 8.579 termos de uma palavra.
Alvo: termo difícil (complexidade >= 0,5). Validação: 5 dobras agrupadas por palavra x 10 repetições,
as mesmas dobras de lofo_complex.py. Métricas: AUC, precisão nos 10% (proporção de difíceis entre os 10%
de maior probabilidade), precisão média (área sob a curva precisão-revocação, adequada à classe rara) e
Brier (erro quadrático da probabilidade; mede calibração; não se aplica à regra, que não dá probabilidade). Hiperparâmetros escolhidos uma vez, em 70% das palavras (mesma divisão do LOFO).
Comparações: teste t corrigido de Nadeau & Bengio (2003), razão teste/treino 0,25.

Uso: python modelos_leitura.py        (~15 min em 2 núcleos; gera resultados/saidas_leitura/)
"""
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
OUT = Path("resultados/saidas_leitura"); OUT.mkdir(parents=True, exist_ok=True)
SEMENTE, N_JOBS, REPS, RAZAO = 42, 2, 10, 0.25
F = ["frequencia", "comprimento", "silabas", "sigla", "nome_proprio", "tam_frase", "posicao", "forma_pt"]
t = pd.read_csv("tabelas/tabela_complex_reconstruida.csv", keep_default_na=False, dtype={"termo": str, "grupo": str})
X, y, g = t[F].astype(float), t["dificil"].values, t["grupo"].values
DOBRAS = [(tr, te) for rep in range(REPS) for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=rep).split(X, y, g)]
HP_RF = json.load(open("resultados/saidas_complex/resumo.json"))["HP"] if Path("resultados/saidas_complex/resumo.json").exists() \
    else {"max_features": 0.3, "min_samples_leaf": 10}


def prec10(yt, p):
    k = int(np.ceil(0.10 * len(p))); return yt[np.argsort(-p, kind="stable")[:k]].mean()


def nb(d):
    d = np.asarray(d, float); k = len(d)
    ep = np.sqrt((1 / k + RAZAO) * d.var(ddof=1)); tt = d.mean() / ep if ep > 0 else 0.0
    return {"dif_media": d.mean(), "p": 2 * stats.t.sf(abs(tt), k - 1)}


t0 = time.time()
# ------------------------------------------------ hiperparâmetros da árvore e do XGBoost (70% das palavras)
tr70, _ = next(GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=SEMENTE).split(X, y, g))
X70, y70, g70 = X.iloc[tr70], y[tr70], g[tr70]
internas = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEMENTE).split(X70, y70, g70))


def auc_interna(fab):
    return np.array([roc_auc_score(y70[te], fab().fit(X70.iloc[tr], y70[tr]).predict_proba(X70.iloc[te])[:, 1])
                     for tr, te in internas])


# árvore: menor número de folhas cuja AUC fica a 1 erro-padrão da melhor (regra do 1-EP)
poda = []
for folhas in range(2, 31):
    a = auc_interna(lambda: DecisionTreeClassifier(max_leaf_nodes=folhas, min_samples_leaf=20, random_state=SEMENTE))
    poda.append({"folhas": folhas, "auc": a.mean(), "ep": a.std(ddof=1) / np.sqrt(len(a))})
poda = pd.DataFrame(poda)
melhor = poda.loc[poda["auc"].idxmax()]
FOLHAS = int(poda.loc[poda["auc"] >= melhor["auc"] - melhor["ep"], "folhas"].min())
# XGBoost: pequena busca em grade
grade = []
for prof in [2, 3, 4, 6]:
    for lr in [0.03, 0.1]:
        a = auc_interna(lambda: XGBClassifier(n_estimators=300, max_depth=prof, learning_rate=lr, subsample=0.8,
                                              colsample_bytree=0.8, random_state=SEMENTE, n_jobs=N_JOBS, verbosity=0))
        grade.append({"max_depth": prof, "learning_rate": lr, "auc": a.mean()})
grade = pd.DataFrame(grade)
HP_XGB = grade.loc[grade["auc"].idxmax(), ["max_depth", "learning_rate"]].to_dict()
HP_XGB = {"max_depth": int(HP_XGB["max_depth"]), "learning_rate": float(HP_XGB["learning_rate"])}
print(f"árvore: {FOLHAS} folhas · XGBoost: {HP_XGB} · RF: {HP_RF} ({time.time()-t0:.0f} s)")

MODELOS = {
    "Regressão logística": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "Árvore podada": lambda: DecisionTreeClassifier(max_leaf_nodes=FOLHAS, min_samples_leaf=20, random_state=SEMENTE),
    "EBM": lambda: ExplainableBoostingClassifier(interactions=0, random_state=SEMENTE, n_jobs=1),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=500, random_state=SEMENTE, n_jobs=1, **HP_RF),
    "XGBoost": lambda: XGBClassifier(n_estimators=300, subsample=0.8, colsample_bytree=0.8, random_state=SEMENTE,
                                     n_jobs=1, verbosity=0, **HP_XGB),
}


def uma_dobra(k):
    tr, te = DOBRAS[k]; yt = y[te]
    regra = -X.iloc[te]["frequencia"].values
    linhas = [{"dobra": k, "modelo": "Regra das palavras raras", "auc": roc_auc_score(yt, regra),
               "prec10": prec10(yt, regra), "pr_auc": average_precision_score(yt, regra), "brier": np.nan}]
    for nome, fab in MODELOS.items():
        p = fab().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        linhas.append({"dobra": k, "modelo": nome, "auc": roc_auc_score(yt, p), "prec10": prec10(yt, p),
                       "pr_auc": average_precision_score(yt, p), "brier": brier_score_loss(yt, p)})
    return linhas


res = pd.DataFrame([l for ls in Parallel(n_jobs=N_JOBS)(delayed(uma_dobra)(k) for k in range(len(DOBRAS))) for l in ls])
res.to_csv(OUT / "desempenho_por_dobra.csv", index=False)
ordem = ["Regra das palavras raras"] + list(MODELOS)
tab = res.groupby("modelo")[["auc", "prec10", "pr_auc", "brier"]].agg(["mean", "std"]).reindex(ordem)
tab.columns = ["auc", "auc_dp", "prec10", "prec10_dp", "pr_auc", "pr_auc_dp", "brier", "brier_dp"]
tab["prevalencia"] = y.mean()
tab.to_csv(OUT / "desempenho.csv")
print(tab.round(3))

# comparações pareadas entre os quatro melhores modelos, nas três métricas (teste t corrigido)
from itertools import combinations
testes = []
for metrica in ["auc", "pr_auc", "brier"]:
    larga = res.pivot(index="dobra", columns="modelo", values=metrica)
    for a, b in combinations(["Random Forest", "EBM", "Regressão logística", "XGBoost"], 2):
        t = nb(larga[a] - larga[b]); testes.append({"metrica": metrica, "a": a, "b": b, **t})
    t = nb(larga["EBM"] - larga["Árvore podada"]); testes.append({"metrica": metrica, "a": "EBM", "b": "Árvore podada", **t})
testes = pd.DataFrame(testes); testes.to_csv(OUT / "testes_pareados.csv", index=False)
print(testes.round(4).to_string(index=False))
json.dump({"folhas_arvore": FOLHAS, "HP_xgb": HP_XGB, "HP_rf": HP_RF, "testes": testes.to_dict("records"),
           "poda": poda.to_dict("records"), "grade_xgb": grade.to_dict("records")},
          open(OUT / "resumo.json", "w"), indent=2, ensure_ascii=False)
print(f"ok ({time.time()-t0:.0f} s)")
