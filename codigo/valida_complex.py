"""Teste rápido da tabela da leitura (CompLex): 10 dobras agrupadas por palavra, em vez de 50.

Serve para conferir, em cerca de 1 minuto, se a tabela montada por monta_tabela_complex.py dá os mesmos
resultados que a rodada completa de modelos_leitura.py (resultados/saidas_leitura/desempenho.csv). Com 10 dobras, as
médias podem diferir da rodada completa na terceira casa decimal.
"""
import time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
t = pd.read_csv("tabelas/tabela_complex_reconstruida.csv", keep_default_na=False, dtype={"termo": str, "grupo": str})
F = ["frequencia", "comprimento", "silabas", "sigla", "nome_proprio", "tam_frase", "posicao", "forma_pt"]
X, y, g = t[F].astype(float), t["dificil"].values, t["grupo"].values
print(f"{len(t)} termos; {100 * y.mean():.1f}% difíceis; r(frequência, complexidade) = "
      f"{t[['frequencia', 'complexidade']].corr().iloc[0, 1]:.2f}")


def prec10(yt, p):
    k = int(np.ceil(0.10 * len(p))); return yt[np.argsort(-p, kind="stable")[:k]].mean()


res = {"Regra das palavras raras": [], "Regressão logística": [], "Random Forest": []}
t0 = time.time()
for rep in range(2):
    for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=rep).split(X, y, g):
        yt = y[te]; regra = -X.iloc[te]["frequencia"].values
        res["Regra das palavras raras"].append((roc_auc_score(yt, regra), prec10(yt, regra)))
        p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        res["Regressão logística"].append((roc_auc_score(yt, p), prec10(yt, p)))
        p = RandomForestClassifier(n_estimators=300, max_features=0.3, min_samples_leaf=10, n_jobs=2,
                                   random_state=42).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        res["Random Forest"].append((roc_auc_score(yt, p), prec10(yt, p)))

ref = pd.read_csv("resultados/saidas_leitura/desempenho.csv", index_col=0) if Path("resultados/saidas_leitura/desempenho.csv").exists() else None
for k, v in res.items():
    a = np.array(v)
    extra = f"   | rodada completa: AUC {ref.loc[k, 'auc']:.3f}  prec10 {ref.loc[k, 'prec10']:.3f}" if ref is not None else ""
    print(f"{k:26s} AUC {a[:, 0].mean():.3f}  prec10 {a[:, 1].mean():.3f}{extra}")
print(f"({time.time() - t0:.0f} s)")
