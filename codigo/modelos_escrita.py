"""Escrita (KeyRecs): desempenho dos modelos (Tabela II do artigo), com o protocolo da ATV03 (célula C08).

Linha de base (mediana), regressão linear (contínuas em z-score), árvore podada (custo-complexidade, regra do
1-EP, como na ATV02) e Random Forest (500 árvores; min_samples_leaf e max_features por busca em grade nos 70% de
treino). Desempenho: validação cruzada 5x10 sobre os 96 participantes; teste t corrigido de Nadeau & Bengio.

Uso: python modelos_escrita.py                                         (tabela do grupo; reproduz a ATV03)
     TABELA_ESCRITA=tabelas/tabela_analitica_corrigida.csv SUFIXO=_corrigida python modelos_escrita.py
Gera resultados/saidas_modelos_escrita{SUFIXO}/.
"""
import json, os, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")
TABELA = os.environ.get("TABELA_ESCRITA", "tabelas/tabela_analitica.csv")
SUF = os.environ.get("SUFIXO", "")
OUT = Path("resultados/saidas_modelos_escrita" + SUF); OUT.mkdir(parents=True, exist_ok=True)
SEMENTE = 42
EXP = ["fix_dwell_med", "fix_flight_med", "fix_total_cv", "fix_ganho_pratica", "dem_idade", "dem_genero", "dem_lateralidade"]

dados = pd.read_csv(TABELA, index_col=0)
X, y = dados[EXP].copy(), dados["hesitacao_livre"].copy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=SEMENTE)
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEMENTE)
CAT = {c: list(X[c].value_counts().index) for c in X.select_dtypes(include=["object", "string"]).columns}


def desenho(Xp):
    Xp = Xp.copy()
    for c, cats in CAT.items():
        Xp[c] = pd.Categorical(Xp[c], categories=cats)
    return pd.get_dummies(Xp, drop_first=True, dtype=float)


D_tr, D_te = desenho(X_tr), desenho(X_te).reindex(columns=desenho(X_tr).columns, fill_value=0.0)
COLS = list(D_tr.columns)
CONT = [c for c in COLS if not set(D_tr[c].unique()) <= {0.0, 1.0}]
D_all, y_all = pd.concat([D_tr, D_te]), pd.concat([y_tr, y_te])

# árvore: poda custo-complexidade com a regra do 1-EP (ATV02, C27)
base = DecisionTreeRegressor(random_state=SEMENTE, min_samples_leaf=10, max_depth=5)
alphas = np.unique(base.cost_complexity_pruning_path(D_tr, y_tr).ccp_alphas.clip(min=0))
poda = []
for a in alphas:
    arv = DecisionTreeRegressor(random_state=SEMENTE, min_samples_leaf=10, max_depth=5, ccp_alpha=a)
    sc = cross_val_score(arv, D_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error")
    poda.append({"alpha": a, "folhas": arv.fit(D_tr, y_tr).get_n_leaves(), "rmse": -sc.mean(), "ep": sc.std() / np.sqrt(len(sc))})
poda = pd.DataFrame(poda)
i = poda["rmse"].idxmin()
ALPHA = float(poda.loc[poda[poda["rmse"] <= poda.loc[i, "rmse"] + poda.loc[i, "ep"]]["folhas"].idxmin(), "alpha"])

busca = GridSearchCV(RandomForestRegressor(n_estimators=500, random_state=SEMENTE, n_jobs=1),
                     {"min_samples_leaf": [3, 5, 10, 15, 20], "max_features": [0.4, 0.7, 1.0]},
                     cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEMENTE),
                     scoring="neg_root_mean_squared_error", n_jobs=-1).fit(D_tr, y_tr)
HP = dict(busca.best_params_)

MODELOS = {
    "Linha de base (mediana)": lambda: DummyRegressor(strategy="median"),
    "Regressão linear": lambda: make_pipeline(ColumnTransformer([("z", StandardScaler(), CONT)], remainder="passthrough"),
                                              LinearRegression()),
    "Árvore podada": lambda: DecisionTreeRegressor(random_state=SEMENTE, min_samples_leaf=10, max_depth=5, ccp_alpha=ALPHA),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=500, random_state=SEMENTE, n_jobs=-1, **HP),
}
linhas, dobras = [], {}
for nome, fab in MODELOS.items():
    r = cross_validate(fab(), D_all, y_all, cv=cv, scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2",
                                                            "mae": "neg_mean_absolute_error"})
    dobras[nome] = -r["test_rmse"]
    linhas.append({"modelo": nome, "RMSE": -r["test_rmse"].mean(), "RMSE_dp": r["test_rmse"].std(),
                   "MAE": -r["test_mae"].mean(), "R2": r["test_r2"].mean()})
tab = pd.DataFrame(linhas).set_index("modelo")
tab.to_csv(OUT / "desempenho.csv")


def teste(a, b, razao=0.25):
    d = np.asarray(a) - np.asarray(b); k = len(d)
    t = d.mean() / np.sqrt((1 / k + razao) * d.var(ddof=1))
    return float(2 * stats.t.sf(abs(t), k - 1))


arv = MODELOS["Árvore podada"]().fit(D_tr, y_tr)
res = {"tabela": TABELA, "HP_rf": HP, "alpha_arvore": ALPHA, "folhas_arvore": int(arv.get_n_leaves()),
       "arvore": export_text(arv, feature_names=COLS, decimals=4),
       "p_linear_vs_rf": teste(dobras["Regressão linear"], dobras["Random Forest"]),
       "p_arvore_vs_rf": teste(dobras["Árvore podada"], dobras["Random Forest"]),
       "r_voo_hesitacao": float(dados[["fix_flight_med", "hesitacao_livre"]].corr().iloc[0, 1])}
json.dump(res, open(OUT / "resumo.json", "w"), indent=2, ensure_ascii=False)
print(tab.round(4)); print({k: v for k, v in res.items() if k != "arvore"}); print(res["arvore"])
