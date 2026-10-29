"""Atividade 02: tabela analítica da escrita e os dois modelos interpretáveis.

Os arquivos originais da ATV02 não estão no repositório. Este script refaz a atividade com o mesmo código,
que foi reaproveitado nas células C04, C05, C07 e C08 do notebook da ATV03. A única mudança é a leitura de
colunas numéricas, ajustada para o pandas 3.

Rode de dentro desta pasta:  python atv02_keyrecs.py
Gera saidas/tabela_analitica.csv e saidas/desempenho_modelos.csv.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

SEMENTE = 42
np.random.seed(SEMENTE)
warnings.filterwarnings("ignore")
DIR_DADOS = Path(__file__).resolve().parent / "../../dados/keyrecs"
DIR_SAIDA = Path(__file__).resolve().parent / "saidas"
DIR_SAIDA.mkdir(exist_ok=True)

# ---------------------------------------------------------------- carga e esquema (ATV03, C04)
def carregar_keyrecs(caminho):
    # Lê o cabeçalho bruto para (a) preservar rótulos, (b) resolver duplicatas, (c) descartar coluna sem nome.
    with open(caminho, encoding="utf-8") as fh:
        brutos = [c.strip() for c in fh.readline().rstrip("\r\n").split(",")]
    vistos, nomes = {}, []
    for c in brutos:
        k = vistos.get(c, 0)
        vistos[c] = k + 1
        nomes.append(c if k == 0 else f"{c}__{k}")
    df = pd.read_csv(caminho, header=0, names=nomes, low_memory=False)
    rotulos = dict(zip(nomes, brutos))
    for n in [n for n, r in rotulos.items() if r == ""]:
        df = df.drop(columns=[n], errors="ignore")
        rotulos.pop(n)
    return df, rotulos


def partes_rotulo(rotulo):
    # 'DU.v.v' -> ('DU', 'v', 'v');  qualquer outro formato -> None
    p = str(rotulo).split(".")
    if len(p) == 3 and p[0].upper() in ("DD", "DU", "UD", "UU"):
        return p[0].upper(), p[1], p[2]
    return None


def detectar_blocos(df, rotulos):
    # Bloco de dígrafo = dwell(k1) seguido de DD/DU/UD/UU do par k1->k2.
    cols, blocos, i = list(df.columns), [], 0
    while i < len(cols):
        p = partes_rotulo(rotulos.get(cols[i], ""))
        if p and p[0] == "DU" and p[1] == p[2] and i + 4 < len(cols):
            seq = [partes_rotulo(rotulos.get(cols[i + k], "")) for k in range(1, 5)]
            if all(seq) and [s[0] for s in seq] == ["DD", "DU", "UD", "UU"]:
                blocos.append({"tecla": p[1], "par": rotulos[cols[i + 1]], "hold": cols[i],
                               "DD": cols[i + 1], "DU": cols[i + 2], "UD": cols[i + 3], "UU": cols[i + 4]})
                i += 5
                continue
        i += 1
    return blocos


demo = pd.read_csv(DIR_DADOS / "demographics.csv")
fixo, rot_fixo = carregar_keyrecs(DIR_DADOS / "fixed-text.csv")
livre, rot_livre = carregar_keyrecs(DIR_DADOS / "free-text.csv")

# ── texto fixo ───────────────────────────────────────────────────────────
HOLDS_FIXO = [c for c in fixo.columns if partes_rotulo(rot_fixo[c])
              and partes_rotulo(rot_fixo[c])[0] == "DU"
              and partes_rotulo(rot_fixo[c])[1] == partes_rotulo(rot_fixo[c])[2]]
BLOCOS_FIXO = detectar_blocos(fixo, rot_fixo)
col_sessao_fixo = next(c for c in fixo.columns if c.lower().startswith("session"))
col_total = next(c for c in fixo.columns if "total" in rot_fixo[c].lower())

# ── texto livre ──────────────────────────────────────────────────────────
col_k1 = next(c for c in livre.columns if c.lower() in ("key1", "k1"))
col_k2 = next(c for c in livre.columns if c.lower() in ("key2", "k2"))
COLS_LAT_LIVRE = [c for c in livre.columns if partes_rotulo(rot_livre[c])]
HOLD_LIVRE = next(c for c in COLS_LAT_LIVRE
                  if partes_rotulo(rot_livre[c])[1] == partes_rotulo(rot_livre[c])[2])
COL_DD_LIVRE = next(c for c in COLS_LAT_LIVRE if partes_rotulo(rot_livre[c])[0] == "DD")

# ── saneamento numérico (linhas desalinhadas viram NaN) ──────────────────
cols_num_fixo = (HOLDS_FIXO + [c for b in BLOCOS_FIXO for c in (b["DD"], b["DU"], b["UD"], b["UU"])]
                 + [col_total])
for df, cols in ((livre, COLS_LAT_LIVRE), (fixo, cols_num_fixo)):
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

assert HOLDS_FIXO and BLOCOS_FIXO and HOLD_LIVRE and COL_DD_LIVRE, "Esquema não detectado."


# ---------------------------------------------------------------- tabela analítica (ATV03, C05)
LIMITE_PAUSA = 10.0    # s — acima disso, interrupção da sessão
LIMITE_SENHA = 60.0    # s

# ── Texto livre: flight por dígrafo e filtro ─────────────────────────────
livre["flight"] = livre[COL_DD_LIVRE] - livre[HOLD_LIVRE]
mask_livre = (livre["flight"].abs() <= LIMITE_PAUSA) & livre["flight"].notna() \
             & (livre[HOLD_LIVRE].abs() <= LIMITE_PAUSA)
livre_ok = livre.loc[mask_livre].copy()

# ── Texto fixo: filtro + flight explícito por dígrafo ────────────────────
mask_fixo = fixo[col_total].between(0, LIMITE_SENHA) & fixo[col_total].notna()
for c in HOLDS_FIXO:
    mask_fixo &= fixo[c].abs() <= LIMITE_PAUSA
fixo_ok = fixo.loc[mask_fixo].copy()

COLS_FLIGHT_FIXO = []
for b in BLOCOS_FIXO:
    nome = "flight." + b["par"].split(".", 1)[1]
    fixo_ok[nome] = fixo_ok[b["DD"]] - fixo_ok[b["hold"]]
    COLS_FLIGHT_FIXO.append(nome)


def mediana_geral(df, colunas, chave="participant", limite=LIMITE_PAUSA):
    # Mediana de TODOS os valores das colunas indicadas, por participante, sem implausíveis.
    longo = df.melt(id_vars=[chave], value_vars=list(colunas), value_name="v")
    longo = longo[longo["v"].abs() <= limite]
    return longo.groupby(chave)["v"].median()


# ── Features da tarefa MOTORA (explicativas) ─────────────────────────────
fix_dwell_med = mediana_geral(fixo_ok, HOLDS_FIXO).rename("fix_dwell_med")
fix_flight_med = mediana_geral(fixo_ok, COLS_FLIGHT_FIXO).rename("fix_flight_med")
g = fixo_ok.groupby("participant")[col_total]
fix_total_med = g.median().rename("fix_total_med")
fix_total_cv = (g.std() / g.mean()).rename("fix_total_cv")
med_sessao = fixo_ok.groupby(["participant", col_sessao_fixo])[col_total].median().unstack()
s_ord = sorted(med_sessao.columns)
fix_ganho_pratica = ((med_sessao[s_ord[0]] - med_sessao[s_ord[1]]) / med_sessao[s_ord[0]]).rename("fix_ganho_pratica")
features_fixo = pd.concat([fix_dwell_med, fix_flight_med, fix_total_med, fix_total_cv, fix_ganho_pratica], axis=1)

# ── Resposta (tarefa LINGUÍSTICA) ────────────────────────────────────────
hesitacao_livre = livre_ok.groupby("participant")["flight"].median().rename("hesitacao_livre")
p90_global = livre_ok["flight"].quantile(0.90)
taxa_pausa_longa = (livre_ok.assign(_l=livre_ok["flight"] > p90_global)
                    .groupby("participant")["_l"].mean().rename("taxa_pausa_longa"))
eh_rev = (livre_ok[col_k1].astype(str).isin(["Backspace", "Delete"])
          | livre_ok[col_k2].astype(str).isin(["Backspace", "Delete"]))
taxa_revisao = livre_ok.assign(_r=eh_rev).groupby("participant")["_r"].mean().rename("taxa_revisao")
n_digrafos = livre_ok.groupby("participant").size().rename("n_digrafos_livre")

# ── Montagem ─────────────────────────────────────────────────────────────
dados = (demo.set_index("participant")
         .rename(columns={"age": "dem_idade", "gender": "dem_genero", "handedness": "dem_lateralidade"})
         [["dem_idade", "dem_genero", "dem_lateralidade", "nationality"]]
         .join(features_fixo, how="inner").join(hesitacao_livre, how="inner")
         .join(taxa_pausa_longa, how="inner").join(taxa_revisao, how="inner").join(n_digrafos, how="inner"))
antes = len(dados)
dados = dados[dados["hesitacao_livre"] > 0]
dados = dados.dropna(subset=["hesitacao_livre", "fix_dwell_med", "fix_flight_med",
                             "fix_total_cv", "fix_ganho_pratica", "dem_idade"])

print(f"Eventos usados — texto livre: {len(livre_ok):,}   texto fixo (repetições): {len(fixo_ok):,}")
print(f"Participantes: {antes} -> {len(dados)}  (removidos {antes - len(dados)} por hesitação <= 0)")
assert len(dados) >= 30, "Poucos participantes — revisar mapeamento/filtros."


dados.to_csv(DIR_SAIDA / "tabela_analitica.csv")

# ---------------------------------------------------------------- divisão e matriz de desenho (ATV03, C07)
ALVO = "hesitacao_livre"
EXPLICATIVAS = ["fix_dwell_med", "fix_flight_med", "fix_total_cv", "fix_ganho_pratica",
                "dem_idade", "dem_genero", "dem_lateralidade"]
X_bruto = dados[EXPLICATIVAS].copy()
y_bruto = dados[ALVO].copy()
X_tr, X_te, y_tr, y_te = train_test_split(X_bruto, y_bruto, test_size=0.30, random_state=SEMENTE)
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEMENTE)
CATEGORIAS = {c: list(X_bruto[c].value_counts().index)
              for c in X_bruto.select_dtypes(include=["object", "string"]).columns}


def montar_desenho(X):
    # Dummies com a categoria mais frequente como referência.
    X = X.copy()
    for c, cats in CATEGORIAS.items():
        X[c] = pd.Categorical(X[c], categories=cats)
    return pd.get_dummies(X, drop_first=True, dtype=float)


D_tr = montar_desenho(X_tr)
D_te = montar_desenho(X_te).reindex(columns=D_tr.columns, fill_value=0.0)
COLS = list(D_tr.columns)
COLS_CONT = [c for c in COLS if not set(D_tr[c].unique()) <= {0.0, 1.0}]
D_all, y_all = pd.concat([D_tr, D_te]), pd.concat([y_tr, y_te])

# ---------------------------------------------------------------- modelos (ATV03, C08)
def novo_linear():
    # z-score só nas contínuas; as dummies ficam 0/1.
    ct = ColumnTransformer([("z", StandardScaler(), COLS_CONT)], remainder="passthrough",
                           verbose_feature_names_out=False).set_output(transform="pandas")
    return make_pipeline(ct, LinearRegression())


# árvore: poda por custo-complexidade com a regra de 1 erro padrão
arv_base = DecisionTreeRegressor(random_state=SEMENTE, min_samples_leaf=10, max_depth=5)
alphas = np.unique(arv_base.cost_complexity_pruning_path(D_tr, y_tr).ccp_alphas.clip(min=0))
poda = []
for a in alphas:
    arv_a = DecisionTreeRegressor(random_state=SEMENTE, min_samples_leaf=10, max_depth=5, ccp_alpha=a)
    sc = cross_val_score(arv_a, D_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error")
    poda.append({"alpha": a, "folhas": arv_a.fit(D_tr, y_tr).get_n_leaves(),
                 "rmse_cv": -sc.mean(), "ep": sc.std() / np.sqrt(len(sc))})
poda = pd.DataFrame(poda)
i_min = poda["rmse_cv"].idxmin()
cand = poda[poda["rmse_cv"] <= poda.loc[i_min, "rmse_cv"] + poda.loc[i_min, "ep"]]
ALPHA_ARV = float(poda.loc[cand["folhas"].idxmin(), "alpha"])


def nova_arvore():
    return DecisionTreeRegressor(random_state=SEMENTE, min_samples_leaf=10, max_depth=5, ccp_alpha=ALPHA_ARV)


FABRICAS = {"Mediana": lambda: DummyRegressor(strategy="median"),
            "Regressão linear": novo_linear,
            "Árvore podada": nova_arvore}
linhas = []
for nome, fab in FABRICAS.items():
    r = cross_validate(fab(), D_all, y_all, cv=cv,
                       scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"})
    linhas.append({"modelo": nome, "RMSE": -r["test_rmse"].mean(), "RMSE_dp": r["test_rmse"].std(),
                   "MAE": -r["test_mae"].mean(), "R2": r["test_r2"].mean()})
desempenho = pd.DataFrame(linhas).set_index("modelo")
desempenho.to_csv(DIR_SAIDA / "desempenho_modelos.csv")
print("\nValidação cruzada 5 x 10 com os", len(D_all), "participantes")
print(desempenho.round(4).to_string())

lin = novo_linear().fit(D_tr, y_tr)
coef = pd.Series(lin[-1].coef_, index=lin[0].get_feature_names_out()).sort_values(key=abs, ascending=False)
print("\nRegressão linear, coeficientes em segundos por desvio padrão (dummies: diferença para a referência)")
print(coef.round(4).to_string())
print("\nÁrvore podada, alpha =", round(ALPHA_ARV, 6))
print(export_text(nova_arvore().fit(D_tr, y_tr), feature_names=COLS, decimals=3))
