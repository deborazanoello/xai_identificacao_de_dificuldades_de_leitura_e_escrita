"""ATV04 - Testes complementares do LOFO no KeyRecs.

Complementa lofo_keyrecs.py com o MESMO problema, as MESMAS 50 dobras
(RepeatedKFold 5x10, semente 42) e os MESMOS hiperparametros da floresta
(lidos de resultados/saidas/resumo.json, gerado pelo script original).

Analises (cada uma salva um CSV em resultados/saidas_comp/):
  A. LOFO por blocos de variaveis
  B. Suficiencia: modelos com uma variavel so (o "contrario" do LOFO)
  C. PFI nas mesmas 50 dobras (comparacao pareada com o LOFO)
  D. Outras perdas: MAE e RMSE nos 25% mais hesitantes
  E. Ruido da floresta x ruido das dobras (varia so a semente da floresta)
  F. LOFO com hiperparametros reajustados sem cada variavel
  G. Outros alvos: taxa de pausas longas e taxa de revisao

Uso:  python lofo_complementar.py            (roda tudo)
      python lofo_complementar.py A B C      (roda so as partes escolhidas)
Intervalos e p-valores: teste t corrigido de Nadeau & Bengio (2003).
"""
import json
import sys
import time
import warnings
import os
TABELA = os.environ.get("TABELA_ESCRITA", "tabelas/tabela_analitica.csv")   # tabela da escrita (ou a corrigida)
SUF = os.environ.get("SUFIXO", "")                                  # sufixo das pastas de saída
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split

warnings.filterwarnings("ignore")
SEMENTE = 42
N_JOBS = 2
OUT = Path("resultados/saidas_comp" + SUF)
OUT.mkdir(parents=True, exist_ok=True)

FEATS = ["fix_dwell_med", "fix_flight_med", "fix_total_cv", "fix_ganho_pratica",
         "dem_idade", "dem_genero", "dem_lateralidade"]
CATS = {"dem_genero": "Male", "dem_lateralidade": "Right-Handed"}
NOMES = {"fix_flight_med": "tempo de voo", "fix_total_cv": "CV do tempo total",
         "fix_dwell_med": "pressionamento", "fix_ganho_pratica": "ganho com a prática",
         "dem_idade": "idade", "dem_genero": "gênero", "dem_lateralidade": "lateralidade",
         "fix_total_med": "tempo total"}

dados = pd.read_csv(TABELA, index_col="participant")


def desenho(X):
    X = X.copy()
    for c, ref in CATS.items():
        if c in X:
            X[c] = (X[c] != ref).astype(float)
    return X.astype(float)


X_all = desenho(dados[FEATS + ["fix_total_med"]])
Y = {a: dados[a] for a in ["hesitacao_livre", "taxa_pausa_longa", "taxa_revisao"]}
DOBRAS = list(RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEMENTE).split(X_all))
K = len(DOBRAS)
RAZAO = 0.25                      # n_teste / n_treino com 5 dobras
HP = json.load(open(f"resultados/saidas{SUF}/resumo.json"))["HP"]


def rf(hp=None, semente=SEMENTE):
    return RandomForestRegressor(n_estimators=300, random_state=semente, n_jobs=1, **(hp or HP))


# ---------------------------------------------------------------- metricas e testes
def metricas(y_te, p, y_tr):
    e = np.asarray(y_te) - np.asarray(p)
    alto = np.asarray(y_te) >= np.quantile(y_tr, 0.75)
    return {"rmse": np.sqrt(np.mean(e ** 2)), "mae": np.mean(np.abs(e)),
            "rmse_top25": np.sqrt(np.mean(e[alto] ** 2)) if alto.any() else np.nan}


def nb(d):
    """Media, IC 95% e p do teste t corrigido de Nadeau & Bengio para diferencas por dobra."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    k = len(d)
    ep = np.sqrt((1 / k + RAZAO) * d.var(ddof=1))
    t = d.mean() / ep if ep > 0 else 0.0
    m = stats.t.ppf(0.975, k - 1) * ep
    return {"media": d.mean(), "dp": d.std(ddof=1), "ic_inf": d.mean() - m, "ic_sup": d.mean() + m,
            "p": 2 * stats.t.sf(abs(t), k - 1), "frac_pos": float((d > 0).mean())}


def ajusta_prediz(feats, dobra, alvo="hesitacao_livre", hp=None, semente=SEMENTE, modelo=None):
    tr, te = DOBRAS[dobra]
    y = Y[alvo]
    m = (modelo or rf(hp, semente)).fit(X_all.iloc[tr][feats], y.iloc[tr])
    return m.predict(X_all.iloc[te][feats]), m


def avalia(configs, alvo="hesitacao_livre", hp_por_config=None, semente=SEMENTE):
    """configs: {nome: lista de variaveis}. Devolve {nome: DataFrame (dobra x metricas)}."""
    hp_por_config = hp_por_config or {}

    def uma(dobra, nome, feats):
        tr, te = DOBRAS[dobra]
        if len(feats) == 0:
            p = np.full(len(te), Y[alvo].iloc[tr].median())
        else:
            p, _ = ajusta_prediz(feats, dobra, alvo, hp_por_config.get(nome), semente)
        return nome, dobra, metricas(Y[alvo].iloc[te], p, Y[alvo].iloc[tr])

    res = Parallel(n_jobs=N_JOBS)(delayed(uma)(k, n, f) for k in range(K) for n, f in configs.items())
    tabs = {n: pd.DataFrame(index=range(K), columns=["rmse", "mae", "rmse_top25"], dtype=float)
            for n in configs}
    for n, k, m in res:
        tabs[n].loc[k] = m
    return tabs


def compara(tabs, ref, nomes, metrica="rmse", escala=1000.0):
    linhas = []
    for n in nomes:
        d = escala * (tabs[n][metrica] - tabs[ref][metrica]).values
        linhas.append({"config": n, "metrica": metrica, **nb(d)})
    return pd.DataFrame(linhas)


resumo = {"HP": HP, "K": K}
partes = set(sys.argv[1:]) or set("ABCDEFG")
t_ini = time.time()

# ================================================================ A, B, D (mesmos ajustes)
if partes & set("ABD"):
    t0 = time.time()
    configs = {"todas": FEATS}
    configs.update({f"sem {v}": [f for f in FEATS if f != v] for v in FEATS})
    BLOCOS = {
        "{demográficas}": ["dem_idade", "dem_genero", "dem_lateralidade"],
        "{motoras}": ["fix_dwell_med", "fix_flight_med", "fix_total_cv", "fix_ganho_pratica"],
        "{motoras sem voo}": ["fix_dwell_med", "fix_total_cv", "fix_ganho_pratica"],
        "{voo, pressionamento}": ["fix_flight_med", "fix_dwell_med"],
    }
    configs.update({f"sem {b}": [f for f in FEATS if f not in cols] for b, cols in BLOCOS.items()})
    configs.update({f"só {v}": [v] for v in FEATS})
    configs["só {voo, CV}"] = ["fix_flight_med", "fix_total_cv"]
    configs["mediana"] = []
    tabs = avalia(configs)
    print(f"[A/B/D] {len(configs)} configurações x {K} dobras em {time.time()-t0:.0f}s")
    base = tabs["todas"]
    resumo["rmse_todas_ms"] = 1000 * base["rmse"].mean()
    resumo["rmse_mediana_ms"] = 1000 * tabs["mediana"]["rmse"].mean()

    # A. LOFO individual (reproduz o original) + blocos
    ind = compara(tabs, "todas", [f"sem {v}" for v in FEATS])
    ind["variavel"] = [NOMES[v] for v in FEATS]
    blo = compara(tabs, "todas", [f"sem {b}" for b in BLOCOS])
    blo["variavel"] = list(BLOCOS)
    blo["n_vars"] = [len(c) for c in BLOCOS.values()]
    blo["soma_individuais"] = [ind.set_index("config").loc[[f"sem {v}" for v in c], "media"].sum()
                               for c in BLOCOS.values()]
    ind.to_csv(OUT / "A_lofo_individual.csv", index=False)
    blo.to_csv(OUT / "A_lofo_blocos.csv", index=False)
    print("\nA. LOFO individual (ms):\n", ind[["variavel", "media", "ic_inf", "ic_sup", "p", "frac_pos"]].round(3).to_string())
    print("\nA. LOFO por blocos (ms):\n", blo[["variavel", "media", "ic_inf", "ic_sup", "p", "soma_individuais"]].round(3).to_string())

    # B. suficiencia: RMSE de cada modelo reduzido e diferenca para o completo
    suf = []
    for n in [f"só {v}" for v in FEATS] + ["só {voo, CV}", "mediana", "todas"]:
        r = {"config": n, "rmse_ms": 1000 * tabs[n]["rmse"].mean()}
        if n != "todas":
            r.update({f"dif_{k}": v for k, v in nb(1000 * (tabs[n]["rmse"] - base["rmse"]).values).items()})
        suf.append(r)
    suf = pd.DataFrame(suf)
    suf.to_csv(OUT / "B_suficiencia.csv", index=False)
    print("\nB. Suficiência (RMSE em ms; dif = modelo reduzido - completo):\n",
          suf[["config", "rmse_ms", "dif_media", "dif_ic_inf", "dif_ic_sup", "dif_p"]].round(3).to_string())

    # D. outras perdas
    outras = []
    for met in ["rmse", "mae", "rmse_top25"]:
        c = compara(tabs, "todas", [f"sem {v}" for v in FEATS], metrica=met)
        c["variavel"] = [NOMES[v] for v in FEATS]
        c["base_ms"] = 1000 * base[met].mean()
        outras.append(c)
    outras = pd.concat(outras)
    outras.to_csv(OUT / "D_outras_perdas.csv", index=False)
    print("\nD. LOFO com outras perdas (ms):\n",
          outras.pivot(index="variavel", columns="metrica", values="media").round(3).to_string())
    for met in ["mae", "rmse_top25"]:
        resumo[f"base_{met}_ms"] = 1000 * base[met].mean()

# ================================================================ C. PFI nas mesmas dobras
if "C" in partes:
    t0 = time.time()

    def pfi_dobra(dobra, feats, n_rep=20):
        tr, te = DOBRAS[dobra]
        y = Y["hesitacao_livre"]
        p0, m = ajusta_prediz(feats, dobra)
        r0 = np.sqrt(np.mean((y.iloc[te].values - p0) ** 2))
        rng = np.random.default_rng(1000 + dobra)
        Xte = X_all.iloc[te][feats].copy()
        out = {}
        for f in feats:
            vals = []
            for _ in range(n_rep):
                Xp = Xte.copy()
                Xp[f] = rng.permutation(Xp[f].values)
                vals.append(np.sqrt(np.mean((y.iloc[te].values - m.predict(Xp)) ** 2)) - r0)
            out[f] = np.mean(vals)
        return dobra, out

    linhas = []
    tabs_pfi = {}
    for rotulo, feats in [("7 variáveis", FEATS), ("8 variáveis (+ tempo total)", FEATS + ["fix_total_med"])]:
        res = Parallel(n_jobs=N_JOBS)(delayed(pfi_dobra)(k, feats) for k in range(K))
        tab = pd.DataFrame({k: o for k, o in res}).T.sort_index()
        tab.to_csv(OUT / f"C_pfi_por_dobra_{len(feats)}vars.csv")
        tabs_pfi[len(feats)] = tab
        for f in feats:
            linhas.append({"modelo": rotulo, "variavel": NOMES[f], **nb(1000 * tab[f].values)})
    pfi = pd.DataFrame(linhas)
    pfi.to_csv(OUT / "C_pfi_mesmas_dobras.csv", index=False)
    d = 1000 * (tabs_pfi[8]["fix_flight_med"] - tabs_pfi[7]["fix_flight_med"]).values
    queda = nb(d)
    resumo["pfi_voo_queda_com_total_ms"] = queda
    print("\nC. Variação pareada da PFI do voo ao incluir o tempo total (ms):", {k: round(v, 3) for k, v in queda.items()})
    print(f"\nC. PFI nas mesmas 50 dobras ({time.time()-t0:.0f}s):\n",
          pfi[["modelo", "variavel", "media", "ic_inf", "ic_sup", "p"]].round(3).to_string())

# ================================================================ E. ruido da floresta x das dobras
if "E" in partes:
    t0 = time.time()
    SEMENTES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    DOBRAS_E = list(range(10))                  # 2 repeticoes completas de 5 dobras
    alvo_vars = ["fix_flight_med", "fix_total_cv"]

    def uma(dobra, s):
        tr, te = DOBRAS[dobra]
        y = Y["hesitacao_livre"]
        out = {}
        p, _ = ajusta_prediz(FEATS, dobra, semente=s)
        r0 = np.sqrt(np.mean((y.iloc[te].values - p) ** 2))
        for v in alvo_vars:
            fs = [f for f in FEATS if f != v]
            p, _ = ajusta_prediz(fs, dobra, semente=s)
            out[v] = 1000 * (np.sqrt(np.mean((y.iloc[te].values - p) ** 2)) - r0)
        return dobra, s, out

    res = Parallel(n_jobs=N_JOBS)(delayed(uma)(k, s) for k in DOBRAS_E for s in SEMENTES)
    linhas = []
    for v in alvo_vars:
        M = np.array([[next(o[v] for k2, s2, o in res if k2 == k and s2 == s) for s in SEMENTES] for k in DOBRAS_E])
        var_sem = M.var(axis=1, ddof=1).mean()          # ruido da floresta (dentro da dobra)
        var_dob = M.mean(axis=1).var(ddof=1)            # ruido das dobras (entre dobras)
        linhas.append({"variavel": NOMES[v], "media": M.mean(), "dp_total": M.std(ddof=1),
                       "dp_semente": np.sqrt(var_sem), "dp_dobra": np.sqrt(var_dob),
                       "frac_var_semente": var_sem / (var_sem + var_dob)})
        pd.DataFrame(M, index=[f"dobra {k}" for k in DOBRAS_E],
                     columns=[f"semente {s}" for s in SEMENTES]).to_csv(OUT / f"E_matriz_{v}.csv")
    ruido = pd.DataFrame(linhas)
    ruido.to_csv(OUT / "E_ruido.csv", index=False)
    print(f"\nE. Ruído da floresta x das dobras ({time.time()-t0:.0f}s):\n", ruido.round(3).to_string())

# ================================================================ F. hiperparametros reajustados
GRADE = {"min_samples_leaf": [3, 5, 10, 15, 20], "max_features": [0.4, 0.7, 1.0]}
if "F" in partes:
    t0 = time.time()
    X_tr, _, y_tr, _ = train_test_split(X_all, Y["hesitacao_livre"], test_size=0.30, random_state=SEMENTE)

    def busca(feats):
        g = GridSearchCV(RandomForestRegressor(n_estimators=300, random_state=SEMENTE, n_jobs=1), GRADE,
                         cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEMENTE),
                         scoring="neg_root_mean_squared_error", n_jobs=N_JOBS).fit(X_tr[feats], y_tr)
        return g.best_params_

    hp_j = {}
    for v in FEATS:
        fs = [f for f in FEATS if f != v]
        hp = busca(fs)
        # max_features como fracao: com menos variaveis, 0,4 pode virar 2 em vez de 3
        hp_j[f"sem {v}"] = hp
        print(f"   HP sem {NOMES[v]}: {hp}")
    configs = {"todas": FEATS, **{f"sem {v}": [f for f in FEATS if f != v] for v in FEATS}}
    tabs = avalia(configs, hp_por_config=hp_j)
    ret = compara(tabs, "todas", [f"sem {v}" for v in FEATS])
    ret["variavel"] = [NOMES[v] for v in FEATS]
    ret["hp"] = [json.dumps(hp_j[f"sem {v}"]) for v in FEATS]
    ret.to_csv(OUT / "F_lofo_hp_reajustado.csv", index=False)
    resumo["hp_reajustados"] = hp_j
    print(f"\nF. LOFO com HP reajustado ({time.time()-t0:.0f}s):\n",
          ret[["variavel", "media", "ic_inf", "ic_sup", "p"]].round(3).to_string())

# ================================================================ G. outros alvos
if "G" in partes:
    t0 = time.time()
    linhas = []
    for alvo in ["taxa_pausa_longa", "taxa_revisao"]:
        X_tr, _, y_tr, _ = train_test_split(X_all, Y[alvo], test_size=0.30, random_state=SEMENTE)
        g = GridSearchCV(RandomForestRegressor(n_estimators=300, random_state=SEMENTE, n_jobs=1), GRADE,
                         cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEMENTE),
                         scoring="neg_root_mean_squared_error", n_jobs=N_JOBS).fit(X_tr[FEATS], y_tr)
        hp = g.best_params_
        configs = {"todas": FEATS, "mediana": [], **{f"sem {v}": [f for f in FEATS if f != v] for v in FEATS}}
        tabs = avalia(configs, alvo=alvo, hp_por_config={n: hp for n in configs})
        c = compara(tabs, "todas", [f"sem {v}" for v in FEATS], escala=100.0)   # pontos percentuais
        c["variavel"] = [NOMES[v] for v in FEATS]
        c["alvo"] = alvo
        c["rmse_base_pp"] = 100 * tabs["todas"]["rmse"].mean()
        c["rmse_mediana_pp"] = 100 * tabs["mediana"]["rmse"].mean()
        c["corr_voo"] = dados[["fix_flight_med", alvo]].corr().iloc[0, 1]
        c["hp"] = json.dumps(hp)
        linhas.append(c)
        print(f"\nG. Alvo {alvo}: RMSE floresta {100*tabs['todas']['rmse'].mean():.2f} pp "
              f"x mediana {100*tabs['mediana']['rmse'].mean():.2f} pp; HP {hp}")
        print(c[["variavel", "media", "ic_inf", "ic_sup", "p"]].round(3).to_string())
    pd.concat(linhas).to_csv(OUT / "G_outros_alvos.csv", index=False)
    print(f"   ({time.time()-t0:.0f}s)")

json.dump(resumo, open(OUT / f"resumo_{''.join(sorted(partes))}.json", "w"), indent=2, default=float)
print(f"\nTempo total: {time.time()-t_ini:.0f}s")
