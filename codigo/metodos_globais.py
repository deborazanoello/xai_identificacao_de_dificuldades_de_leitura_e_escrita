"""Métodos globais de XAI aplicados ao Random Forest das duas tarefas (Seções V-B a V-E do artigo).

Escrita (KeyRecs): tabelas/tabela_analitica.csv, 96 participantes, a mesma floresta da ATV03
  (500 árvores, max_features = 1,0, min_samples_leaf = 10), treinada nos 70% de treino (semente 42).
Leitura (CompLex): tabelas/tabela_complex_reconstruida.csv, 8.579 termos, floresta com os hiperparâmetros do LOFO
  (resultados/saidas_complex/resumo.json), 500 árvores, treinada em 70% das palavras (GroupShuffleSplit, semente 42).
Na leitura, a "previsão" é a probabilidade de o termo ser difícil.

Métodos:
  1. H de Friedman (por variável e por par), com dependências parciais centradas
  2. Decomposição funcional: R² entre a previsão e constante + efeitos principais (no teste)
  5. Substitutos globais: árvore de profundidade 3 e regressão linear treinadas nas previsões do RF
  6. Lista de regras extraída das árvores do RF (caminhos de até 2 condições, seleção gulosa)
  7. MMD-critic: protótipos e críticas (na leitura, entre os termos difíceis)
  (3 e 4, permutação e LOFO, estão em lofo_complementar.py e lofo_complex.py; aqui só a correlação entre eles.)

Uso: python metodos_globais.py   (~10 min em 2 núcleos; gera resultados/saidas_globais/)
"""
import json, time, warnings
import os
TABELA = os.environ.get("TABELA_ESCRITA", "tabelas/tabela_analitica.csv")   # tabela da escrita (ou a corrigida)
SUF = os.environ.get("SUFIXO", "")                                  # sufixo das pastas de saída
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
OUT = Path("resultados/saidas_globais" + SUF); OUT.mkdir(parents=True, exist_ok=True)
SEMENTE, N_JOBS = 42, 2
t0 = time.time()


# ════════════════════════════════════════════════════════════════ dados e modelos
def dados_escrita():
    d = pd.read_csv(TABELA, index_col=0)
    exp = ["fix_dwell_med", "fix_flight_med", "fix_total_cv", "fix_ganho_pratica", "dem_idade", "dem_genero",
           "dem_lateralidade"]
    X = d[exp].copy()
    for c in ["dem_genero", "dem_lateralidade"]:                       # dummies como na ATV03 (referência = mais frequente)
        X[c] = pd.Categorical(X[c], categories=list(X[c].value_counts().index))
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    return X, d["hesitacao_livre"].values


def dados_leitura():
    t = pd.read_csv("tabelas/tabela_complex_reconstruida.csv", keep_default_na=False, dtype={"termo": str, "grupo": str})
    F = ["frequencia", "comprimento", "silabas", "sigla", "nome_proprio", "tam_frase", "posicao", "forma_pt"]
    return t, t[F].astype(float), t["dificil"].values, t["grupo"].values


NOMES = {"fix_flight_med": "tempo de voo", "fix_dwell_med": "pressionamento", "fix_total_cv": "CV do tempo total",
         "fix_ganho_pratica": "ganho com a prática", "dem_idade": "idade", "frequencia": "frequência",
         "comprimento": "comprimento", "silabas": "sílabas", "sigla": "sigla", "nome_proprio": "nome próprio",
         "tam_frase": "tamanho da frase", "posicao": "posição", "forma_pt": "forma em português"}


def nome(c):
    if c.startswith("dem_genero"): return "gênero"
    if c.startswith("dem_lateralidade"): return "lateralidade"
    return NOMES.get(c, c)


Xe, ye = dados_escrita()
Xe_tr, Xe_te, ye_tr, ye_te = train_test_split(Xe, ye, test_size=0.30, random_state=SEMENTE)
rf_e = RandomForestRegressor(n_estimators=500, max_features=1.0, min_samples_leaf=10, random_state=SEMENTE,
                             n_jobs=N_JOBS).fit(Xe_tr, ye_tr)
f_e = rf_e.predict

tab_l, Xl, yl, gl = dados_leitura()
tr70, te30 = next(GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=SEMENTE).split(Xl, yl, gl))
HP_L = json.load(open("resultados/saidas_complex/resumo.json"))["HP"]
rf_l = RandomForestClassifier(n_estimators=500, random_state=SEMENTE, n_jobs=N_JOBS, **HP_L).fit(Xl.iloc[tr70], yl[tr70])
f_l = lambda X: rf_l.predict_proba(X)[:, 1]
print(f"modelos ajustados ({time.time()-t0:.0f} s)")


# ════════════════════════════════════════════════════════════════ 1-2. dependência parcial, H e decomposição
def pd_grade(f, Xpontos, Xfundo, cols):
    """PD_S avaliada em cada linha de Xpontos: média, sobre Xfundo, de f com as colunas S trocadas pelas do ponto."""
    n, m = len(Xpontos), len(Xfundo)
    grande = pd.concat([Xfundo] * n, ignore_index=True)
    for c in cols:
        grande[c] = np.repeat(Xpontos[c].values, m)
    return f(grande).reshape(n, m).mean(axis=1)


def friedman_e_decomposicao(f, Xamostra, Xteste, tag):
    cols = list(Xamostra.columns)
    fx = f(Xamostra); fx_c = fx - fx.mean()
    pdj = {c: pd_grade(f, Xamostra, Xamostra, [c]) for c in cols}
    pdj = {c: v - v.mean() for c, v in pdj.items()}
    # H_j: interação de j com todas as outras
    hj = {}
    for c in cols:
        resto = [o for o in cols if o != c]
        pdr = pd_grade(f, Xamostra, Xamostra, resto); pdr = pdr - pdr.mean()
        hj[nome(c)] = float(np.sqrt(max(np.sum((fx_c - pdj[c] - pdr) ** 2), 0) / np.sum(fx_c ** 2)))
    # H_jk por par
    pares = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pab = pd_grade(f, Xamostra, Xamostra, [a, b]); pab = pab - pab.mean()
            den = np.sum(pab ** 2)
            h = float(np.sqrt(max(np.sum((pab - pdj[a] - pdj[b]) ** 2), 0) / den)) if den > 1e-12 else 0.0
            pares.append({"par": f"{nome(a)} × {nome(b)}", "H": h, "var_pd_par": float(den / len(pab))})
    pares = pd.DataFrame(pares).sort_values("H", ascending=False)
    pares.to_csv(OUT / f"{tag}_H_pares.csv", index=False)
    # decomposição: efeitos principais estimados na amostra de treino, avaliados no teste
    ft = f(Xteste)
    comp = {c: pd_grade(f, Xteste, Xamostra, [c]) for c in cols}
    medias = {c: pd_grade(f, Xamostra, Xamostra, [c]).mean() for c in cols}   # centra pela média no treino
    aditivo = fx.mean() + sum(comp[c] - medias[c] for c in cols)
    r2_aditivo = r2_score(ft, aditivo)
    var_comp = pd.Series({nome(c): float(np.var(comp[c])) for c in cols}).sort_values(ascending=False)
    return {"H_variavel": dict(sorted(hj.items(), key=lambda kv: -kv[1])), "H_pares_top5": pares.head(5).to_dict("records"),
            "r2_efeitos_principais": float(r2_aditivo), "variancia_componentes": var_comp.to_dict(),
            "n_amostra": len(Xamostra), "n_teste": len(Xteste)}, aditivo, ft


# escrita: amostra = 67 de treino; teste = 29. leitura: 300 termos de treino e 300 de teste (sorteio fixo)
rng = np.random.default_rng(SEMENTE)
amostra_l = Xl.iloc[rng.choice(tr70, 300, replace=False)].reset_index(drop=True)
teste_l = Xl.iloc[rng.choice(te30, 300, replace=False)].reset_index(drop=True)
res = {"escrita": {}, "leitura": {}}
res["escrita"]["H_decomp"], adit_e, ft_e = friedman_e_decomposicao(f_e, Xe_tr.reset_index(drop=True),
                                                                   Xe_te.reset_index(drop=True), "escrita")
print(f"H e decomposição da escrita ({time.time()-t0:.0f} s)")
res["leitura"]["H_decomp"], adit_l, ft_l = friedman_e_decomposicao(f_l, amostra_l, teste_l, "leitura")
print(f"H e decomposição da leitura ({time.time()-t0:.0f} s)")


# ════════════════════════════════════════════════════════════════ 5. substitutos globais
def substitutos(f, Xtr, Xte):
    alvo_tr, alvo_te = f(Xtr), f(Xte)
    arv = DecisionTreeRegressor(max_depth=3, random_state=SEMENTE).fit(Xtr, alvo_tr)
    lin = LinearRegression().fit(Xtr, alvo_tr)
    tr_ = arv.tree_
    return {"arvore_r2": float(r2_score(alvo_te, arv.predict(Xte))), "linear_r2": float(r2_score(alvo_te, lin.predict(Xte))),
            "arvore_folhas": int(arv.get_n_leaves()),
            "primeira_divisao": {"variavel": nome(Xtr.columns[tr_.feature[0]]), "limiar": float(tr_.threshold[0])},
            "segundas_divisoes": [{"variavel": nome(Xtr.columns[tr_.feature[k]]), "limiar": float(tr_.threshold[k])}
                                  for k in (tr_.children_left[0], tr_.children_right[0]) if tr_.feature[k] >= 0]}


res["escrita"]["substitutos"] = substitutos(f_e, Xe_tr, Xe_te)
res["leitura"]["substitutos"] = substitutos(f_l, Xl.iloc[tr70], Xl.iloc[te30])
print(f"substitutos ({time.time()-t0:.0f} s)")


# ════════════════════════════════════════════════════════════════ 6. lista de regras extraída do RF
def regras_candidatas(rf, cols, prof_max=2):
    """Todos os caminhos de até prof_max condições em todas as árvores, sem repetição."""
    vistas = set()
    for est in rf.estimators_:
        t = est.tree_
        pilha = [(0, ())]
        while pilha:
            no, conds = pilha.pop()
            if conds and conds not in vistas:
                vistas.add(conds)
            if len(conds) == prof_max or t.feature[no] < 0:
                continue
            j, lim = int(t.feature[no]), float(np.round(t.threshold[no], 4))
            pilha.append((t.children_left[no], tuple(sorted(conds + ((j, "<=", lim),)))))
            pilha.append((t.children_right[no], tuple(sorted(conds + ((j, ">", lim),)))))
    return sorted(vistas)                                                # ordem fixa: resultado reprodutível


def cobre(X, conds):
    m = np.ones(len(X), bool)
    for j, op, lim in conds:
        v = X.iloc[:, j].values
        m &= (v <= lim) if op == "<=" else (v > lim)
    return m


def texto_regra(conds, cols):
    """Junta condições da mesma variável (ex.: voo > 0,11 E voo > 0,17 vira voo > 0,17)."""
    num = lambda v: f"{v:.3g}".replace(".", ",")
    partes = []
    for j in sorted({j for j, _, _ in conds}):
        c = cols[j]
        acima = [l for jj, op, l in conds if jj == j and op == ">"]
        abaixo = [l for jj, op, l in conds if jj == j and op == "<="]
        if c in ("sigla", "nome_proprio", "forma_pt") or c.startswith("dem_"):
            rot = {"sigla": "é sigla", "nome_proprio": "é nome próprio", "forma_pt": "tem forma em português"}.get(c, nome(c))
            partes.append(rot if acima else "não " + rot)
        elif acima and abaixo:
            partes.append(f"{num(max(acima))} < {nome(c)} ≤ {num(min(abaixo))}")
        elif acima:
            partes.append(f"{nome(c)} > {num(max(acima))}")
        else:
            partes.append(f"{nome(c)} ≤ {num(min(abaixo))}")
    return " E ".join(partes)


def lista_de_regras(rf, Xtr, Xte, f, min_frac=0.05, max_regras=8, ganho_min=0.01):
    cols = list(Xtr.columns); alvo_tr, alvo_te = f(Xtr), f(Xte)
    cand = regras_candidatas(rf, cols)
    M = np.array([cobre(Xtr, c) for c in cand])                          # candidatas × treino
    restam = np.ones(len(Xtr), bool); lista = []; sst = np.sum((alvo_tr - alvo_tr.mean()) ** 2)
    min_n = max(5, int(np.ceil(min_frac * len(Xtr)))); r2_ant = 0.0
    while len(lista) < max_regras:
        cob = M & restam
        n1 = cob.sum(1); n0 = restam.sum() - n1
        ok = (n1 >= min_n) & (n0 >= min_n)
        if not ok.any(): break
        s1 = cob @ alvo_tr; s0 = (restam * alvo_tr).sum() - s1
        q1 = cob @ alvo_tr ** 2; q0 = (restam * alvo_tr ** 2).sum() - q1
        # como no inTrees: primeiro a regra mais "pura" (menor variância das previsões do RF entre os casos que
        # ela cobre), desde que cubra ao menos 5% dos casos; empate decidido pela maior cobertura
        mse1 = np.where(ok, (q1 - s1 ** 2 / np.maximum(n1, 1)) / np.maximum(n1, 1), np.inf)
        k = int(np.lexsort((-n1, np.round(mse1 / np.var(alvo_tr), 3)))[0])
        prev_tr = np.empty(len(Xtr)); tmp = np.ones(len(Xtr), bool)
        for c, v in lista: m = cobre(Xtr, c) & tmp; prev_tr[m] = v; tmp &= ~m
        nova = M[k] & restam; v = float(alvo_tr[nova].mean()); prev_tr[nova] = v; tmp2 = tmp & ~nova
        prev_tr[tmp2] = alvo_tr[tmp2].mean()
        r2 = 1 - np.sum((alvo_tr - prev_tr) ** 2) / sst
        if r2 - r2_ant < ganho_min: break
        lista.append((cand[k], v)); restam &= ~nova; r2_ant = r2
    senao = float(alvo_tr[restam].mean())

    def aplica(X):
        p = np.full(len(X), senao); livre = np.ones(len(X), bool); quem = np.full(len(X), len(lista))
        for i, (c, v) in enumerate(lista):
            m = cobre(X, c) & livre; p[m] = v; quem[m] = i; livre &= ~m
        return p, quem

    p_te, _ = aplica(Xte)
    Xall = pd.concat([Xtr, Xte]); _, quem_all = aplica(Xall)
    regras = [{"regra": "SE " + texto_regra(c, cols), "previsao": v, "casos_pct": float(100 * np.mean(quem_all == i))}
              for i, (c, v) in enumerate(lista)]
    regras.append({"regra": "SENÃO", "previsao": senao, "casos_pct": float(100 * np.mean(quem_all == len(lista)))})
    return {"n_regras": len(lista), "fidelidade_r2_teste": float(r2_score(alvo_te, p_te)),
            "fidelidade_r2_treino": float(r2_ant), "n_candidatas": len(cand), "regras": regras}


res["escrita"]["regras"] = lista_de_regras(rf_e, Xe_tr, Xe_te, f_e)
res["leitura"]["regras"] = lista_de_regras(rf_l, Xl.iloc[tr70], Xl.iloc[te30], f_l)
print(f"regras ({time.time()-t0:.0f} s)")


# ════════════════════════════════════════════════════════════════ 7. MMD-critic
def mmd_critic(Z, m, c):
    """Kim et al. (2016): protótipos por MMD gulosa; críticas pela função testemunha com regularizador log-det."""
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    gama = 1 / np.median(d2[np.triu_indices(len(Z), 1)])
    K = np.exp(-gama * d2); n = len(Z); media_col = K.mean(0)
    protos = []
    for _ in range(m):
        melhor, val = None, -np.inf
        for i in range(n):
            if i in protos: continue
            S = protos + [i]
            v = 2 * media_col[S].sum() / len(S) - K[np.ix_(S, S)].sum() / len(S) ** 2
            if v > val: melhor, val = i, v
        protos.append(melhor)
    teste = np.abs(media_col - K[:, protos].mean(1))                      # |função testemunha|
    criticas = []
    for _ in range(c):
        melhor, val = None, -np.inf
        for i in range(n):
            if i in protos or i in criticas: continue
            C = criticas + [i]
            v = teste[C].sum() + np.linalg.slogdet(K[np.ix_(C, C)])[1]
            if v > val: melhor, val = i, v
        criticas.append(melhor)
    return protos, criticas, teste


# escrita: 96 participantes, variáveis padronizadas; erro fora da amostra (KFold 5, mesma floresta)
oof_e = np.empty(len(Xe))
for tr, te in KFold(5, shuffle=True, random_state=SEMENTE).split(Xe):
    m = RandomForestRegressor(n_estimators=500, max_features=1.0, min_samples_leaf=10, random_state=SEMENTE,
                              n_jobs=N_JOBS).fit(Xe.iloc[tr], ye[tr])
    oof_e[te] = m.predict(Xe.iloc[te])
Ze = ((Xe - Xe.mean()) / Xe.std()).values
pe, ce, _ = mmd_critic(Ze, 4, 4)
erro_e = np.abs(ye - oof_e)
d_e = pd.read_csv(TABELA, index_col=0)
def desc_e(ix): return [{"id": Xe.index[i], "hesitacao": float(ye[i]), "previsto": float(oof_e[i]),
                         "lateralidade": d_e.loc[Xe.index[i], "dem_lateralidade"], "genero": d_e.loc[Xe.index[i], "dem_genero"],
                         "idade": int(d_e.loc[Xe.index[i], "dem_idade"]), "voo": float(d_e.loc[Xe.index[i], "fix_flight_med"])}
                        for i in ix]
res["escrita"]["mmd"] = {"prototipos": desc_e(pe), "criticas": desc_e(ce),
                         "erro_medio_prototipos": float(erro_e[pe].mean()), "erro_medio_criticas": float(erro_e[ce].mean()),
                         "erro_medio_todos": float(erro_e.mean()),
                         "canhotos_na_amostra": int((d_e["dem_lateralidade"] != d_e["dem_lateralidade"].mode()[0]).sum())}

# leitura: termos difíceis; probabilidade fora da amostra (5 dobras agrupadas por palavra)
oof_l = np.empty(len(Xl))
for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=0).split(Xl, yl, gl):
    m = RandomForestClassifier(n_estimators=500, random_state=SEMENTE, n_jobs=N_JOBS, **HP_L).fit(Xl.iloc[tr], yl[tr])
    oof_l[te] = m.predict_proba(Xl.iloc[te])[:, 1]
dif = np.where(yl == 1)[0]
Zl = ((Xl.iloc[dif] - Xl.iloc[dif].mean()) / Xl.iloc[dif].std()).values
pl, cl, _ = mmd_critic(Zl, 6, 6)
def desc_l(ix): return [{"termo": tab_l.iloc[dif[i]]["termo"], "corpus": tab_l.iloc[dif[i]]["corpus"],
                         "complexidade": float(yl_c[dif[i]]), "prob": float(oof_l[dif[i]]),
                         **{NOMES[c]: float(Xl.iloc[dif[i]][c]) for c in Xl.columns}} for i in ix]
yl_c = tab_l["complexidade"].values
pior = dif[np.argsort(oof_l[dif])[:5]]
res["leitura"]["mmd"] = {"n_dificeis": int(len(dif)), "prototipos": desc_l(pl), "criticas": desc_l(cl),
                         "prob_media_prototipos": float(oof_l[dif[pl]].mean()), "prob_media_criticas": float(oof_l[dif[cl]].mean()),
                         "prob_media_dificeis": float(oof_l[dif].mean()),
                         "piores_dificeis": [{"termo": tab_l.iloc[i]["termo"], "corpus": tab_l.iloc[i]["corpus"], "id": tab_l.iloc[i]["id"],
                                              "complexidade": float(yl_c[i]), "prob": float(oof_l[i]),
                                              "frequencia": float(Xl.iloc[i]["frequencia"])} for i in pior]}
pd.DataFrame({"oof": oof_l}).to_csv(OUT / "leitura_prob_fora_da_amostra.csv", index=False)
print(f"MMD-critic ({time.time()-t0:.0f} s)")


# ════════════════════════════════════════════════════════════════ 3-4. permutação × LOFO (resultados já calculados)
def rho(lofo, pfi):
    comum = lofo.index.intersection(pfi.index)
    return float(stats.spearmanr(lofo[comum], pfi[comum]).statistic)


KR = {"fix_flight_med": "tempo de voo", "fix_total_cv": "CV do tempo total", "fix_dwell_med": "pressionamento",
      "fix_ganho_pratica": "ganho com a prática", "dem_idade": "idade", "dem_genero": "gênero", "dem_lateralidade": "lateralidade"}
lofo_e = pd.read_csv(f"resultados/saidas{SUF}/lofo_rf.csv").assign(variavel=lambda d: d["variavel"].map(KR)).set_index("variavel")["lofo_ms"]
pfi_e = pd.read_csv(f"resultados/saidas_comp{SUF}/C_pfi_mesmas_dobras.csv").query("modelo == '7 variáveis'").set_index("variavel")
res["escrita"]["rho_lofo_pfi_mesmas_dobras"] = rho(lofo_e, pfi_e["media"])
res["escrita"]["pfi_mesmas_dobras"] = pfi_e[["media", "p"]].to_dict("index")
pfi_l = pd.read_csv("resultados/saidas_complex/D_pfi_mesmas_dobras.csv")
res["leitura"]["pfi_mesmas_dobras"] = pfi_l.to_dict("records")
res["leitura"]["rho_lofo_pfi_mesmas_dobras"] = json.load(open("resultados/saidas_complex/resumo.json")).get("rho_lofo_pfi")

json.dump(res, open(OUT / "resumo.json", "w"), indent=2, ensure_ascii=False, default=str)
print(json.dumps(res, indent=1, ensure_ascii=False, default=str)[:6000])
print(f"ok ({time.time()-t0:.0f} s)")
