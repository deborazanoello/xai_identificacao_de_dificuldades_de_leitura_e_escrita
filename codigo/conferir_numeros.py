"""Confere os números do artigo contra as saídas dos scripts.

Para cada número do artigo, lê o valor na saída que o gerou, formata como no texto (vírgula decimal) e
verifica se ele aparece em artigo/main.tex. Não refaz nenhum modelo: rode depois dos outros scripts.

Uso: python conferir_numeros.py [caminho/para/main.tex]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

TEX = Path(sys.argv[1] if len(sys.argv) > 1 else "../artigo/main.tex")
tex = TEX.read_text(encoding="utf-8")


def f(v, casas):
    return f"{v:.{casas}f}".replace(".", ",")


itens = []                                                           # (onde, descrição, texto esperado)
def item(onde, desc, txt): itens.append((onde, desc, txt))


# ---------------------------------------------------------------- dados
t = pd.read_csv("tabelas/tabela_analitica_corrigida.csv", index_col=0)
item("Tabela I", "participantes", f"{len(t)} participantes")
item("Tabela I", "r voo x hesitação", f(t[["fix_flight_med", "hesitacao_livre"]].corr().iloc[0, 1], 2))
item("Seção II", "r voo x tempo total", f(t[["fix_flight_med", "fix_total_med"]].corr().iloc[0, 1], 2))
c = pd.read_csv("tabelas/tabela_complex_reconstruida.csv", keep_default_na=False, dtype={"termo": str, "grupo": str})
item("Tabela I", "termos", f"{len(c):,}".replace(",", "."))
item("Tabela I", "% difíceis", f(100 * c["dificil"].mean(), 1) + "\\%")
item("Tabela I", "r frequência x complexidade", f(c[["frequencia", "complexidade"]].corr().iloc[0, 1], 2).replace("-", "-"))
item("Seção V-C", "r comprimento x sílabas", f(c[["comprimento", "silabas"]].corr().iloc[0, 1], 2))

# ---------------------------------------------------------------- Tabela II
d = pd.read_csv("resultados/saidas_modelos_escrita_corrigida/desempenho.csv", index_col=0)
for m in d.index:
    item("Tabela II", f"{m} RMSE", f(d.loc[m, "RMSE"], 4)); item("Tabela II", f"{m} MAE", f(d.loc[m, "MAE"], 4))
    item("Tabela II", f"{m} R2", f(d.loc[m, "R2"], 2).lstrip("-"))
r = json.load(open("resultados/saidas_modelos_escrita_corrigida/resumo.json"))
item("Tabela II", "p linear x RF", f"p = {f(r['p_linear_vs_rf'], 2)}")

# ---------------------------------------------------------------- Tabela III
d = pd.read_csv("resultados/saidas_leitura/desempenho.csv", index_col=0)
for m in d.index:
    for col in ["auc", "prec10", "pr_auc"] + ([] if m.startswith("Regra") else ["brier"]):
        item("Tabela III", f"{m} {col}", f(d.loc[m, col], 3))
tp = pd.read_csv("resultados/saidas_leitura/testes_pareados.csv")
p = lambda met, a, b: tp.query("metrica == @met and a == @a and b == @b")["p"].iloc[0]
item("Tabela III", "AUC RF x EBM", f"p = {f(p('auc', 'Random Forest', 'EBM'), 2)}")
item("Tabela III", "AUC RF x logística", f"p = {f(p('auc', 'Random Forest', 'Regressão logística'), 2)}")
item("Tabela III", "AUC RF x XGBoost", f"p = {f(p('auc', 'Random Forest', 'XGBoost'), 2)}")

# ---------------------------------------------------------------- LOFO escrita
l = pd.read_csv("resultados/saidas_holm/lofo_escrita_holm.csv").set_index("variavel")
for v, row in l.iterrows():
    item("Tabela IV", f"LOFO {v}", f(abs(row["lofo_ms"]), 2)); item("Tabela IV", f"dp {v}", f(row["dp_ms"], 2))
    item("Tabela IV", f"dobras > 0 {v}", f"{100 * row['frac_pos']:.0f}\\%")
item("Tabela IV", "p Holm voo", f(l.loc["fix_flight_med", "p_holm"], 3))
red = pd.read_csv("resultados/saidas_corrigida/lofo_redundancia.csv").set_index("variavel")
item("Fig. 2b", "LOFO voo com tempo total", f(red.loc["fix_flight_med", "lofo_ms"], 1))
item("Fig. 2b", "LOFO grupo", f(red.loc["{flight + total_med}", "lofo_ms"], 1))
item("Fig. 2b", "p voo com tempo total", f"p = {f(red.loc['fix_flight_med', 'p_nb'], 2)}")
pf = pd.read_csv("resultados/saidas_comp_corrigida/C_pfi_mesmas_dobras.csv")
item("Fig. 2c", "PFI voo 7 vars", f(pf.query("modelo == '7 variáveis' and variavel == 'tempo de voo'")["media"].iloc[0], 1))
item("Fig. 2c", "PFI voo 8 vars", f(pf.query("modelo != '7 variáveis' and variavel == 'tempo de voo'")["media"].iloc[0], 1))
item("Fig. 2c", "PFI tempo total", f(pf.query("variavel == 'tempo total'")["media"].iloc[0], 1))
rl = json.load(open("resultados/saidas_corrigida/resumo.json"))
item("Seção V-C", "RMSE floresta LOFO", f(rl["rf_base_rmse_ms"], 1)); item("Seção V-C", "RMSE mediana", f(rl["baseline_rmse_ms"], 1))
item("Seção V-C", "RMSE linear LOFO", f(rl["lin_base_rmse_ms"], 1))
li = pd.read_csv("resultados/saidas_corrigida/lofo_linear.csv").set_index("variavel")
item("Seção V-C", "LOFO voo linear", f(li.loc["fix_flight_med", "lofo_ms"], 1))
s = pd.read_csv("resultados/saidas_comp_corrigida/B_suficiencia.csv").set_index("config")
item("Tabela V", "só voo", f(s.loc["só fix_flight_med", "rmse_ms"], 1))
item("Tabela V", "p só voo", f"p = {f(s.loc['só fix_flight_med', 'dif_p'], 2)}")

# ---------------------------------------------------------------- LOFO leitura
ll = pd.read_csv("resultados/saidas_holm/lofo_leitura_holm.csv").set_index("variavel")
item("Fig. 3a", "LOFO frequência", f(ll.loc["frequência", "media"], 1))
item("Fig. 3a", "LOFO sigla", f(ll.loc["sigla", "media"], 2)); item("Fig. 3a", "LOFO comprimento", f(ll.loc["comprimento", "media"], 2))
item("Fig. 3a", "p Holm sigla", f"p = {f(ll.loc['sigla', 'p_holm'], 2)}")
bl = pd.read_csv("resultados/saidas_complex/B_lofo_blocos.csv").set_index("variavel")
item("Fig. 3b", "grupo comprimento e sílabas", f(bl.loc["{comprimento, sílabas}", "media"], 2))
sx = pd.read_csv("resultados/saidas_complex/C_suficiencia.csv").set_index("config")
item("Fig. 3c", "só frequência", f(sx.loc["só frequencia", "auc"], 3))
item("Fig. 3c", "frequência + comprimento", f(sx.loc["só {frequência, comprimento}", "auc"], 3))

# ---------------------------------------------------------------- métodos globais
g = json.load(open("resultados/saidas_globais_corrigida/resumo.json"))
for base in ["escrita", "leitura"]:
    h = g[base]["H_decomp"]
    item("Seção V-B", f"{base}: R2 efeitos principais", f(h["r2_efeitos_principais"], 2))
    item("Fig. 4", f"{base}: árvore substituta", f(g[base]["substitutos"]["arvore_r2"], 2))
    item("Fig. 4", f"{base}: linear substituta", f(g[base]["substitutos"]["linear_r2"], 2).lstrip("-"))
    item("Fig. 4", f"{base}: regras", f(g[base]["regras"]["fidelidade_r2_teste"], 2))
item("Seção V-B", "H frequência", f(g["leitura"]["H_decomp"]["H_variavel"]["frequência"], 2))
item("Seção V-B", "H tempo de voo", f(g["escrita"]["H_decomp"]["H_variavel"]["tempo de voo"], 2))
item("Seção V-B", "1ª divisão leitura", f(g["leitura"]["substitutos"]["primeira_divisao"]["limiar"], 2))
item("Seção V-B", "1ª divisão escrita", f(g["escrita"]["substitutos"]["primeira_divisao"]["limiar"], 3))
item("Seção V-C", "rho escrita", f(g["escrita"]["rho_lofo_pfi_mesmas_dobras"], 2))
item("Seção V-C", "rho leitura", f(g["leitura"]["rho_lofo_pfi_mesmas_dobras"], 2))
m = g["escrita"]["mmd"]
item("Seção V-E", "erro críticas", f(m["erro_medio_criticas"], 3)); item("Seção V-E", "erro protótipos", f(m["erro_medio_prototipos"], 3))
m = g["leitura"]["mmd"]
item("Seção V-E", "prob críticas", f(m["prob_media_criticas"], 2)); item("Seção V-E", "prob protótipos", f(m["prob_media_prototipos"], 2))
for x in g["leitura"]["mmd"]["criticas"] + g["leitura"]["mmd"]["prototipos"]:
    item("Seção V-E", "termo do MMD-critic", x["termo"])

falta = [(o, dsc, txt) for o, dsc, txt in itens if txt not in tex]
print(f"{len(itens) - len(falta)} de {len(itens)} números encontrados no artigo.")
for o, dsc, txt in falta:
    print(f"  NÃO ENCONTRADO  {o:<10} {dsc:<40} {txt}")
