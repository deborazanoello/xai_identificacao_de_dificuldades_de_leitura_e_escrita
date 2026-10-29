"""Figuras que comparam a escrita, na KeyRecs, com a leitura, na CompLex.

Le as saidas da escrita e resultados/saidas_complex/, gerado por lofo_complex.py. Mesmo estilo de figuras_lofo.py.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import figuras_lofo as base          # reaproveita estilo, cores, salva() e hbar_ic()

C, br, nota, salva, hbar_ic, FATOR_IC = base.C, base.br, base.nota, base.salva, base.hbar_ic, base.FATOR_IC
S, SC, SX = base.S, base.SC, Path("resultados/saidas_complex")      # mesmas pastas da escrita que figuras_lofo.py
KR = {"fix_flight_med": "tempo de voo", "fix_total_cv": "CV do tempo total", "fix_dwell_med": "pressionamento",
      "fix_ganho_pratica": "ganho com a prática", "dem_idade": "idade", "dem_genero": "gênero", "dem_lateralidade": "lateralidade"}
rx = json.load(open(SX / "resumo.json"))
re_ = json.load(open(S / "resumo.json"))

# ════════ 12. LOFO lado a lado: escrita x leitura
kr = pd.read_csv(S / "lofo_rf.csv").sort_values("lofo_ms")
cx = pd.read_csv(SX / "A_lofo_individual.csv").query("modelo == 'rf' and metrica == 'auc'").sort_values("media")
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
m = kr["lofo_ms"].values; h = FATOR_IC * kr["dp_ms"].values
hbar_ic(axes[0], [KR[v] for v in kr["variavel"]], m, m - h, m + h, kr["p_nb"].values)
axes[0].set_title("Escrita · KeyRecs (96 pessoas)"); axes[0].set_xlabel("Δ RMSE ao remover (ms)")
hbar_ic(axes[1], cx["variavel"].tolist(), cx["media"].values, cx["ic_inf"].values, cx["ic_sup"].values, cx["p"].values,
        unid=" pt", cor_pos=C["azul"])
axes[1].set_title("Leitura · CompLex (8.579 termos)"); axes[1].set_xlabel("Δ AUC ao remover (pontos de AUC × 100)")
plt.tight_layout()
nota(fig, f"Mesmo protocolo: Random Forest, 5×10 dobras (na leitura, agrupadas por palavra), IC 95% corrigido · "
          f"base: RMSE {br(re_['rf_base_rmse_ms'])} ms (escrita) · AUC {br(rx['auc_rf'], 3)} (leitura) · * p < 0,05")
salva("12_comparacao_lofo.png")

# ════════ 13. Substitutas: experimento (escrita) x substituta natural (leitura)
red = pd.read_csv(S / "lofo_redundancia.csv").set_index("variavel")
lofo7 = pd.read_csv(S / "lofo_rf.csv").set_index("variavel")
cxi = cx.set_index("variavel")
blo = pd.read_csv(SX / "B_lofo_blocos.csv").set_index("variavel")
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
w = 0.6
itens_e = [("flight\nsozinho", lofo7.loc["fix_flight_med", "lofo_ms"], FATOR_IC * lofo7.loc["fix_flight_med", "dp_ms"], C["cinza"]),
           ("flight\n+ tempo total", red.loc["fix_flight_med", "lofo_ms"], FATOR_IC * red.loc["fix_flight_med", "dp_ms"], C["lar"]),
           ("tempo total\n+ flight", red.loc["fix_total_med", "lofo_ms"], FATOR_IC * red.loc["fix_total_med", "dp_ms"], C["lar"]),
           ("grupo\n{flight, total}", red.loc["{flight + total_med}", "lofo_ms"], FATOR_IC * red.loc["{flight + total_med}", "dp_ms"], C["amar"])]
g = blo.loc["{comprimento, sílabas}"]
itens_l = [("comprimento", cxi.loc["comprimento", "media"], cxi.loc["comprimento", "ic_sup"] - cxi.loc["comprimento", "media"], C["azul"]),
           ("sílabas", cxi.loc["sílabas", "media"], cxi.loc["sílabas", "ic_sup"] - cxi.loc["sílabas", "media"], C["azul"]),
           ("grupo\n{comprimento, sílabas}", g["media"], g["ic_sup"] - g["media"], C["amar"])]
for ax, itens, tit, yl in [(axes[0], itens_e, "Escrita: substituta colocada de propósito", "Δ RMSE ao remover (ms)"),
                           (axes[1], itens_l, "Leitura: substituta natural", "Δ AUC ao remover (pontos)")]:
    x = np.arange(len(itens))
    for i, (lab, v, e, cor) in enumerate(itens):
        ax.bar(i, v, w, color=cor, hatch="//" if cor == C["amar"] else None, edgecolor=C["sup"])
        ax.errorbar(i, v, yerr=e, fmt="none", ecolor=C["tinta"], capsize=4, lw=1.3)
        ax.text(i, v + e + 0.03 * max(it[1] + it[2] for it in itens), br(v, 2 if abs(v) < 1 else 1), ha="center", fontsize=14, color=C["tinta2"])
    ax.set_xticks(x, [it[0] for it in itens], fontsize=12.5); ax.grid(axis="x", visible=False)
    ax.axhline(0, color=C["tinta"], lw=1.2); ax.set_ylabel(yl); ax.set_title(tit)
plt.tight_layout()
r_cs = 0.866
nota(fig, f"Escrita: tempo total com r = {br(re_['rho_flight_total'], 2)} com o tempo de voo · leitura: comprimento e sílabas com r = {br(r_cs, 2)} · IC 95% corrigido")
salva("13_comparacao_substitutas.png")

# ════════ 14. Suficiencia: escrita x leitura
sk = pd.read_csv(SC / "B_suficiencia.csv").set_index("config")
sx = pd.read_csv(SX / "C_suficiencia.csv").set_index("config")
fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
ax = axes[0]
lab = ["nenhuma\n(mediana)", "só o\nflight", "as 7\nvariáveis"]
val = [sk.loc["mediana", "rmse_ms"], sk.loc["só fix_flight_med", "rmse_ms"], sk.loc["todas", "rmse_ms"]]
ax.bar(range(3), val, 0.6, color=[C["cinza"], C["lar"], C["azul"]])
for i, v in enumerate(val): ax.text(i, v + 0.8, br(v) + " ms", ha="center", fontsize=14, color=C["tinta2"])
ax.set_xticks(range(3), lab); ax.grid(axis="x", visible=False); ax.set_ylim(0, max(val) * 1.2)
ax.set_ylabel("RMSE (ms) · menor é melhor"); ax.set_title("Escrita: só o flight já empata com as 7")
ax = axes[1]
lab = ["regra das palavras\nraras (frequência)", "frequência +\ncomprimento", "as 8\nvariáveis"]
val = [rx["auc_regra"], sx.loc["só {frequência, comprimento}", "auc"], sx.loc["todas", "auc"]]
ax.bar(range(3), val, 0.6, color=[C["cinza"], C["azul"], C["lar"]])
for i, v in enumerate(val): ax.text(i, v + 0.004, br(v, 3), ha="center", fontsize=14, color=C["tinta2"])
ax.set_xticks(range(3), lab); ax.grid(axis="x", visible=False); ax.set_ylim(0.75, 0.95)
ax.set_ylabel("AUC · maior é melhor"); ax.set_title("Leitura: frequência e comprimento chegam perto das 8")
plt.tight_layout()
nota(fig, "Mesmas dobras e hiperparâmetros do LOFO de cada base · na leitura, o eixo começa em 0,75")
salva("14_comparacao_suficiencia.png")
print("ok")
