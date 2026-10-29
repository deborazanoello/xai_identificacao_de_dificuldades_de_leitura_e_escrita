"""Gera figuras/lofo_keyrecs.pdf (Fig. 3 do resumo expandido).

Valores de lofo_keyrecs.py e lofo_complementar.py (parte C) rodados com a tabela corrigida
(TABELA_ESCRITA=tabela_analitica_corrigida.csv SUFIXO=_corrigida): saidas_corrigida/lofo_rf.csv,
saidas_corrigida/lofo_redundancia.csv e saidas_comp_corrigida/C_pfi_mesmas_dobras.csv.
Random Forest, CV 5 dobras x 10 repeticoes.
Rode a partir da pasta do projeto:  python codigo_figuras/fig_lofo.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# ------------------------------------------------ (a) LOFO por variavel
variaveis = ["tempo de voo", "CV do tempo total", "lateralidade", "gênero",
             "ganho com a prática", "idade", "pressionamento"]
lofo = np.array([10.36, 0.52, 0.00, -0.02, -0.04, -0.04, -0.09])   # ms
dp = np.array([5.38, 1.02, 0.00, 0.07, 0.11, 0.12, 0.17])          # ms, entre 50 dobras
signif = [True, False, False, False, False, False, False]           # p < 0,05 (Nadeau-Bengio)
# IC 95% com a correcao de Nadeau & Bengio (2003): as 50 dobras nao sao independentes.
# Var corrigida = (1/K + n_teste/n_treino) * dp^2, com K = 50 e razao 0,25 (5 dobras).
from scipy import stats
ic95 = stats.t.ppf(0.975, 49) * np.sqrt(1 / 50 + 0.25) * dp

# ------------------------------------------------ (b)/(c) variavel substituta
lofo_7 = {"voo": 10.4}
lofo_8 = {"voo": 3.0, "total": 0.8, "grupo": 11.1}
pfi_7 = {"voo": 17.4}   # PFI nas mesmas 50 dobras (lofo_complementar.py, parte C)
pfi_8 = {"voo": 12.9, "total": 2.2}

# ------------------------------------------------ estilo
fonte = "Liberation Serif"
if not any(f.name == fonte for f in font_manager.fontManager.ttflist):
    fonte = "serif"
plt.rcParams.update({
    "font.family": fonte, "font.size": 7.5, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "hatch.linewidth": 0.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
AZUL, CINZA = "#2a78d6", "#a3a8b2"
TINTA, TINTA2, GRADE = "#1a1a1a", "#555555", "#e3e3e3"


def v(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",").replace("-", "−")


fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25),
                         gridspec_kw={"width_ratios": [1.45, 1.0, 0.85]})

# (a)
ax = axes[0]
y = np.arange(len(variaveis))[::-1]
ax.barh(y, lofo, xerr=ic95, height=0.6, color=[AZUL if s else CINZA for s in signif],
        error_kw={"ecolor": TINTA, "capsize": 2, "lw": 0.7})
ax.axvline(0, color=TINTA, lw=0.7)
for yi, val, e, s in zip(y, lofo, ic95, signif):
    ax.text(max(val, 0) + e + 0.25, yi, ("+" if val > 0 else "") + v(val, 2) + (" *" if s else ""),
            va="center", fontsize=6.8, color=TINTA)
ax.set_yticks(y, variaveis, fontsize=7)
ax.tick_params(axis="y", length=0)
ax.set_xlim(-1, 18)
ax.set_xticks([0, 5, 10, 15], ["0", "5", "10", "15"])
ax.set_xlabel("Δ RMSE ao remover, ms", fontsize=7.2)
ax.set_title("a. LOFO por variável", fontsize=7.8, loc="left", fontweight="bold")
ax.xaxis.grid(True, color=GRADE, lw=0.5); ax.set_axisbelow(True)


def barras_redundancia(ax, cats, s7, s8, titulo, ylab, ymax):
    x = np.arange(len(cats)); w = 0.36
    chaves = [c[0] for c in cats]
    for xi, k in zip(x, chaves):
        if k in s7:
            ax.bar(xi - w / 2, s7[k], w, color=CINZA, edgecolor="white", lw=0.4)
            ax.text(xi - w / 2, s7[k] + ymax * 0.02, v(s7[k]), ha="center", fontsize=6.8, color=TINTA)
        if k in s8:
            pos = xi + w / 2 if k != "grupo" else xi
            ax.bar(pos, s8[k], w, color=AZUL, edgecolor="white", lw=0.4,
                   hatch="////" if k == "grupo" else None)
            ax.text(pos, s8[k] + ymax * 0.02, v(s8[k]), ha="center", fontsize=6.8, color=TINTA)
    ax.set_xticks(x, [c[1] for c in cats], fontsize=6.9)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylab, fontsize=7.2)
    ax.set_title(titulo, fontsize=7.8, loc="left", fontweight="bold")
    ax.yaxis.grid(True, color=GRADE, lw=0.5); ax.set_axisbelow(True)


barras_redundancia(axes[1], [("voo", "tempo\nde voo"), ("total", "tempo\ntotal"), ("grupo", "grupo\n{voo, total}")],
                   lofo_7, lofo_8, "b. LOFO com substituta", "Δ RMSE ao remover, ms", 13)
barras_redundancia(axes[2], [("voo", "tempo\nde voo"), ("total", "tempo\ntotal")],
                   pfi_7, pfi_8, "c. Permutação, mesmas dobras", "Δ RMSE ao permutar, ms", 22)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=CINZA, label="7 variáveis"),
                    Patch(facecolor=AZUL, label="7 variáveis + tempo total da senha")],
           loc="upper right", bbox_to_anchor=(0.995, 1.02), ncol=2, frameon=False, fontsize=6.9,
           handlelength=1.2, columnspacing=1.0)

fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0, 0, 1, 0.93))
fig.savefig("figuras/lofo_keyrecs.pdf")
print("ok")
