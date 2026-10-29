"""Gera figuras/lofo_leitura.pdf (LOFO na leitura, CompLex).

Valores de lofo_complex.py e modelos_leitura.py: saidas_complex/A_lofo_individual.csv (rf, auc),
saidas_complex/B_lofo_blocos.csv, saidas_complex/C_suficiencia.csv e saidas_leitura/desempenho.csv.
Random Forest, 5 dobras agrupadas por palavra x 10 repeticoes; queda da AUC em pontos (x 100).
Rode a partir da pasta do projeto:  python codigo_figuras/fig_lofo_leitura.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# ------------------------------------------------ (a) LOFO por variavel
variaveis = ["frequência", "sigla", "comprimento", "nome próprio", "sílabas", "tamanho da frase",
             "forma em português", "posição"]
lofo = np.array([16.68, 0.86, 0.80, 0.06, 0.05, 0.01, -0.02, -0.03])
ic = np.array([1.69, 0.58, 0.53, 0.16, 0.22, 0.21, 0.16, 0.20])     # meia-largura do IC 95% corrigido
signif = [True, True, True, False, False, False, False, False]

# ------------------------------------------------ (b) substituta natural
subst = [("comprimento", 0.80, 0.53, False), ("sílabas", 0.05, 0.22, False),
         ("grupo\n{comprimento,\nsílabas}", 1.19, 0.76, True)]

# ------------------------------------------------ (c) suficiencia (AUC)
sufic = [("só a\nfrequência", 0.808), ("regra das\nraras", 0.849),
         ("frequência\n+ compr.", 0.901), ("as 8\nvariáveis", 0.919)]

fonte = "Liberation Serif"
if not any(f.name == fonte for f in font_manager.fontManager.ttflist):
    fonte = "serif"
plt.rcParams.update({
    "font.family": fonte, "font.size": 7.5, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "hatch.linewidth": 0.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#a3a8b2"
TINTA, GRADE = "#1a1a1a", "#e3e3e3"


def v(x, casas=2):
    return f"{x:.{casas}f}".replace(".", ",").replace("-", "−")


fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.35, 0.9, 1.15]})

ax = axes[0]
y = np.arange(len(variaveis))[::-1]
ax.barh(y, lofo, xerr=ic, height=0.6, color=[LARANJA if s else CINZA for s in signif],
        error_kw={"ecolor": TINTA, "capsize": 2, "lw": 0.7})
ax.axvline(0, color=TINTA, lw=0.7)
for yi, val, e, s in zip(y, lofo, ic, signif):
    ax.text(max(val, 0) + e + 0.3, yi, ("+" if val > 0 else "") + v(val) + (" *" if s else ""),
            va="center", fontsize=6.8, color=TINTA)
ax.set_yticks(y, variaveis, fontsize=7)
ax.tick_params(axis="y", length=0)
ax.set_xlim(-1, 22)
ax.set_xlabel("Δ AUC × 100 ao remover", fontsize=7.2)
ax.set_title("a. LOFO por variável", fontsize=7.8, loc="left", fontweight="bold")
ax.xaxis.grid(True, color=GRADE, lw=0.5); ax.set_axisbelow(True)

ax = axes[1]
for i, (rot, val, e, grupo) in enumerate(subst):
    ax.bar(i, val, 0.6, color=LARANJA, edgecolor="white", lw=0.4, hatch="////" if grupo else None)
    ax.errorbar(i, val, yerr=e, fmt="none", ecolor=TINTA, capsize=2, lw=0.7)
    ax.text(i + 0.08, val + e + 0.05, v(val), ha="left", fontsize=6.8, color=TINTA)
ax.set_xticks(range(len(subst)), [s[0] for s in subst], fontsize=6.6)
ax.tick_params(axis="x", length=0)
ax.axhline(0, color=TINTA, lw=0.6)
ax.set_ylim(-0.3, 2.3)
ax.set_ylabel("Δ AUC × 100 ao remover", fontsize=7.2)
ax.set_title("b. Substituta natural", fontsize=7.8, loc="left", fontweight="bold")
ax.yaxis.grid(True, color=GRADE, lw=0.5); ax.set_axisbelow(True)

ax = axes[2]
cores = [CINZA, CINZA, LARANJA, LARANJA]
for i, ((rot, val), cor) in enumerate(zip(sufic, cores)):
    ax.bar(i, val, 0.62, color=cor, edgecolor="white", lw=0.4)
    ax.text(i, val + 0.003, v(val, 3), ha="center", fontsize=6.8, color=TINTA)
ax.set_xticks(range(len(sufic)), [s[0] for s in sufic], fontsize=6.3)
ax.tick_params(axis="x", length=0)
ax.set_ylim(0.75, 0.94)
ax.set_ylabel("AUC, eixo a partir de 0,75", fontsize=7.2)
ax.set_yticks([0.75, 0.80, 0.85, 0.90], ["0,75", "0,80", "0,85", "0,90"])
ax.set_title("c. Suficiência", fontsize=7.8, loc="left", fontweight="bold")
ax.yaxis.grid(True, color=GRADE, lw=0.5); ax.set_axisbelow(True)

fig.tight_layout(pad=0.3, w_pad=1.2)
fig.savefig("figuras/lofo_leitura.pdf")
print("ok")
