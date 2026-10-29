"""Gera figuras/fidelidade.pdf (Fig. 2 do resumo expandido).

Valores de metodos_globais.py (saidas_globais_corrigida/resumo.json): R^2, no conjunto de teste,
entre as previsoes de cada resumo interpretavel e as do Random Forest.
Rode a partir da pasta do projeto:  python codigo_figuras/fig_fidelidade.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- dados
resumos = [
    "Só efeitos principais,\ndecomposição funcional",
    "Árvore substituta\nde profundidade 3",
    "Lista de regras\nextraída do forest",
    "Regressão linear\nsubstituta",
]
escrita = [0.97, 0.98, 0.90, -0.38]  # KeyRecs (tabela corrigida)
leitura = [0.93, 0.91, 0.79, 0.79]  # CompLex

# ---------------------------------------------------------------- estilo
fonte = "Liberation Serif"  # metricas de Times, igual ao texto IEEEtran
if not any(f.name == fonte for f in font_manager.fontManager.ttflist):
    fonte = "serif"
plt.rcParams.update({
    "font.family": fonte,
    "font.size": 8,
    "pdf.fonttype": 42,  # fontes embutidas (exigencia comum do IEEE)
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "hatch.linewidth": 0.5,
})
AZUL, LARANJA = "#2a78d6", "#eb6834"   # paleta validada (CVD ok)
TINTA, TINTA2 = "#1a1a1a", "#555555"


def virgula(v):
    return f"{v:.2f}".replace(".", ",")


fig, ax = plt.subplots(figsize=(3.45, 2.45))
n = len(resumos)
altura = 0.36
ys = list(range(n))[::-1]

for i, y in enumerate(ys):
    for val, cor, desloc, hatch in (
        (escrita[i], AZUL, +altura / 2 + 0.01, None),
        (leitura[i], LARANJA, -altura / 2 - 0.01, "////"),
    ):
        ax.barh(y + desloc, val, height=altura, color=cor, edgecolor="white",
                linewidth=0.6, hatch=hatch)
        ax.text(max(val, 0) + 0.012, y + desloc, virgula(val).replace("-", "−"), va="center", ha="left",
                fontsize=7, color=TINTA)

ax.set_yticks(ys)
ax.set_yticklabels(resumos, fontsize=7.2, color=TINTA)
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.42, 1.1)
ax.set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["−0,25", "0", "0,25", "0,50", "0,75", "1"], color=TINTA2)
ax.axvline(0, color=TINTA, lw=0.6)
ax.tick_params(axis="x", length=2, width=0.6, colors=TINTA2)
ax.set_xlabel("Fidelidade ao Random Forest, R²", color=TINTA, fontsize=7.5)
ax.xaxis.grid(True, color="#e3e3e3", linewidth=0.5)
ax.set_axisbelow(True)

from matplotlib.patches import Patch
leg = [Patch(facecolor=AZUL, edgecolor="white", label="Escrita · KeyRecs"),
       Patch(facecolor=LARANJA, edgecolor="white", hatch="////",
             label="Leitura · CompLex")]
ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2,
          frameon=False, fontsize=7, handlelength=1.4, columnspacing=1.2,
          borderaxespad=0.2)

fig.tight_layout(pad=0.3)
fig.savefig("figuras/fidelidade.pdf")
print("ok")
