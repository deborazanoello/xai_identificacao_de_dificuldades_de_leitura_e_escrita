"""Figuras finais para os slides (fontes grandes), a partir das tabelas salvas."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("figs_slides"); OUT.mkdir(exist_ok=True)
C = {"sup": "#F7F6F1", "tinta": "#1B2233", "tinta2": "#4A5263", "suave": "#7A8090", "grade": "#E2E0D8",
     "azul": "#2A68C2", "lar": "#D9572B", "amar": "#E0A400", "cinza": "#9AA0AC"}
mpl.rcParams.update({
    "figure.facecolor": C["sup"], "axes.facecolor": C["sup"], "savefig.facecolor": C["sup"],
    "axes.edgecolor": "#C9C6BA", "axes.labelcolor": C["tinta2"], "axes.titlecolor": C["tinta"],
    "axes.titlesize": 17, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": C["grade"], "grid.linewidth": 1,
    "xtick.color": C["tinta2"], "ytick.color": C["tinta2"], "legend.frameon": False,
    "font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 15, "legend.fontsize": 14,
    "savefig.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans"})


def hbar(ax, nomes, vals, err=None, cores=None, fmt="{:+.1f}", unid="", lim_txt=None):
    y = np.arange(len(nomes))
    cores = cores or [C["lar"] if v > 0 else C["azul"] for v in vals]
    ax.barh(y, vals, xerr=err, color=cores, height=0.62,
            error_kw={"ecolor": C["tinta"], "capsize": 4, "lw": 1.3})
    ax.set_yticks(y, nomes); ax.axvline(0, color=C["tinta"], lw=1.2); ax.grid(axis="y", visible=False)
    span = max(np.max(np.abs(vals)), 1e-9)
    for i, v in enumerate(vals):
        e = 0 if err is None else err[i]
        txt = "≈ 0" if abs(v) < 0.05 * span else fmt.format(v) + unid
        ax.text(max(v, 0) + e + 0.02 * span, i, txt, va="center", fontsize=14, color=C["tinta2"])
    ax.set_xlim(min(0, np.min(vals)) - 0.05 * span, np.max(vals) + 0.25 * span)


# ═══ KeyRecs ═══
R = {"fix_flight_med": "flight (pausa entre teclas)", "fix_total_cv": "CV do tempo total",
     "fix_dwell_med": "dwell (tecla pressionada)", "fix_ganho_pratica": "ganho com prática",
     "dem_idade": "idade", "dem_genero": "gênero", "dem_lateralidade": "lateralidade"}
rf = pd.read_csv("saidas/lofo_rf.csv").sort_values("lofo_ms")
lin = pd.read_csv("saidas/lofo_linear.csv").set_index("variavel")
fig, ax = plt.subplots(figsize=(8.6, 5.2))
hbar(ax, [R[v] for v in rf["variavel"]], rf["lofo_ms"].values, (1.96 * rf["dp_ms"] / np.sqrt(50)).values, unid=" ms")
ax.set_xlabel("aumento do RMSE ao remover a variável (ms)")
plt.savefig(OUT / "kr_lofo_rf.png"); plt.close()

ordem = rf["variavel"].tolist()
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
for ax, t, tt in [(axes[0], rf.set_index("variavel").loc[ordem, "lofo_ms"], "Random Forest  (RMSE 34,6 ms)"),
                  (axes[1], lin.loc[ordem, "lofo_ms"], "Regressão linear  (RMSE 31,4 ms)")]:
    hbar(ax, [R[v] for v in ordem], t.values, unid="")
    ax.set_title(tt); ax.set_xlabel("Δ RMSE (ms)")
plt.tight_layout(); plt.savefig(OUT / "kr_modelos.png"); plt.close()

red = pd.read_csv("saidas/lofo_redundancia.csv").set_index("variavel")["lofo_ms"]
pfi = pd.read_csv("saidas/pfi.csv", index_col=0)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
ax = axes[0]
x = np.arange(3); w = 0.38
v7 = rf.set_index("variavel")["lofo_ms"]["fix_flight_med"]
ax.bar([-w / 2], [v7], w, color=C["cinza"], label="7 variáveis")
ax.bar(x + w / 2, [red["fix_flight_med"], red["fix_total_med"], red["{flight + total_med}"]], w,
       color=[C["lar"], C["lar"], C["amar"]], label="+ tempo total (ρ = 0,79)")
for xi, v in zip([-w / 2, w / 2, 1 + w / 2, 2 + w / 2], [v7, red["fix_flight_med"], red["fix_total_med"], red["{flight + total_med}"]]):
    ax.text(xi, v + 0.25, f"{v:.1f}", ha="center", fontsize=14, color=C["tinta2"])
ax.set_xticks(x, ["flight", "tempo total", "grupo\n{flight, total}"]); ax.grid(axis="x", visible=False)
ax.set_title("LOFO: a importância se divide"); ax.set_ylabel("Δ RMSE (ms)"); ax.legend(loc="upper center")
ax.set_ylim(0, 13)
ax = axes[1]
ax.bar([-w / 2], [pfi.loc["fix_flight_med", "pfi_7"]], w, color=C["cinza"], label="7 variáveis")
ax.bar(np.arange(2) + w / 2, [pfi.loc["fix_flight_med", "pfi_8"], pfi.loc["fix_total_med", "pfi_8"]], w,
       color=C["azul"], label="+ tempo total")
for xi, v in zip([-w / 2, w / 2, 1 + w / 2], [pfi.loc["fix_flight_med", "pfi_7"], pfi.loc["fix_flight_med", "pfi_8"], pfi.loc["fix_total_med", "pfi_8"]]):
    ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=14, color=C["tinta2"])
ax.set_xticks(np.arange(2), ["flight", "tempo total"]); ax.grid(axis="x", visible=False)
ax.set_title("PFI: o modelo treinado nem percebe"); ax.set_ylabel("Δ RMSE ao permutar (ms)")
ax.legend(loc="upper right"); ax.set_ylim(0, 22)
plt.tight_layout(); plt.savefig(OUT / "kr_redundancia.png"); plt.close()

# ═══ Pinguins ═══
P = {"bill_length_mm": "compr. do bico", "bill_depth_mm": "prof. do bico", "flipper_length_mm": "nadadeira",
     "body_mass_g": "massa corporal", "species": "espécie", "island": "ilha"}
prf = pd.read_csv("saidas_pinguins/lofo_rf.csv").set_index("variavel")
plg = pd.read_csv("saidas_pinguins/lofo_logistica.csv").set_index("variavel")
ppfi = pd.read_csv("saidas_pinguins/pfi.csv", index_col=0)
ind = prf.loc[list(P)].sort_values("lofo")
ordem = ind.index.tolist()
fig, ax = plt.subplots(figsize=(8.6, 5.0))
hbar(ax, [P[v] for v in ordem], ind["lofo"].values, (1.96 * ind["dp"] / np.sqrt(50)).values, fmt="{:+.3f}")
ax.set_xlabel("aumento da log-loss ao remover a variável")
plt.savefig(OUT / "pg_lofo_rf.png"); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
hbar(axes[0], [P[v] for v in ordem], ppfi.loc[ordem, "rf"].values, cores=[C["azul"]] * 6, fmt="{:.2f}")
axes[0].set_title("PFI: permutar no modelo treinado"); axes[0].set_xlabel("Δ log-loss ao permutar")
hbar(axes[1], [P[v] for v in ordem], ind["lofo"].values, fmt="{:+.3f}")
axes[1].set_title("LOFO: remover e re-treinar"); axes[1].set_xlabel("Δ log-loss ao remover")
plt.tight_layout(); plt.savefig(OUT / "pg_pfi_lofo.png"); plt.close()

fig, ax = plt.subplots(figsize=(12.5, 5))
gs = [("body_mass_g", "massa"), ("bill_length_mm", "compr.\nbico"), ("bill_depth_mm", "prof.\nbico"),
      ("{bico}", "grupo\n{bico}"), ("{4 medidas}", "grupo\n{4 medidas}"), ("species", "espécie"),
      ("island", "ilha"), ("{species,island}", "grupo\n{espécie, ilha}")]
vals = [prf.loc[g, "lofo"] for g, _ in gs]
cores = [C["amar"] if g.startswith("{") else C["lar"] for g, _ in gs]
ax.bar(range(len(gs)), vals, color=cores, width=0.62)
for i, (v, (g, _)) in enumerate(zip(vals, gs)):
    ax.text(i, max(v, 0) + 0.01, f"{v:.3f}\n({prf.loc[g, 'queda_acuracia_pp']:.0f} pp)", ha="center", fontsize=13,
            color=C["tinta2"])
ax.set_xticks(range(len(gs)), [n for _, n in gs]); ax.axhline(0, color=C["tinta"], lw=1.2)
ax.set_ylabel("Δ log-loss ao remover"); ax.grid(axis="x", visible=False); ax.set_ylim(-0.02, 0.56)
ax.set_title("Sozinhas valem pouco; juntas, valem tudo  (entre parênteses: queda de acurácia)")
plt.tight_layout(); plt.savefig(OUT / "pg_grupos.png"); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
hbar(axes[0], [P[v] for v in ordem], ind["lofo"].values, fmt="{:+.3f}")
axes[0].set_title("Random Forest  (log-loss 0,243)"); axes[0].set_xlabel("Δ log-loss")
hbar(axes[1], [P[v] for v in ordem], plg.loc[ordem, "lofo"].values, fmt="{:+.3f}")
axes[1].set_title("Regressão logística  (log-loss 0,230)"); axes[1].set_xlabel("Δ log-loss")
plt.tight_layout(); plt.savefig(OUT / "pg_modelos.png"); plt.close()
print("ok")
