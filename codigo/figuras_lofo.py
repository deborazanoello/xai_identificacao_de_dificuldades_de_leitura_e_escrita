"""Figuras do LOFO na escrita.

Barras de erro com o IC de 95% corrigido de Nadeau e Bengio (2003), coerente com os valores p.
Le resultados/saidas/ e resultados/saidas_comp/ e grava em resultados/figuras/.
Com SUFIXO=_corrigida, usa as saidas da tabela corrigida e grava em resultados/figuras_corrigida/.
Com TEMA=ufpr, usa as cores da UFPR e grava em resultados/figuras_ufpr/.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import os

SUF = os.environ.get("SUFIXO", "")
OUT = Path("resultados/figuras" + SUF); OUT.mkdir(parents=True, exist_ok=True)
S, SC = Path("resultados/saidas" + SUF), Path("resultados/saidas_comp" + SUF)
C = {"sup": "#F7F6F1", "tinta": "#1B2233", "tinta2": "#4A5263", "suave": "#7A8090", "grade": "#E2E0D8",
     "azul": "#2A68C2", "lar": "#D9572B", "amar": "#E0A400", "cinza": "#9AA0AC"}
if os.environ.get("TEMA") == "ufpr":      # cores do logo da UFPR, fundo branco
    C = {"sup": "#FFFFFF", "tinta": "#363435", "tinta2": "#4D4D4F", "suave": "#7D7D80", "grade": "#E6E9EC",
         "azul": "#7FA8C0", "lar": "#2C566F", "amar": "#5F6B73", "cinza": "#BAC0C6"}
    OUT = Path("resultados/figuras_ufpr"); OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({
    "figure.facecolor": C["sup"], "axes.facecolor": C["sup"], "savefig.facecolor": C["sup"],
    "axes.edgecolor": "#C9C6BA" if os.environ.get("TEMA") != "ufpr" else "#C3C8CD", "axes.labelcolor": C["tinta2"], "axes.titlecolor": C["tinta"],
    "axes.titlesize": 17, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": C["grade"], "grid.linewidth": 1,
    "xtick.color": C["tinta2"], "ytick.color": C["tinta2"], "legend.frameon": False,
    "font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 15, "legend.fontsize": 14,
    "savefig.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans"})
K, RAZAO = 50, 0.25
FATOR_IC = stats.t.ppf(0.975, K - 1) * np.sqrt(1 / K + RAZAO)   # meia-largura = FATOR_IC * dp
R = {"fix_flight_med": "tempo de voo", "fix_total_cv": "CV do tempo total",
     "fix_dwell_med": "pressionamento", "fix_ganho_pratica": "ganho com a prática",
     "dem_idade": "idade", "dem_genero": "gênero", "dem_lateralidade": "lateralidade"}


def br(x, casas=1, sinal=False):
    s = f"{x:+.{casas}f}" if sinal else f"{x:.{casas}f}"
    return s.replace(".", ",").replace("-", "−")


def hbar_ic(ax, nomes, media, lo, hi, p, unid=" ms", casas=None, cor_pos=None):
    """Barras horizontais com IC corrigido; destaca em laranja so o que tem p < 0,05."""
    y = np.arange(len(nomes))
    cores = [(cor_pos or C["lar"]) if pi < 0.05 else C["cinza"] for pi in p]
    ax.barh(y, media, color=cores, height=0.62)
    ax.errorbar(media, y, xerr=[media - lo, hi - media], fmt="none", ecolor=C["tinta"], capsize=4, lw=1.3)
    ax.set_yticks(y, nomes); ax.axvline(0, color=C["tinta"], lw=1.2); ax.grid(axis="y", visible=False)
    span = max(np.max(np.abs(hi)), np.max(np.abs(lo)), 1e-9)
    for i in range(len(nomes)):
        if casas is not None:
            c = casas
        elif unid == "%":
            c = 1 if abs(media[i]) < 10 else 0
        else:
            c = 2 if abs(media[i]) < 1 else 1
        txt = br(media[i], c, sinal=True) + unid + ("  *" if p[i] < 0.05 else "")
        ax.text(max(hi[i], 0) + 0.03 * span, i, txt, va="center", fontsize=13.5, color=C["tinta2"])
    ax.set_xlim(min(0, np.min(lo)) - 0.06 * span, np.max(hi) + 0.42 * span)


import matplotlib.ticker as mticker


def _fmt(v, _pos=None):
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}".replace("-", "−")
    return f"{v:.2f}".rstrip("0").rstrip(".").replace(".", ",").replace("-", "−")


import re as _re

# Vocabulario do deck (flight, dwell) no lugar do vocabulario do artigo (voo, pressionamento)
LONGO = {"tempo de voo": "flight (pausa entre teclas)", "pressionamento": "dwell (tecla pressionada)"}


def deck(t, longo=False):
    if longo and t in LONGO:
        return LONGO[t]
    for a, b in [("só tempo de voo", "só flight"), ("só pressionamento", "só dwell"), ("tempo de voo", "flight"),
                 ("pressionamento", "dwell"), ("press.,", "dwell,"), ("ganho com a prática", "ganho com prática"),
                 ("mediana do voo", "mediana do flight")]:
        t = t.replace(a, b)
    return _re.sub(r"\bvoo\b", "flight", t)


def salva(nome):
    fig = plt.gcf()
    fig.canvas.draw()
    for a in fig.axes:
        for eixo, longo in ((a.xaxis, False), (a.yaxis, True)):
            f = eixo.get_major_formatter()
            if isinstance(f, mticker.ScalarFormatter):
                eixo.set_major_formatter(mticker.FuncFormatter(_fmt))
            elif isinstance(f, mticker.FuncFormatter) and f.func is not _fmt:      # eixo categorico
                locs = eixo.get_ticklocs()
                labs = [t.get_text() for t in eixo.get_ticklabels()]
                if any(labs):                       # eixos compartilhados sem rotulo: nao mexer
                    vis = [t.get_visible() for t in eixo.get_ticklabels()]
                    eixo.set_ticks(locs, [deck(x, longo) for x in labs])
                    for t, v in zip(eixo.get_ticklabels(), vis):
                        t.set_visible(v)
            eixo.label.set_text(deck(eixo.label.get_text()))
        for tt in (a.title, a._left_title, a._right_title):     # preserva pad e estilo
            tt.set_text(deck(tt.get_text()))
        for t in a.texts:
            t.set_text(deck(t.get_text()))
        leg = a.get_legend()
        if leg:
            for t in leg.get_texts():
                t.set_text(deck(t.get_text()))
    for t in fig.texts:
        t.set_text(deck(t.get_text()))
    for leg in fig.legends:
        for t in leg.get_texts():
            t.set_text(deck(t.get_text()))
    plt.savefig(OUT / nome); plt.close()


def nota(fig, txt, y=-0.02):
    fig.text(0.01, y, txt, fontsize=11.5, color=C["suave"], ha="left", va="top")


# ════════ 1. LOFO individual com IC corrigido (substitui a figura do slide 8)
rf = pd.read_csv(S / "lofo_rf.csv").sort_values("lofo_ms")
m = rf["lofo_ms"].values; h = FATOR_IC * rf["dp_ms"].values
fig, ax = plt.subplots(figsize=(9.2, 5.2))
hbar_ic(ax, [R[v] for v in rf["variavel"]], m, m - h, m + h, rf["p_nb"].values)
ax.set_xlabel("aumento do RMSE ao remover a variável (ms)")
nota(fig, "Barras: IC 95% com correção de Nadeau & Bengio (2003) para dobras dependentes · * p < 0,05")
salva("01_lofo_rf_ic_corrigido.png")

# ════════ 2. Antes x depois: IC ingenuo x corrigido
fig, ax = plt.subplots(figsize=(9.2, 5.2))
y = np.arange(len(rf)); ing = 1.96 * rf["dp_ms"].values / np.sqrt(K)
ax.errorbar(m, y + 0.17, xerr=ing, fmt="o", color=C["suave"], ecolor=C["suave"], capsize=4, lw=1.6, ms=7,
            label="IC sem correção: 1,96·dp/√50 (trata as 50 dobras como independentes)")
ax.errorbar(m, y - 0.17, xerr=h, fmt="o", color=C["lar"], ecolor=C["lar"], capsize=4, lw=1.6, ms=7,
            label="IC corrigido (Nadeau & Bengio)")
ax.axvline(0, color=C["tinta"], lw=1.2)
ax.set_yticks(y, [R[v] for v in rf["variavel"]]); ax.grid(axis="y", visible=False)
ax.set_xlim(-1.2, 3.0)
ax.annotate(f"tempo de voo: {br(m[-1],1)} ms\n(fora da escala)", xy=(3.0, y[-1]), xytext=(1.25, y[-1]),
            va="center", fontsize=12.5, color=C["tinta2"], arrowprops={"arrowstyle": "->", "color": C["suave"]})
ax.set_xlabel("aumento do RMSE ao remover (ms) · zoom perto de zero")
ax.legend(loc="lower right", fontsize=12)
salva("02_ic_ingenuo_x_corrigido.png")

# ════════ 3. Blocos
if (SC / "A_lofo_blocos.csv").exists():
    blo = pd.read_csv(SC / "A_lofo_blocos.csv")
    ind = pd.read_csv(SC / "A_lofo_individual.csv").set_index("variavel")
    ordem = ["{demográficas}", "{motoras sem voo}", "{voo, pressionamento}", "{motoras}"]
    blo = blo.set_index("variavel").loc[ordem].reset_index()
    rot = {"{demográficas}": "demográficas\n(idade, gênero, lat.)", "{motoras sem voo}": "motoras sem o voo\n(press., CV, ganho)",
           "{voo, pressionamento}": "voo + pressionamento", "{motoras}": "as 4 motoras"}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    hbar_ic(ax, [rot[b] for b in blo["variavel"]], blo["media"].values, blo["ic_inf"].values,
            blo["ic_sup"].values, blo["p"].values, cor_pos=C["amar"])
    ax.scatter(blo["soma_individuais"], np.arange(len(blo)), marker="D", s=70, color=C["tinta"], zorder=5,
               label="soma dos LOFO individuais")
    ax.set_xlabel("aumento do RMSE ao remover o bloco (ms)")
    ax.legend(loc="lower right", fontsize=13)
    nota(fig, "Barras: LOFO do bloco inteiro, IC 95% corrigido · ◆ soma dos LOFO de cada variável do bloco")
    salva("03_lofo_blocos.png")

# ════════ 4. Suficiencia
if (SC / "B_suficiencia.csv").exists():
    suf = pd.read_csv(SC / "B_suficiencia.csv")
    rot = {f"só {k}": f"só {v}" for k, v in R.items()}
    rot.update({"só {voo, CV}": "só voo + CV", "mediana": "nenhuma (mediana)", "todas": "as 7 variáveis"})
    suf = suf.sort_values("rmse_ms", ascending=False)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    y = np.arange(len(suf))
    cores = []
    for n in suf["config"]:
        if n == "todas": cores.append(C["azul"])
        elif n in ("só fix_flight_med", "só {voo, CV}"): cores.append(C["lar"])
        else: cores.append(C["cinza"])
    ax.barh(y, suf["rmse_ms"], color=cores, height=0.62)
    full = suf.loc[suf["config"] == "todas", "rmse_ms"].iloc[0]
    ax.axvline(full, color=C["azul"], lw=1.4, ls="--")
    for i, (r_, n) in enumerate(zip(suf["rmse_ms"], suf["config"])):
        ax.text(r_ + 1.2, i, br(r_, 1) + " ms", va="center", fontsize=13.5, color=C["tinta2"])
    ax.set_yticks(y, [rot[n] for n in suf["config"]]); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, suf["rmse_ms"].max() * 1.18)
    ax.set_xlabel("RMSE da floresta em validação cruzada (ms) · menor é melhor")
    salva("04_suficiencia.png")

# ════════ 5. PFI x LOFO nas mesmas dobras
if (SC / "C_pfi_mesmas_dobras.csv").exists():
    pfi = pd.read_csv(SC / "C_pfi_mesmas_dobras.csv")
    p7 = pfi[pfi["modelo"] == "7 variáveis"].set_index("variavel")
    ind = pd.read_csv(SC / "A_lofo_individual.csv").set_index("variavel")
    ordem = ind.sort_values("media").index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, tab, tt, xl in [(axes[0], p7, "Permutar (PFI)", "Δ RMSE ao permutar (ms)"),
                            (axes[1], ind, "Remover e retreinar (LOFO)", "Δ RMSE ao remover (ms)")]:
        t = tab.loc[ordem]
        hbar_ic(ax, ordem, t["media"].values, t["ic_inf"].values, t["ic_sup"].values, t["p"].values,
                cor_pos=C["azul"] if ax is axes[0] else C["lar"])
        ax.set_title(tt); ax.set_xlabel(xl)
    plt.tight_layout()
    nota(fig, "Mesmas 50 dobras e mesmo modelo nos dois painéis · IC 95% corrigido · * p < 0,05")
    salva("05_pfi_x_lofo_mesmas_dobras.png")

    p8 = pfi[pfi["modelo"] != "7 variáveis"].set_index("variavel")
    red = pd.read_csv(S / "lofo_redundancia.csv").set_index("variavel")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    # (a) LOFO
    ax = axes[0]; w = 0.38
    itens = [(-w / 2, rf.set_index("variavel").loc["fix_flight_med"], C["cinza"]),
             (w / 2, red.loc["fix_flight_med"], C["lar"]), (1 + w / 2, red.loc["fix_total_med"], C["lar"]),
             (2, red.loc["{flight + total_med}"], C["amar"])]
    for xi, row, cor in itens:
        mm, hh = row["lofo_ms"], FATOR_IC * row["dp_ms"]
        ax.bar(xi, mm, w, color=cor, hatch="//" if cor == C["amar"] else None, edgecolor=C["sup"])
        ax.errorbar(xi, mm, yerr=hh, fmt="none", ecolor=C["tinta"], capsize=4, lw=1.3)
        ax.text(xi, mm + hh + 0.5, br(mm), ha="center", fontsize=14, color=C["tinta2"])
    ax.set_xticks([0, 1, 2], ["tempo de voo", "tempo total", "grupo\n{voo, total}"]); ax.grid(axis="x", visible=False)
    ax.axhline(0, color=C["tinta"], lw=1.2)
    ax.set_ylabel("Δ RMSE ao remover (ms)"); ax.set_title("LOFO: a importância se divide")
    # (b) PFI
    ax = axes[1]
    itens = [(-w / 2, p7.loc["tempo de voo"], C["cinza"]), (w / 2, p8.loc["tempo de voo"], C["azul"]),
             (1 + w / 2, p8.loc["tempo total"], C["azul"])]
    for xi, row, cor in itens:
        ax.bar(xi, row["media"], w, color=cor, edgecolor=C["sup"])
        ax.errorbar(xi, row["media"], yerr=[[row["media"] - row["ic_inf"]], [row["ic_sup"] - row["media"]]],
                    fmt="none", ecolor=C["tinta"], capsize=4, lw=1.3)
        ax.text(xi, row["ic_sup"] + 0.5, br(row["media"]), ha="center", fontsize=14, color=C["tinta2"])
    ax.set_xticks([0, 1], ["tempo de voo", "tempo total"]); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Δ RMSE ao permutar (ms)"); ax.set_title("PFI: cai, mas bem menos")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=C["cinza"], label="7 variáveis"),
                        Patch(facecolor=C["lar"], label="+ tempo total (LOFO)"),
                        Patch(facecolor=C["azul"], label="+ tempo total (PFI)")],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), fontsize=13.5)
    plt.tight_layout()
    l7, l8 = rf.set_index("variavel").loc["fix_flight_med", "lofo_ms"], red.loc["fix_flight_med", "lofo_ms"]
    q7, q8 = p7.loc["tempo de voo", "media"], p8.loc["tempo de voo", "media"]
    nota(fig, f"LOFO e PFI nas mesmas 50 dobras · IC 95% corrigido · LOFO do voo: {br(l7)} → {br(l8)} ms "
              f"(−{100 * (1 - l8 / l7):.0f}%) · PFI do voo: {br(q7)} → {br(q8)} ms (−{100 * (1 - q8 / q7):.0f}%)")
    salva("06_substituta_lofo_x_pfi.png")

# ════════ 6. Ruido da floresta x das dobras
if (SC / "E_ruido.csv").exists():
    ru = pd.read_csv(SC / "E_ruido.csv")
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = np.arange(len(ru)); w = 0.36
    ax.barh(y + w / 2, ru["dp_dobra"], w, color=C["azul"], label="variação entre dobras (dados)")
    ax.barh(y - w / 2, ru["dp_semente"], w, color=C["amar"], label="variação entre sementes (floresta)")
    for i in range(len(ru)):
        ax.text(ru["dp_dobra"][i] + 0.08, i + w / 2, br(ru["dp_dobra"][i], 2) + " ms", va="center", fontsize=13.5, color=C["tinta2"])
        ax.text(ru["dp_semente"][i] + 0.08, i - w / 2, br(ru["dp_semente"][i], 2) + " ms", va="center", fontsize=13.5, color=C["tinta2"])
    ax.set_yticks(y, ru["variavel"]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("desvio-padrão do LOFO (ms)"); ax.legend(loc="lower right", fontsize=13)
    ax.set_xlim(0, max(ru["dp_dobra"].max(), ru["dp_semente"].max()) * 1.3)
    nota(fig, "10 dobras × 10 sementes da floresta, mesmas partições")
    salva("07_ruido_floresta_x_dobras.png")

# ════════ 7. Hiperparametros fixos x reajustados
if (SC / "F_lofo_hp_reajustado.csv").exists() and (pd.read_csv(SC / "F_lofo_hp_reajustado.csv")["media"].values != pd.read_csv(SC / "A_lofo_individual.csv")["media"].values).any():
    fx = pd.read_csv(SC / "A_lofo_individual.csv").set_index("variavel")
    rj = pd.read_csv(SC / "F_lofo_hp_reajustado.csv").set_index("variavel")
    ordem = fx.sort_values("media").index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(ordem))
    for i, v in enumerate(ordem):
        a, b = fx.loc[v, "media"], rj.loc[v, "media"]
        ax.plot([a, b], [i, i], color=C["grade"], lw=3, zorder=1)
    ax.scatter(fx.loc[ordem, "media"], y, s=110, color=C["cinza"], zorder=3, label="hiperparâmetros fixos (original)")
    ax.scatter(rj.loc[ordem, "media"], y, s=110, color=C["lar"], zorder=4, label="reajustados sem a variável")
    for i, v in enumerate(ordem):
        ax.text(max(fx.loc[v, "media"], rj.loc[v, "media"]) + 0.35, i,
                f"{br(fx.loc[v,'media'],2,True)} → {br(rj.loc[v,'media'],2,True)}", va="center", fontsize=12.5, color=C["tinta2"])
    ax.axvline(0, color=C["tinta"], lw=1.2)
    ax.set_yticks(y, ordem); ax.grid(axis="y", visible=False)
    ax.set_xlim(min(-0.6, fx["media"].min() - 0.3), max(fx["media"].max(), rj["media"].max()) * 1.45)
    ax.set_xlabel("LOFO (ms)"); ax.legend(loc="lower right", fontsize=13)
    salva("08_hp_fixos_x_reajustados.png")

# ════════ 8. Outras perdas (em % da perda do modelo completo)
if (SC / "D_outras_perdas.csv").exists():
    d = pd.read_csv(SC / "D_outras_perdas.csv")
    d["pct"] = 100 * d["media"] / d["base_ms"]
    d["pct_lo"] = 100 * d["ic_inf"] / d["base_ms"]; d["pct_hi"] = 100 * d["ic_sup"] / d["base_ms"]
    ordem = d[d["metrica"] == "rmse"].sort_values("media")["variavel"].tolist()
    mets = [("rmse", "RMSE (todos)", C["lar"]), ("mae", "MAE (todos)", C["amar"]),
            ("rmse_top25", "RMSE nos 25% mais hesitantes", C["azul"])]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    y = np.arange(len(ordem)); w = 0.26
    for j, (met, lab, cor) in enumerate(mets):
        t = d[d["metrica"] == met].set_index("variavel").loc[ordem]
        pos = y + (1 - j) * w
        ax.barh(pos, t["pct"], w, color=cor, label=lab)
        ax.errorbar(t["pct"], pos, xerr=[t["pct"] - t["pct_lo"], t["pct_hi"] - t["pct"]], fmt="none",
                    ecolor=C["tinta"], capsize=3, lw=1)
    ax.axvline(0, color=C["tinta"], lw=1.2)
    ax.set_yticks(y, ordem); ax.grid(axis="y", visible=False)
    ax.set_xlabel("aumento da perda ao remover (% da perda do modelo completo)")
    ax.legend(loc="lower right", fontsize=12.5)
    salva("09_outras_perdas.png")

# ════════ 9. Outros alvos
if (SC / "G_outros_alvos.csv").exists():
    g = pd.read_csv(SC / "G_outros_alvos.csv")
    h0 = pd.read_csv(SC / "A_lofo_individual.csv")
    base_h = pd.read_csv(SC / "D_outras_perdas.csv").query("metrica == 'rmse'")["base_ms"].iloc[0]
    h0 = h0.assign(alvo="hesitacao_livre", pct=100 * h0["media"] / base_h,
                   pct_lo=100 * h0["ic_inf"] / base_h, pct_hi=100 * h0["ic_sup"] / base_h)
    g = g.assign(pct=100 * g["media"] / g["rmse_base_pp"], pct_lo=100 * g["ic_inf"] / g["rmse_base_pp"],
                 pct_hi=100 * g["ic_sup"] / g["rmse_base_pp"])
    todos = pd.concat([h0, g])
    tit = {"hesitacao_livre": "Hesitação\n(mediana do voo)", "taxa_pausa_longa": "Taxa de\npausas longas",
           "taxa_revisao": "Taxa de\nrevisão"}
    ordem = h0.sort_values("media")["variavel"].tolist()
    rm = pd.read_csv(SC / "G_outros_alvos.csv").groupby("alvo")[["rmse_base_pp", "rmse_mediana_pp"]].first()
    import json as _json
    rz = next(r for r in (_json.load(open(f)) for f in sorted(SC.glob("resumo_*.json"))) if "rmse_todas_ms" in r)
    sub = {"hesitacao_livre": f"RMSE {br(rz['rmse_todas_ms'])} × {br(rz['rmse_mediana_ms'])} ms da mediana",
           "taxa_pausa_longa": f"RMSE {br(rm.loc['taxa_pausa_longa','rmse_base_pp'],2)} × {br(rm.loc['taxa_pausa_longa','rmse_mediana_pp'],2)} pp da mediana",
           "taxa_revisao": f"RMSE {br(rm.loc['taxa_revisao','rmse_base_pp'],2)} × {br(rm.loc['taxa_revisao','rmse_mediana_pp'],2)} pp: quase sem sinal"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True)
    for ax, a in zip(axes, tit):
        t = todos[todos["alvo"] == a].set_index("variavel").loc[ordem]
        hbar_ic(ax, ordem, t["pct"].values, t["pct_lo"].values, t["pct_hi"].values, t["p"].values, unid="%")
        ax.set_title(tit[a], fontsize=15, pad=28); ax.set_xlabel("Δ perda (% do RMSE)")
        ax.text(0.0, 1.015, sub[a], transform=ax.transAxes, fontsize=12, color=C["suave"], va="bottom", ha="left")
    plt.tight_layout()
    nota(fig, "Mesmas 7 variáveis da senha e do perfil; floresta reajustada para cada alvo · IC 95% corrigido · * p < 0,05, sem correção para as 7 comparações")
    salva("10_outros_alvos.png")

# ════════ 11. Robustez: tabela do grupo x reconstruida da base bruta
if (SC / "lofo_robustez_resumo.csv").exists():
    rs = pd.read_csv(SC / "lofo_robustez_resumo.csv")
    rt = pd.read_csv(SC / "lofo_robustez_tabelas.csv")
    versoes = rs["versao"].tolist()
    rot = {"tabela do grupo (96)": "tabela do\ngrupo (96)", "reconstruida (96)": "reconstruída\nda base (96)",
           "reconstruida (99)": "reconstruída +\n3 excluídos (99)"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]; x = np.arange(len(versoes))
    fl = rt[rt["variavel"] == "fix_flight_med"].set_index("versao").loc[versoes]
    ax.bar(x, fl["lofo_ms"], 0.55, color=C["lar"])
    for i, (m_, p_) in enumerate(zip(fl["lofo_ms"], fl["p"])):
        ax.text(i, m_ + 0.3, br(m_) + " ms\np " + ("< 0,001" if p_ < 0.001 else "= " + br(p_, 3)), ha="center", fontsize=13.5, color=C["tinta2"])
    ax.set_xticks(x, [rot[v] for v in versoes], fontsize=12.5); ax.grid(axis="x", visible=False)
    ax.set_ylabel("LOFO do voo (ms)"); ax.set_ylim(0, fl["lofo_ms"].max() * 1.35); ax.set_title("LOFO do voo")
    ax = axes[1]; w = 0.26
    for j, (col, lab, cor) in enumerate([("rmse_ms", "7 variáveis", C["azul"]), ("so_voo_ms", "só voo", C["lar"]),
                                         ("mediana_ms", "nenhuma (mediana)", C["cinza"])]):
        ax.bar(x + (j - 1) * w, rs[col], w, color=cor, label=lab)
        for i, v_ in enumerate(rs[col]):
            ax.text(i + (j - 1) * w, v_ + 0.6, br(v_), ha="center", fontsize=11.5, color=C["tinta2"])
    ax.set_xticks(x, [rot[v] for v in versoes], fontsize=12.5); ax.grid(axis="x", visible=False)
    ax.set_ylabel("RMSE (ms)"); ax.set_ylim(0, rs["mediana_ms"].max() * 1.22); ax.legend(loc="upper left", ncol=3, fontsize=12.5)
    ax.set_title("Só o voo empata com as 7 nas três versões")
    plt.tight_layout()
    nota(fig, "Mesmo protocolo (floresta, 5×10 dobras) · os 3 excluídos têm hesitação ≤ 0 (digitação com teclas sobrepostas)")
    salva("11_robustez_base.png")

print("figuras em", OUT)
