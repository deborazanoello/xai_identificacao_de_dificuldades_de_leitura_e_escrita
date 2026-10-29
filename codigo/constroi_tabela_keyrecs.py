"""Constrói a tabela analítica da escrita (KeyRecs) com o código do grupo.

Código copiado das células C04 e C05 do notebook ATV3-KeyRecs.ipynb, que por sua vez
reproduzem a construção da ATV02. Única mudança: compatibilidade com o pandas 3
(colunas de texto deixaram de ter dtype 'object').

Os arquivos brutos (demographics.csv, fixed-text.csv, free-text.csv) ficam em ../dados/keyrecs;
outra pasta pode ser indicada como argumento:
    python constroi_tabela_keyrecs.py [pasta_dos_csv]
Gera tabelas/tabela_analitica_refeita.csv (igual à do grupo) e tabelas/tabela_analitica_corrigida.csv, em que o tempo de voo
da senha é a pausa real (ver o fim do script). Compara a primeira com tabelas/tabela_analitica.csv (se existir).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DIR_DADOS = Path(sys.argv[1] if len(sys.argv) > 1 else "../dados/keyrecs")
display = lambda *a, **k: None

# ══ C04 (ATV3) ══
# ══ C04 · Carga robusta + esquema (condensado da ATV2) ═══════════════════
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
        if not pd.api.types.is_numeric_dtype(df[c]):  # pandas 3: texto não é mais 'object'
            df[c] = pd.to_numeric(df[c], errors="coerce")

assert HOLDS_FIXO and BLOCOS_FIXO and HOLD_LIVRE and COL_DD_LIVRE, "Esquema não detectado."

print(f"demographics : {demo.shape}   participantes = {demo['participant'].nunique()}")
print(f"fixed-text   : {fixo.shape}   dwell = {len(HOLDS_FIXO)}   dígrafos = {len(BLOCOS_FIXO)}")
print(f"free-text    : {livre.shape}   (formato longo: 1 linha por dígrafo digitado)")
print(f"\nMapeamento: dwell_livre={HOLD_LIVRE!r}  DD_livre={COL_DD_LIVRE!r}  total_fixo={col_total!r}")

# ══ C05 (ATV3) ══
# ══ C05 · Tabela analítica (mesma construção da ATV2) ════════════════════
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


dados.to_csv("tabelas/tabela_analitica_refeita.csv")
print("\nGravada: tabelas/tabela_analitica_refeita.csv")

# ── Comparação com a tabela usada nos modelos ────────────────────────────
ref_path = Path("tabelas/tabela_analitica.csv")
if ref_path.exists():
    ref = pd.read_csv(ref_path, index_col=0)
    num = [c for c in ref.columns if ref[c].dtype.kind in "fi"]
    mesmo_idx = ref.index.equals(dados.index)
    dif = float((ref[num] - dados[num].reindex(ref.index)).abs().max().max())
    print(f"Mesmos participantes: {mesmo_idx}   maior diferença numérica: {dif:.1e}")

# ── Por que o flight da senha tem r = 0,994 (e não 1) com a pausa real ───
# No fixed-text.csv os rótulos dos pares estão trocados: a coluna rotulada DD é, na
# verdade, soltar 1ª → soltar 2ª (UU), e a rotulada DU é a pausa real (soltar 1ª → apertar 2ª).
# Assim, "DD − dwell(1ª)" = pausa real + dwell(2ª) − dwell(1ª). No free-text.csv os rótulos estão certos.
excl = hesitacao_livre[hesitacao_livre <= 0]
print("Excluídos por hesitação <= 0:", ", ".join(f"{p} ({v:.3f} s)" for p, v in excl.items()))
cols_real = []
for b in BLOCOS_FIXO:
    n = "pausa_real." + b["par"].split(".", 1)[1]
    fixo_ok[n] = fixo_ok[b["DU"]]
    cols_real.append(n)
pausa_real = mediana_geral(fixo_ok, cols_real).reindex(dados.index)
r = np.corrcoef(dados["fix_flight_med"], pausa_real)[0, 1]
dif = (dados["fix_flight_med"] - pausa_real).abs()
print(f"flight do grupo × pausa real: r = {r:.4f}; diferença mediana {dif.median()*1000:.1f} ms; máxima {dif.max()*1000:.1f} ms")

# ── Tabela corrigida: tempo de voo da senha = mediana da pausa real ───────
corrigida = dados.copy()
corrigida["fix_flight_med"] = mediana_geral(fixo_ok, cols_real).reindex(corrigida.index)
corrigida.to_csv("tabelas/tabela_analitica_corrigida.csv")
print("Gravada: tabelas/tabela_analitica_corrigida.csv (só fix_flight_med muda)")
