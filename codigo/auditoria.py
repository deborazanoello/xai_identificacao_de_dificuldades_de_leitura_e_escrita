"""Auditoria da base KeyRecs bruta x tabelas/tabela_analitica.csv."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
B = sys.argv[1] if len(sys.argv) > 1 else "../dados/keyrecs"   # pasta com os CSV brutos
dem = pd.read_csv(f"{B}/demographics.csv")
fx = pd.read_csv(f"{B}/fixed-text.csv")
fr = pd.read_csv(f"{B}/free-text.csv", low_memory=False, dtype=str)
fr.columns = [c.strip() for c in fr.columns]
tab = pd.read_csv("tabelas/tabela_analitica.csv", index_col="participant")   # tabela do grupo, nesta pasta
print("=== DEMOGRAFIA")
print("linhas", len(dem), "| ids unicos", dem.participant.nunique(), "| faltantes", int(dem.isna().sum().sum()))
print("idade", dem.age.min(), "-", dem.age.max(), "| genero", dem.gender.value_counts().to_dict(), "| lateralidade", dem.handedness.value_counts().to_dict())
ids = set(dem.participant); print("ids ausentes em p001..p100:", sorted({f'p{i:03d}' for i in range(1,101)} - ids))
print("ids iguais nos 3 arquivos:", ids == set(fx.participant) == set(fr.participant))
# conferir demografia da tabela
t = tab.join(dem.set_index("participant"), rsuffix="_dem")
print("tabela x demographics: idade", (t.dem_idade == t.age).all(), "| genero", (t.dem_genero == t.gender).all(),
      "| lateralidade", (t.dem_lateralidade == t.handedness).all(), "| nacionalidade", (t.nationality == t.nationality_dem).all())

print("\n=== SENHA (fixed-text)")
cols = fx.columns.tolist()
dw = [c for c in cols if c.startswith("DU.") and c.split(".")[1] == c.split(".")[2]]
pause = [c for c in cols if c.startswith("DU.") and c.split(".")[1] != c.split(".")[2]]   # pausa real (rotulo trocado)
print("faltantes:", int(fx.isna().sum().sum()), "| duplicadas:", int(fx.duplicated().sum()),
      "| duplicadas (participante, sessao, repeticao):", int(fx.duplicated(["participant","session","repetition"]).sum()))
n = fx.groupby(["participant","session"]).size().unstack()
print("repeticoes por sessao: min", int(n.min().min()), "max", int(n.max().max()), "| participantes com sessao != 100:", n[(n != 100).any(axis=1)].index.tolist())
print("dwell <= 0:", int((fx[dw] <= 0).sum().sum()), "| dwell > 1 s:", int((fx[dw] > 1).sum().sum()))
print("pausa < 0 (sobreposicao):", round(100 * (fx[pause] < 0).values.mean(), 2), "% | pausa > 5 s:", int((fx[pause] > 5).sum().sum()))
tt = fx["total time"]
print("tempo total <= 0:", fx.loc[tt <= 0, ["participant","session","repetition","total time"]].values.tolist())
print("tempo total > 30 s:", fx.loc[tt > 30, ["participant","session","repetition","total time"]].values.tolist())
soma = fx[dw].sum(axis=1) + fx[pause].sum(axis=1)
ok = np.isclose(soma, tt, atol=0.01)
print("tempo total = soma dos dwell + pausas:", round(100 * ok.mean(), 2), "% das linhas | linhas que nao batem:", int((~ok).sum()))
print("   exemplos:", fx.loc[~ok, ["participant","session","repetition","total time"]].assign(soma=soma[~ok].round(3)).head(8).values.tolist())

print("\n=== TEXTO LIVRE (free-text)")
num = ["DU.key1.key1","DD.key1.key2","DU.key1.key2","UD.key1.key2","UU.key1.key2"]
for c in num:
    v = pd.to_numeric(fr[c], errors="coerce"); bad = fr[c][v.isna() & fr[c].notna()]
    print(f"{c}: nao numericos = {len(bad)} {bad.unique()[:5].tolist()} | faltantes = {int(fr[c].isna().sum())}")
    fr[c] = v
print("coluna extra (virgula no fim da linha):", [c for c in fr.columns if c.startswith("Unnamed")], "-> vazia:", fr[[c for c in fr.columns if c.startswith('Unnamed')]].isna().all().all())
ident = np.isclose(fr["UD.key1.key2"], fr["DD.key1.key2"] - fr["DU.key1.key1"], atol=2e-3)
print("identidade UD = DD - dwell1 no texto livre:", round(100 * ident.mean(), 2), "% (rotulos corretos)")
print("sessoes por participante:", fr.groupby("participant").session.nunique().value_counts().to_dict())
ud = fr["UD.key1.key2"]
print("pausas: < 0:", round(100*(ud<0).mean(),1), "% | > 2 s:", round(100*(ud>2).mean(),2), "% | > 60 s:", int((ud>60).sum()), "| max:", ud.max())
print("participantes com pausa > 60 s:", fr.loc[ud>60,"participant"].value_counts().to_dict())
g = fr.groupby("participant")
rec = pd.DataFrame({"hes": g["UD.key1.key2"].median(), "n": g.size(), "n_validos": g["UD.key1.key2"].count(),
                    "backspace": g.apply(lambda d: (d.key1=="Backspace").mean())})
r = rec.reindex(tab.index)
print("hesitacao: |dif| max", round((r.hes - tab.hesitacao_livre).abs().max(),4), "| n_digrafos: dif = n - tabela:", (r.n - tab.n_digrafos_livre).describe()[["min","50%","max"]].to_dict())
print("   n_validos (UD numerico) - tabela:", (r.n_validos - tab.n_digrafos_livre).describe()[["min","50%","max"]].to_dict())

print("\n=== EXCLUIDOS (p043, p071, p078) x os 96")
fl = fx.groupby("participant").apply(lambda d: np.nanmedian(d[pause].values))
dd = pd.DataFrame({"voo": fl, "dwell": fx.groupby("participant").apply(lambda d: np.nanmedian(d[dw].values)),
                   "total": fx.groupby("participant")["total time"].median(), "hes": rec.hes,
                   "backspace": rec.backspace, "n_livre": rec.n})
z = (dd - dd.loc[tab.index].mean()) / dd.loc[tab.index].std()
for p in ["p043","p071","p078"]:
    print(p, dd.loc[p].round(3).to_dict(), "| z:", z.loc[p].round(1).to_dict())
