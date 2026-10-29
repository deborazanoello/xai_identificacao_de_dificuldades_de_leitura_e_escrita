"""Reconstroi a tabela da leitura (CompLex, termos de uma palavra) com as 8 variaveis do trabalho."""
import csv, re, sys
import numpy as np, pandas as pd
import pyphen
from wordfreq import zipf_frequency
R = sys.argv[1] if len(sys.argv) > 1 else "../dados/complex"   # copia dos arquivos do repositorio CompLex
partes = {"train": f"{R}/train/lcp_single_train.tsv", "test": f"{R}/test-labels/lcp_single_test.tsv"}   # 7.662 + 917 = 8.579
d = pd.concat([pd.read_csv(f, sep="\t", quoting=csv.QUOTE_NONE, keep_default_na=False).assign(parte=k)
               for k, f in partes.items()], ignore_index=True)
d["complexity"] = d["complexity"].astype(float)
hy = pyphen.Pyphen(lang="en_US")

def silabas(w):
    w = w.lower()
    n = len(hy.positions(w)) + 1
    v = len(re.findall(r"[aeiouy]+", w)) - (1 if w.endswith("e") and len(w) > 3 and not w.endswith("le") else 0)
    return max(n, v, 1)

def palavras(s): return re.findall(r"[A-Za-zÀ-ÿ0-9'-]+", s)

linhas = []
for _, r in d.iterrows():
    t, s = r["token"], r["sentence"]
    ws = palavras(s)
    idx = next((i for i, w in enumerate(ws) if w == t), next((i for i, w in enumerate(ws) if w.lower() == t.lower()), 0))
    sigla = (sum(c.isupper() for c in t) >= 2) or any(c.isupper() for c in t[1:])
    linhas.append({
        "id": r["id"], "corpus": r["corpus"], "parte": r["parte"], "termo": t, "grupo": t.lower(),
        "frequencia": zipf_frequency(t.lower(), "en"),
        "comprimento": len(t),
        "silabas": silabas(t),
        "sigla": int(sigla),
        "nome_proprio": int(t[:1].isupper() and not sigla and idx > 0),
        "tam_frase": len(ws),
        "posicao": idx / max(len(ws) - 1, 1),
        "forma_pt": int(zipf_frequency(t.lower(), "pt") >= 3.0),
        "complexidade": r["complexity"],
        "dificil": int(r["complexity"] >= 0.5)})
tab = pd.DataFrame(linhas)
tab.to_csv(sys.argv[2] if len(sys.argv) > 2 else "tabelas/tabela_complex_reconstruida.csv", index=False)
print("termos:", len(tab), "| palavras distintas:", tab.grupo.nunique(), "| dificeis:", round(100 * tab.dificil.mean(), 1), "%")
print("r(frequencia, complexidade) =", round(tab[["frequencia", "complexidade"]].corr().iloc[0, 1], 3))
print("r(comprimento, silabas) =", round(tab[["comprimento", "silabas"]].corr().iloc[0, 1], 3))
print("frequencia = 0 (palavra fora do dicionario):", int((tab.frequencia == 0).sum()))
print(tab[["frequencia", "comprimento", "silabas", "sigla", "nome_proprio", "tam_frase", "posicao", "forma_pt"]].describe().T.round(2).to_string())
print("exemplos de sigla:", tab.loc[tab.sigla == 1, "termo"].value_counts().head(12).to_dict())
print("exemplos de nome proprio:", tab.loc[tab.nome_proprio == 1, "termo"].value_counts().head(12).to_dict())
