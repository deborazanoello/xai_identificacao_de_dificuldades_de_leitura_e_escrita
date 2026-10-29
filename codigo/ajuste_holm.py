"""Correção para comparações múltiplas (Holm, 1979) nos p-valores do LOFO.

Cada LOFO testa várias variáveis ao mesmo tempo (7 na escrita, 8 na leitura). O ajuste de Holm controla a
chance de pelo menos um falso positivo entre elas. Lê os resultados já calculados; não refaz nenhum modelo.

Uso: python ajuste_holm.py     (gera resultados/saidas_holm/lofo_escrita_holm.csv e resultados/saidas_holm/lofo_leitura_holm.csv)
     Na escrita, usa a tabela corrigida (resultados/saidas_corrigida/); com SUFIXO="" usa a tabela do grupo (resultados/saidas/).
"""
import os
from pathlib import Path
import numpy as np, pandas as pd

SUF = os.environ.get("SUFIXO", "_corrigida")
OUT = Path("resultados/saidas_holm"); OUT.mkdir(parents=True, exist_ok=True)


def holm(p):
    p = np.asarray(p, float); m = len(p); ordem = np.argsort(p)
    aj = np.empty(m); acumulado = 0.0
    for posicao, i in enumerate(ordem):
        acumulado = max(acumulado, (m - posicao) * p[i])
        aj[i] = min(acumulado, 1.0)
    return aj


e = pd.read_csv(f"resultados/saidas{SUF}/lofo_rf.csv")
e["p_holm"] = holm(e["p_nb"])
e.to_csv(OUT / "lofo_escrita_holm.csv", index=False)
print("Escrita (RMSE, ms):\n", e[["variavel", "lofo_ms", "p_nb", "p_holm"]].round(4).to_string(index=False))

l = pd.read_csv("resultados/saidas_complex/A_lofo_individual.csv").query("modelo == 'rf' and metrica == 'auc'").copy()
l["p_holm"] = holm(l["p"])
l.to_csv(OUT / "lofo_leitura_holm.csv", index=False)
print("\nLeitura (AUC x 100):\n", l[["variavel", "media", "p", "p_holm"]].round(4).to_string(index=False))
