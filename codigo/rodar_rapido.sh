#!/usr/bin/env bash
# Teste rápido (~10 min em 2 núcleos): refaz as tabelas e confere o artigo com os resultados salvos.
# Rode de dentro da pasta codigo/:  bash rodar_rapido.sh
set -e
python constroi_tabela_keyrecs.py            # deve dizer "maior diferença numérica: 4.4e-16"
python monta_tabela_complex.py               # 8.579 termos
python valida_complex.py                     # regra com AUC ~0,849 e floresta ~0,919
python lofo_complex.py rapido                # LOFO na leitura com 10 dobras (~5 min) -> resultados/saidas_complex_rapido/
python ajuste_holm.py
python conferir_numeros.py ../artigo/main.tex
echo "Pronto. Compare resultados/saidas_complex_rapido/ com resultados/saidas_complex/."
