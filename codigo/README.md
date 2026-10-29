# Código

Rode tudo de dentro desta pasta. Os dados brutos ficam em `../dados/`, as tabelas em `tabelas/` e as saídas de cada
script em `resultados/`.

```bash
pip install -r requirements.txt
bash rodar_rapido.sh   # refaz as tabelas e confere o artigo, uns 10 minutos
bash rodar_tudo.sh     # refaz todas as análises, cerca de 1h40 em 2 núcleos
```

Testado com Python 3.13.

## Scripts

Tabelas

- `constroi_tabela_keyrecs.py`: tabela da escrita, com o código das atividades, e a versão com o tempo de voo corrigido
- `monta_tabela_complex.py`: tabela da leitura, com 8.579 termos e 8 variáveis
- `valida_complex.py`: teste rápido da tabela da leitura
- `auditoria.py`: confere a base bruta da KeyRecs contra a tabela

Escrita

- `modelos_escrita.py`: mediana, regressão linear, árvore podada e Random Forest
- `lofo_keyrecs.py`: LOFO da Atividade 04
- `lofo_complementar.py`: testes complementares do LOFO
- `reconstroi_e_testa.py`: reconstrução independente da tabela e LOFO em três versões

Leitura

- `modelos_leitura.py`: regra das palavras raras, logística, árvore, EBM, Random Forest e XGBoost
- `lofo_complex.py`: LOFO com o mesmo protocolo da escrita

As duas bases

- `metodos_globais.py`: H de Friedman, decomposição funcional, modelos substitutos, lista de regras e MMD-critic
- `ajuste_holm.py`: ajuste de Holm dos valores p do LOFO
- `indicadores_extras.py`: intervalos por bootstrap e calibração
- `figuras_lofo.py` e `figuras_comparacao.py`: figuras
- `conferir_numeros.py`: confere os números do artigo

## Tabela do grupo e tabela corrigida

Nas Atividades 02 a 04, o tempo de voo da senha foi calculado pelo rótulo das colunas, que está trocado no
`fixed-text.csv`. Assim, ele somava à pausa real a diferença entre o tempo de pressionamento da segunda e o da
primeira tecla. A correlação com a pausa real é de 0,994. O artigo usa a tabela corrigida:

```bash
TABELA_ESCRITA=tabelas/tabela_analitica_corrigida.csv SUFIXO=_corrigida python modelos_escrita.py
```

As saídas com a tabela corrigida têm o sufixo `_corrigida`. As conclusões não mudam:

| | Tabela do grupo | Tabela corrigida |
|---|---|---|
| Correlação entre voo e hesitação | 0,738 | 0,725 |
| Regressão linear, RMSE e R² | 0,0310 s e 0,44 | 0,0314 s e 0,42 |
| Árvore podada | 2 cortes no voo, em 0,074 e 0,143 s | 1 corte, voo acima de 0,138 s |
| Random Forest, RMSE e R² | 0,0342 s e 0,34 | 0,0342 s e 0,34 |
| LOFO do tempo de voo | +10,41 ms | +10,36 ms |

## Validação

- Escrita: validação cruzada 5 × 10 com `RepeatedKFold` e semente 42.
- Leitura: validação cruzada 5 × 10 agrupada por palavra, para a mesma palavra nunca estar no treino e no teste.
- Comparações pareadas: teste t corrigido de Nadeau e Bengio, com razão teste/treino de 0,25.
- LOFO com várias variáveis testadas ao mesmo tempo: ajuste de Holm.
