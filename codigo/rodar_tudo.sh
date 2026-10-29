#!/usr/bin/env bash
# Refaz todas as análises, na ordem. Rode de dentro da pasta codigo/:  bash rodar_tudo.sh
# Tempo total: cerca de 1h40 em 2 núcleos. Os dados brutos ficam em ../dados/.
set -e

# ---------- 1. Tabelas ----------
python constroi_tabela_keyrecs.py            # escrita: refaz a tabela do grupo e gera a corrigida (~1 min)
python monta_tabela_complex.py               # leitura: monta a tabela da CompLex (~1 min)
python valida_complex.py                     # confere a tabela da leitura

# ---------- 2. Escrita com a tabela do grupo, a das Atividades 02 a 04 ----------
python modelos_escrita.py                    # reproduz a tabela de desempenho da ATV03     -> resultados/saidas_modelos_escrita/
python lofo_keyrecs.py                       # LOFO original da ATV04 (~6 min)               -> resultados/saidas/
python lofo_complementar.py                  # testes complementares (~10 min)               -> resultados/saidas_comp/
python auditoria.py                          # confere a base bruta contra a tabela (texto na tela)
python reconstroi_e_testa.py                 # reconstrução independente + LOFO em 3 versões -> resultados/saidas_comp/

# ---------- 3. Leitura (CompLex) ----------
python modelos_leitura.py                    # 6 modelos (~7 min)                            -> resultados/saidas_leitura/
python lofo_complex.py                       # LOFO completo (~25 min)                       -> resultados/saidas_complex/

# ---------- 4. Artigo: escrita com o tempo de voo corrigido + métodos globais ----------
export TABELA_ESCRITA=tabelas/tabela_analitica_corrigida.csv SUFIXO=_corrigida
python modelos_escrita.py                    # Tabela II                                     -> resultados/saidas_modelos_escrita_corrigida/
python lofo_keyrecs.py                       # Tabela IV e Fig. 2                            -> resultados/saidas_corrigida/
python lofo_complementar.py                  # Tabela V                                      -> resultados/saidas_comp_corrigida/
python metodos_globais.py                    # H, decomposição, substitutos, regras, MMD     -> resultados/saidas_globais_corrigida/
python ajuste_holm.py                        # p ajustado por Holm no LOFO                   -> resultados/saidas_holm/
unset TABELA_ESCRITA SUFIXO

# ---------- 5. Figuras e conferência do artigo ----------
SUFIXO=_corrigida python figuras_lofo.py         # figuras do LOFO na escrita     -> resultados/figuras_corrigida/
SUFIXO=_corrigida python figuras_comparacao.py   # escrita x leitura              -> resultados/figuras_corrigida/
python conferir_numeros.py ../artigo/main.tex   # confere se cada número do artigo bate com as saídas
