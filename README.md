# Identificação de padrões de dificuldade em leitura e escrita via XAI

Trabalho da disciplina Aprendizado de Máquina Explicável, UFPR, 2026.
Allysson Souza, Débora Zanoello e Marcelo Carvalho. Professor: Anderson Ara.

O trabalho começou nas Atividades 02 a 04 com a KeyRecs, uma base de digitação com 96 participantes. Nesta etapa
entrou uma base nova, a CompLex, que é pública e bem maior, com 8.579 termos em inglês. As duas passaram pelo
mesmo protocolo: modelos interpretáveis contra caixa-preta, LOFO e outros métodos globais de explicabilidade.

- Escrita, KeyRecs: o ritmo de digitação numa senha fixa ajuda a prever quanto a pessoa hesita ao digitar um texto?
- Leitura, CompLex: quais termos de um texto o leitor tende a achar difíceis?

## Pastas

- `dados/`: cópia das duas bases, sem alteração
- `codigo/`: scripts, tabelas e resultados
- `artigo/`: arquivos do artigo para o Overleaf
- `atividades/`: código das Atividades 02, 03 e 04

## Bases

- KeyRecs: original no [Zenodo](https://doi.org/10.5281/zenodo.7886743), cópia em [`dados/keyrecs`](dados/keyrecs)
- CompLex: original em [MMU-TDMLab/CompLex](https://github.com/MMU-TDMLab/CompLex), cópia em [`dados/complex`](dados/complex)

## Como rodar

```bash
pip install -r codigo/requirements.txt
cd codigo
bash rodar_rapido.sh   # uns 10 minutos
bash rodar_tudo.sh     # refaz tudo, cerca de 1h40
```

O último passo, `conferir_numeros.py`, compara cada número do artigo com as saídas dos scripts.

## Observação sobre a KeyRecs

No `fixed-text.csv`, os rótulos das colunas de pares de teclas estão trocados. Nas atividades, o tempo de voo da
senha foi calculado pelo rótulo. No artigo, usamos a pausa real. As duas versões têm correlação de 0,994 e levam às
mesmas conclusões; a comparação está em [`codigo/README.md`](codigo/README.md).
