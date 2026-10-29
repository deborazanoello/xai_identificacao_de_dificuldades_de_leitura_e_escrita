# Dados

Cópia das duas bases públicas usadas no trabalho, sem nenhuma alteração, para tudo rodar sem baixar nada.
As duas têm licença de atribuição; ao usar os dados, cite a fonte original.

## KeyRecs, escrita

- Pasta: `keyrecs/`, com `demographics.csv`, `fixed-text.csv` (senha fixa) e `free-text.csv` (texto livre)
- Original: Zenodo, doi [10.5281/zenodo.7886743](https://doi.org/10.5281/zenodo.7886743), versão 1.0.0
- Artigo: Dias, Vitorino, Maia, Sousa e Praça, KeyRecs: A keystroke dynamics and typing pattern recognition
  dataset, Data in Brief 50, 2023, doi [10.1016/j.dib.2023.109509](https://doi.org/10.1016/j.dib.2023.109509)
- Licença: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

O MD5 dos três arquivos bate com o publicado no Zenodo:

```
da9be3c6c007c4b4a43008351b095e10  demographics.csv
f3c2ef3a42625f4df7183ea36d4543db  fixed-text.csv
a5ca6fcb0970cfdcd8eb958b3fe9f22a  free-text.csv
```

No `fixed-text.csv`, os rótulos das colunas de pares de teclas estão trocados: a coluna rotulada `DU.x.y` é a pausa
real, de soltar a primeira tecla até apertar a segunda. O `free-text.csv` está correto. A verificação está em
`codigo/auditoria.py`.

## CompLex, leitura

- Pasta: `complex/`, com `train/lcp_single_train.tsv` (7.662 termos), `test-labels/lcp_single_test.tsv` (917) e o
  `Readme.md` original
- Original: GitHub, [MMU-TDMLab/CompLex](https://github.com/MMU-TDMLab/CompLex)
- Artigo: Shardlow, Cooper e Zampieri, CompLex: A New Corpus for Lexical Complexity Prediction from Likert Scale
  Data, READI 2020, [ACL Anthology](https://aclanthology.org/2020.readi-1.9/)
- Licença: CC BY, conforme o `Readme.md` original

A tabela da leitura, com as 8 variáveis, é montada por `codigo/monta_tabela_complex.py`.
