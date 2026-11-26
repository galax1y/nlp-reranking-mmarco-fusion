# Reranqueamento de passagens em português com cross-encoders (mMARCO)

Trabalho final da disciplina de Processamento de Linguagem Natural (INF/UFRGS). Tópico:
**reranqueamento**.

O objetivo é reordenar passagens recuperadas pelo BM25 com modelos Transformer
(*cross-encoders*), avaliando no mMARCO em português, e responder:

1. Quanto um cross-encoder ajustado por nós, numa GPU gratuita, ganha sobre o BM25 em comparação
   com um reranqueador publicado?
2. Quanto o protocolo de avaliação muda as conclusões?
3. Um terceiro estágio (monoPTT5) e a fusão de pontuações melhoram o resultado, e a que custo?
4. Em que tipos de pergunta o reranqueamento ajuda mais?

## Principais resultados

Conjunto oficial *dev small* (6.980 perguntas), candidatas = top-100 do BM25 sobre a coleção
completa (8,8 milhões de passagens):

| Sistema | MRR@10 |
|---|---|
| BM25 | 0,153 |
| Cross-encoder ajustado por nós | 0,255 |
| Cross-encoder publicado | 0,285 |
| Nosso cross-encoder → monoPTT5 (top-20) | 0,285 |
| Cross-encoder publicado + monoPTT5, fusão RRF | **0,292** |

- O protocolo preliminar (perguntas de treino, corpus pequeno com distratores aleatórios)
  inflava o MRR@10 em 3,2 vezes.
- Nosso modelo alcança 77% do ganho do modelo publicado sobre o BM25.
- 43% das perguntas não têm passagem relevante no top-100 do BM25: o primeiro estágio é o
  principal gargalo.

Detalhes em [`article/main.pdf`](article/main.pdf) e [`phase2/RESULTS.md`](phase2/RESULTS.md).

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `notebook.ipynb` | Experimento preliminar (v1). Seu protocolo de avaliação tem problemas, descritos em `phase2/WEAKNESSES.md`; os números dele **não** são os resultados finais. |
| `phase2/v2_completo.ipynb` | Pipeline final: dados, ajuste fino do cross-encoder, avaliação, cascata com monoPTT5 e fusão. |
| `phase2/results/` | Resultados exportados pelo notebook (CSVs, figuras, versões das bibliotecas). |
| `phase2/analysis/analyze_v2.py` | Análises extras (teto do BM25, tipos de pergunta, vitórias/derrotas). |
| `phase2/RESULTS.md` | Análise detalhada dos resultados. |
| `phase2/WEAKNESSES.md` | Avaliação crítica do experimento preliminar, que motivou a v2. |
| `phase2/archive/` | Versões antigas da v2, divididas em três notebooks. |
| `article/` | Artigo no formato SBC (LaTeX) e PDF compilado. |
| `research/literature_review.xlsx` | Revisão da literatura. |

## Como reproduzir

1. Abrir `phase2/v2_completo.ipynb` no Google Colab com GPU T4 (plano gratuito).
2. **Ambiente de execução → Executar tudo.** Cada etapa salva um ponto de retomada no Google
   Drive; se a sessão cair, basta executar tudo de novo.
3. Os resultados ficam em `MyDrive/AssignmentPLN_v2/results/`.

Para recompilar o artigo: `pdflatex main`, `bibtex main` e `pdflatex main` duas vezes, dentro de
`article/`.

## Uso de IA generativa

Conforme declarado no artigo, usamos o Claude Code (Anthropic) na busca de literatura, na
geração de código, na análise dos resultados e na redação de uma primeira versão do texto. Todo
o conteúdo foi revisado pelos autores.
