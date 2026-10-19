# Resultados — Experimento 0: o quanto cada camada já sabe a resposta final?

**Data:** 19 de outubro de 2026  
**Modelo:** `Qwen/Qwen2.5-0.5B` (24 camadas), pesos congelados, CPU do Google Colab (~13 min)  
**Script:** [`experiments/exp0_profundidade_por_camada.py`](../experiments/exp0_profundidade_por_camada.py) · **Notebook:** [`notebooks/exp0_profundidade_por_camada.ipynb`](../notebooks/exp0_profundidade_por_camada.ipynb)  
**Leitura das camadas:** *logit lens*, ou seja, `lm_head(norm(h_ℓ))`: o estado da camada ℓ lido direto pela cabeça final, sem tradutor.  
**Textos (1024 tokens cada):** `src/myelin/fast_synapse.py` e `tests/test_fast_synapse.py` (código), `README.md` (português).  
**Portão:** estresse = entropia normalizada, τ = 0,3.

> Os números abaixo foram transcritos da saída do Colab. O JSON bruto se perdeu quando a sessão foi encerrada, e o Qwen2.5-1.5B não chegou a terminar. Os dois serão refeitos no Experimento 0b, que roda a mesma medição de novo como linha de base.

---

## Como ler as colunas

- **Concorda:** fração dos tokens em que a palavra mais provável da camada ℓ é a mesma da camada final.
- **S ≤ τ:** fração dos tokens que **sairiam cedo** nessa camada num Early-Exit Estático (estresse baixo).
- **Precisão se S ≤ τ:** entre os que sairiam cedo, quantos acertariam a resposta da camada final. Esse é o teste de **calibração** do portão: confiante e errado é o pior caso.
- **1ª → 2ª:** valor na 1ª metade do texto → valor na 2ª metade. A diferença mostra o efeito do contexto **sem sinapse nenhuma**.

## Tabela (camadas selecionadas)

| Camada | `fast_synapse.py` concorda | S ≤ τ | precisão | `test_fast_synapse.py` concorda | S ≤ τ | precisão | `README.md` concorda | S ≤ τ | precisão |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | 0% → 0% | 21% → 17% | 0% → 0% | 0% → 0% | 22% → 13% | 0% → 0% | 0% → 0% | 20% → 15% | 0% → 0% |
| **4** | **0% → 0%** | **15% → 14%** | **0% → 0%** | **0% → 0%** | **14% → 11%** | **0% → 0%** | **0% → 0%** | **13% → 13%** | **0% → 0%** |
| 8 | 0% → 0% | 16% → 12% | 1% → 0% | 0% → 0% | 14% → 10% | 0% → 0% | 0% → 0% | 14% → 13% | 0% → 0% |
| 12 | 0% → 0% | 12% → 10% | 0% → 0% | 0% → 0% | 13% → 10% | 0% → 0% | 1% → 0% | 13% → 10% | 0% → 0% |
| 16 | 1% → 0% | 8% → 8% | 0% → 0% | 0% → 0% | 9% → 9% | 0% → 0% | 2% → 1% | 10% → 7% | 6% → 3% |
| 18 | 5% → 4% | 21% → 10% | 9% → 12% | 3% → 4% | 15% → 8% | 8% → 0% | 1% → 2% | 20% → 15% | 5% → 5% |
| 20 | 16% → 19% | 25% → 22% | 33% → 57% | 9% → 19% | 20% → 17% | 23% → 59% | 6% → 5% | 21% → 15% | 15% → 14% |
| 21 | 29% → 33% | 32% → 37% | 59% → 64% | 20% → 35% | 29% → 35% | 44% → 71% | 15% → 9% | 23% → 20% | 30% → 29% |
| 22 | 44% → 57% | 34% → 49% | 81% → 79% | 29% → 57% | 29% → 48% | 58% → 79% | 22% → 21% | 23% → 19% | 54% → 54% |
| 23 | 64% → 77% | 56% → 76% | 86% → 89% | 57% → 80% | 53% → 79% | 77% → 88% | 51% → 52% | 41% → 38% | 74% → 75% |
| 24 (final) | 100% | 68% → 90% | 100% | 100% | 65% → 90% | 100% | 100% | 63% → 68% | 100% |

KL médio $D_{\text{KL}}(p_{24} \parallel p_4)$ na camada 4: **15,9 / 15,8 / 13,5** nats (distribuições completamente diferentes). Ele só cai abaixo de 3 a partir da camada 22.

---

## Achados

1. **Lidas direto pela cabeça final, as camadas 1–16 "não falam a língua" da camada final.** A concordância é de 0% e o KL fica em ~15. A camada 4 não está "quase certa, precisando de um ajuste"; ela está em outra representação. A concordância só passa de ~30% a partir da camada 21 de 24.
2. **As camadas rasas ficam confiantes e erradas.** Na camada 4, 11–15% dos tokens têm estresse baixo, com **0% de precisão**: um Early-Exit Estático com esse portão erraria todos. Hipótese ainda não verificada: as camadas iniciais "ecoam" o token de entrada com alta certeza. Mesmo a camada 23 tem só 74–89% de precisão quando confiante.
3. **O efeito do contexto existe no código, não na prosa.** Nos arquivos de código, as camadas finais melhoram bastante na 2ª metade sem sinapse nenhuma (camada 22 em `test_fast_synapse.py`: 29% → 57%). No README, nada muda. Em textos repetitivos, a atenção já copia padrões sozinha (*induction heads*), e o MYELIN precisa ganhar **além** dessa linha de base.
4. **τ = 0,3 não é trivial nem para a camada final:** ela só fica abaixo do limiar em 63–90% dos tokens.

## Consequências para o projeto

- **A especificação original (k = 4, cabeça final direto, entropia como estresse) não funciona como está.** A distância entre a camada 4 e a 24 é uma mudança inteira de representação, que uma sinapse de posto 16 não cobre. O toy da Fase 1 já mostrava esse limite de capacidade.
- **O tradutor $P_k$ passa a ser obrigatório** (o "projetor leve" que a especificação já previa, na forma do *tuned lens* de Belrose et al., 2023). Divisão de trabalho:

  | Peça | Quando aprende | O que faz |
  |---|---|---|
  | Tradutor $P_k$ | uma vez, offline, e depois fica congelado | traduz a representação da camada k para a da camada final, para qualquer texto |
  | Sinapse Rápida $\Delta W_k$ (MYELIN) | durante a conversa | aprende os atalhos do contexto atual, em cima da tradução |

  A contribuição inédita continua sendo a parte viva: o *tuned lens* sozinho é estático.
- **O k realista provavelmente fica no meio da rede**, não na camada 4. A economia por token será menor que as "28 camadas economizadas" do README; o número real sai do Experimento 0b.
- **O medidor de estresse precisa ser validado depois do tradutor.** Se a precisão com S ≤ τ continuar abaixo de ~90%, serão necessários outro medidor (margem, ou uma cabeça de confiança treinada) e verificações aleatórias acordando o córtex mesmo com estresse baixo.
- **O experimento do H1 precisa comparar contra o efeito do contexto:** um Early-Exit Estático com o mesmo $P_k$, no mesmo texto, sem sinapse.

## Confirmação no Qwen2.5-1.5B (28 camadas, logit lens, GPU T4)

O mesmo padrão aparece num modelo 3× maior:

- **Camadas 1–19:** concordância de 0–4%, com KL ~9–12.
- **Confiantes e erradas:** 15–30% dos tokens com S ≤ τ, com precisão de 0–10%.
- **Concordância acima de 50%:** só a partir da camada 26–27 de 28.
- **Calibração ainda pior:** mesmo na camada 25, a precisão com S ≤ τ fica em 24–70%.

| Camada (de 28) | `fast_synapse.py` concorda | precisão se S ≤ τ | `README.md` concorda | precisão se S ≤ τ |
|---:|---|---|---|---|
| 4 | 1% → 0% | 3% → 0% | 1% → 1% | 1% → 1% |
| 14 | 2% → 1% | 4% → 5% | 3% → 1% | 8% → 4% |
| 20 | 16% → 11% | 30% → 32% | 7% → 6% | 13% → 12% |
| 24 | 37% → 51% | 43% → 57% | 23% → 16% | 28% → 19% |
| 26 | 53% → 65% | 58% → 68% | 43% → 36% | 50% → 40% |
| 27 | 71% → 81% | 76% → 82% | 62% → 65% | 68% → 69% |

## Experimento 0b, parcial: treino do tradutor no 0.5B

O tradutor $P_k$ do Qwen2.5-0.5B foi treinado (500 passos, GPU T4). A medição com ele no notebook falhou sem mostrar o erro e será refeita. O KL no corpus de treino (WikiText-2 + código) caiu bastante:

| Camada (de 24) | 4 | 8 | 12 | 16 | 18 | 20 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|
| KL com tradutor (treino, último log) | 2,98 | 2,54 | 2,27 | 1,76 | 1,07 | 0,81 | 0,35 | 0,18 |
| KL sem tradutor (`fast_synapse.py`, Exp. 0) | 15,9 | 15,6 | 15,4 | 13,4 | 10,7 | 7,3 | 2,8 | 0,9 |

Leitura preliminar: com o tradutor, a camada 18 fica com KL parecido ao da camada 23 sem tradutor, um ganho de ~5 camadas. A camada 4 continua longe (KL ~3). Ressalvas:

- os KLs vêm de corpora diferentes (treino vs. textos de avaliação);
- o treino ainda não tinha convergido (o KL médio seguia caindo, 2,09 → 2,00 nos últimos logs);
- concordância e calibração só saem da medição com o tradutor.

## Próximo passo: Experimento 0b

Treinar $P_k$ para cada camada ([`experiments/exp0b_tuned_lens.py`](../experiments/exp0b_tuned_lens.py)) e repetir esta tabela com ele ([`notebooks/exp0b_tuned_lens.ipynb`](../notebooks/exp0b_tuned_lens.ipynb), GPU no Colab ou no Kaggle). As perguntas são:

- em que camada a concordância passa de ~50%;
- se a precisão com S ≤ τ fica acima de 90%.

As respostas definem o k e se o portão de estresse é confiável.
