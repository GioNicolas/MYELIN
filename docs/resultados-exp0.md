# Resultados — Experimento 0: o quanto cada camada já sabe a resposta final?

**Data:** 19–20 de outubro de 2026  
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
3. **O efeito do contexto existe no código, não na prosa.** Nos arquivos de código, as camadas finais melhoram bastante na 2ª metade sem sinapse nenhuma (camada 22 em `test_fast_synapse.py`: 29% → 57%). No README, nada muda. Uma explicação provável (não testada) é que, em textos repetitivos, a atenção já copia padrões sozinha (*induction heads*). De todo modo, o MYELIN precisa ganhar **além** dessa linha de base.
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

## Experimento 0b: as camadas lidas pelo tradutor $P_k$ (Qwen2.5-0.5B)

**Tradutor:** *tuned lens* treinado por 500 passos (GPU T4), com corpus metade WikiText-2 (inglês) e metade código Python da biblioteca `transformers`. Depois de treinado, fica congelado.
**Medição:** os mesmos 3 textos e o mesmo τ = 0,3 do Experimento 0, rodando na CPU do Colab.
**Leitura:** `lm_head(norm(P_ℓ(h_ℓ)))`.
**JSON:** `exp0_Qwen2.5-0.5B_tunedlens.json`, salvo no Drive.

### Concordância com a camada final (1ª → 2ª metade), sem → com tradutor

| Camada (de 24) | `fast_synapse.py` sem | **com** | `test_fast_synapse.py` sem | **com** | `README.md` sem | **com** |
|---:|---|---|---|---|---|---|
| 4 | 0% → 0% | **13% → 25%** | 0% → 0% | **12% → 28%** | 0% → 0% | **4% → 4%** |
| 8 | 0% → 0% | 16% → 30% | 0% → 0% | 18% → 32% | 0% → 0% | 5% → 4% |
| 12 | 0% → 0% | 18% → 35% | 0% → 0% | 21% → 38% | 1% → 0% | 7% → 6% |
| 16 | 1% → 0% | 25% → 43% | 0% → 0% | 26% → 45% | 2% → 1% | 11% → 7% |
| **17** | — | **39% → 58%** | — | **34% → 61%** | — | **18% → 15%** |
| 18 | 5% → 4% | 40% → 58% | 3% → 4% | 34% → 60% | 1% → 2% | 18% → 16% |
| 20 | 16% → 19% | 44% → 65% | 9% → 19% | 37% → 65% | 6% → 5% | 22% → 18% |
| 21 | 29% → 33% | 50% → 74% | 20% → 35% | 47% → 78% | 15% → 9% | 30% → 28% |
| 22 | 44% → 57% | 58% → 79% | 29% → 57% | 52% → 81% | 22% → 21% | 38% → 39% |
| 23 | 64% → 77% | 68% → 84% | 57% → 80% | 67% → 89% | 51% → 52% | 57% → 58% |

(— = a camada 17 não foi transcrita na tabela do Experimento 0.)

### Calibração: precisão quando S ≤ τ (1ª → 2ª metade), com tradutor

| Camada | `fast_synapse.py` | `test_fast_synapse.py` | `README.md` | Sem tradutor (`fast_synapse.py`) |
|---:|---|---|---|---|
| 4 | 39% → 59% | 39% → 62% | 15% → 19% | 0% → 0% |
| 12 | 46% → 64% | 50% → 65% | 21% → 14% | 0% → 0% |
| 17 | 68% → 79% | 59% → 77% | 33% → 38% | — |
| 20 | 72% → 85% | 62% → 81% | 47% → 35% | 33% → 57% |
| 22 | 82% → 89% | 75% → 89% | 60% → 69% | 81% → 79% |
| 23 | 87% → 91% | 84% → 93% | 76% → 80% | 86% → 89% |

### KL médio $D_{\text{KL}}(p_{24} \parallel p_\ell)$ com tradutor: treino vs. avaliação

| Camada | 4 | 12 | 16 | 17 | 18 | 20 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|
| treino (WikiText + código, último log) | 2,98 | 2,27 | 1,76 | — | 1,07 | 0,81 | 0,35 | 0,18 |
| `fast_synapse.py` | 5,34 | 4,51 | 3,88 | 2,90 | 2,65 | 2,27 | 1,06 | 0,51 |
| `README.md` | 7,70 | 7,02 | 6,33 | 5,50 | 5,11 | 4,55 | 2,33 | 0,90 |

### Achados

1. **O tradutor funciona.**
   - As camadas rasas passam a falar a língua da final. A camada 4 vai de 0% para 12–28% de concordância no código, e a 18 vai de ~5% para 34–60%.
   - Elas deixam de ser confiantes e erradas: na camada 4, a precisão com S ≤ τ sobe de **0% para 39–62%**.
2. **Há um salto entre as camadas 16 e 17.**
   - O KL cai de 3,88 para 2,90 e a concordância sobe ~14 pontos de uma vez, nos dois arquivos de código.
   - A partir da 17, a concordância passa de 50% na 2ª metade dos textos de código.
   - A camada 17 é o candidato natural a k.
3. **O portão de estresse ainda não é confiável.**
   - A precisão com S ≤ τ só chega a 90% na camada 23 (2ª metade).
   - Na camada 17, os tokens confiantes acertam só 59–79%.
   - Um Early-Exit Estático com entropia e τ = 0,3 erraria de 1 a cada 5 a 2 a cada 5 saídas.
4. **O efeito do contexto aparece em todas as camadas, mas só no código.**
   - Sem sinapse nenhuma, a 2ª metade do código é bem melhor (camada 4: 13% → 25%; camada 17: 39% → 58%).
   - É a linha de base exata que o MYELIN precisa superar: Early-Exit Estático com o mesmo $P_k$.
   - No README, nada muda.
5. **O tradutor não generaliza para português.**
   - No README, a camada 17 concorda só 15–18% e o KL é 5,5. O corpus de treino não tinha português.
   - Mesmo no código, o KL de avaliação é ~2× o de treino (os arquivos do MYELIN têm comentários em português).

### Consequências para o projeto

- **k = 17 de 24** como ponto de partida (com k = 12 como comparação, onde sobra mais espaço para a sinapse aprender).
- **A economia realista é menor que a do README:** cada token que sai na camada 17 economiza 7 de 24 camadas (~29%), não "28 camadas". O número final depende de quantos tokens saem.
- **O portão precisa de calibração própria**, por exemplo:
  - τ escolhido por camada para uma precisão-alvo (curva precisão × cobertura);
  - margem entre os 2 maiores candidatos;
  - verificações aleatórias acordando o córtex mesmo com estresse baixo.
- **Retreinar o tradutor com texto em português**, ou avaliar em textos em inglês, antes de tirar conclusões sobre prosa.
- **Pergunta do H1/H2:** a sinapse rápida, em cima de $P_{17}$, sobe a concordância e a calibração da camada 17 **além** do que o contexto sozinho já dá?

## Próximo passo: a Curva de Mielinização no modelo real

`src/myelin/wrapper.py` em modo simulação (o modelo roda inteiro, e o portão só decide qual previsão contaria) e o notebook da Curva de Mielinização. Ele compara, no mesmo texto e com o mesmo $P_k$:

- Early-Exit Estático ($\Delta W_k = 0$);
- MYELIN (sinapse viva).

As métricas são a taxa de saída, a concordância com a final e a precisão das saídas ao longo do texto.
