# Resultados — Experimento 5: a conta de custo fecha num modelo maior?

**Data:** 21 de outubro de 2026  
**Modelo:** `Qwen/Qwen2.5-1.5B` (28 camadas, d = 1536), pesos congelados, GPU T4 do Kaggle  
**Configuração:**
- tradutor $P_k$ treinado como o do 0,5B: 500 passos, metade WikiText-2, metade Python;
- concordância por camada medida em 5 textos de `data/eval/`;
- curvas do Experimento 3 (os mesmos 7 textos, as mesmas 5 configurações) com k = 20 (a mesma proporção do k = 17 de 24) e k = 16.

**Notebook:** [`notebooks/exp5_modelo_maior.ipynb`](../notebooks/exp5_modelo_maior.ipynb) · Análise: [`experiments/exp3_analise.py`](../experiments/exp3_analise.py)

---

## 1. A cabeça pesa menos, como previsto

| | Qwen2.5-0.5B | Qwen2.5-1.5B |
|---|---|---|
| cabeça d × \|V\| em camadas | 9,1 de 24 (38%) | **5,0 de 28 (18%)** |
| k testado | 17 | 20 / 16 |
| saída cedo mínima para empatar com o modelo completo | ~57% | ~38% (k = 20) / ~28% (k = 16) |

## 2. Mas a qualidade no ponto de empate continua ruim

Early-Exit Estático (média dos 7 textos):

| modelo, k | τ | saída | custo real | ΔNLL |
|---|---|---|---|---|
| 0,5B, k = 17 | 0,1 | 22% | 117% | +0,16 |
| | ~0,37 | ~57% | 100% | ~+1,2 |
| 1,5B, k = 20 | 0,1 | 23% | 106% | +0,24 |
| | 0,2 | 38% | **100%** | **+0,58** |
| | 0,3 | 53% | 94% | +1,11 |
| | 0,5 | 85% | 82% | +2,44 |
| 1,5B, k = 16 | 0,1 | 14% | 108% | +0,19 |
| | 0,2 | 28% | **101%** | **+0,56** |
| | 0,5 | 78% | 75% | +2,94 |

O ponto de empate melhora de ~1,2 para ~0,6 nats de perda. Mas 0,6 nats ainda quer dizer perplexidade 1,8× maior. Nenhum ponto de qualidade aceitável (ΔNLL < 0,1) fica mais barato que o modelo completo.

**O teto, mesmo de graça:** se a cabeça não custasse nada, sair na camada 20 de 28 economizaria 8/28 = 29% em cada token que sai. Com τ = 0,1 (23% de saídas, já com +0,24 nats), a economia total seria **~7%**.

## 3. Concordância por camada: proporcional à profundidade

Camada 20 do 1,5B (1ª → 2ª metade do texto):

| texto | concordância |
|---|---|
| `fast_synapse.py` | 36% → 44% |
| `README.md` | 18% → 20% |
| `activation.py` | 59% → 64% |
| Robert Boulter | 57% → 59% |
| Du Fu | 52% → 54% |

É o mesmo patamar da camada 17 de 24 no 0,5B (39–61%). Num modelo congelado, **a camada a ~70% da profundidade concorda ~50% com a final, independentemente do tamanho**.

A concordância salta entre as camadas 19 e 20 em todos os textos (ex.: Robert Boulter, de 50% para 57%, com o KL caindo de 2,28 para 1,56). As camadas finais (24–27) passam de 70–90%.

## 4. A memória se comporta como no 0,5B

Ganho na mesma taxa de saída (nats, negativo = melhor):

| grupo | k = 20: kNN 256 | k = 20: viés + kNN | k = 16: kNN 256 | k = 16: viés + kNN |
|---|---|---|---|---|
| pt | −0,59 | **−0,99** | −0,56 | **−0,91** |
| código-dev | −0,27 | −0,31 | −0,27 | −0,26 |
| código-en | +0,01 | +0,20 | +0,07 | +0,28 |
| prosa-en | −0,02 | +0,10 | +0,16 | +0,29 |

- O ganho fora do domínio se repete: português −0,9 a −1,0, código do projeto −0,3.
- Em inglês, a memória não ajuda. Na camada mais rasa (k = 16) ela **atrapalha** (+0,16 a +0,29), principalmente com o viés.

## Conclusão

**O rumo "modelo maior" responde a pergunta do custo, e a resposta é não**, pelo menos para early exit num modelo congelado:
- O modelo maior cortou o peso da cabeça pela metade, como previsto.
- Mas a camada que concorda ~50% com a final continua a ~70% da profundidade. Então a economia máxima possível é pequena (~7% com perda já grande), mesmo com a cabeça de graça.
- O gargalo deixou de ser a cabeça e passou a ser o próprio early exit sem treino: as camadas rasas de um modelo congelado não sabem a resposta com antecedência suficiente. É coerente com a literatura: CALM e LayerSkip **treinam** o modelo para as saídas cedo.

O que sobrevive é o resultado do Experimento 4, agora confirmado em dois tamanhos: a memória rápida adapta a leitura rasa a domínios fora do tradutor.

## Próximos passos possíveis

1. **Tradutor multilíngue** (o confundidor do Experimento 4): retreinar o $P_{17}$ do 0,5B com vários idiomas e linguagens e rodar o Experimento 4 de novo.
2. **Reenquadrar como decodificação especulativa:** a camada rasa propõe o rascunho e o modelo completo verifica.
   - A qualidade fica **idêntica** à do modelo completo por construção, e o professor fica disponível em todo token.
   - A pergunta passa a ser só a taxa de aceitação, que a memória sobe fora do domínio.
   - Dá para medir com o código atual: concordância com professor a cada token, e o speedup estimado pela fórmula padrão.
