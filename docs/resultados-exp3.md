# Resultados — Experimento 3: o H2 com o portão (qualidade × custo)

**Data:** 21 de outubro de 2026  
**Modelo:** `Qwen/Qwen2.5-0.5B`, pesos congelados, GPU T4 do Colab (~9 min)  
**Configuração:**
- camada rasa k = 17, lida pelo tradutor $P_{17}$;
- portão de estresse de volta, com τ ∈ {0,1; 0,2; 0,3; 0,4; 0,5}; a simulação inteira é refeita para cada τ;
- memórias: kNN 256 (com e sem auditoria de 5%), viés lento (η = 0,03) + kNN 256, sinapse linear r = 64 com chave centrada (η = 0,5, γ = 0,999 por token);
- 7 textos congelados em `data/eval/`, 1024 tokens cada.

**Medição:** ΔNLL = NLL das previsões **usadas** (rasa onde saiu cedo, final onde o córtex acordou) menos a NLL do modelo completo, no próximo token real. Custo real = camadas + cabeça d × |V| (≈ 9,1 camadas no 0,5B), que a leitura rasa paga em todo token e o córtex paga de novo; 100% = modelo completo.

**Scripts:** [`experiments/exp3_portao.py`](../experiments/exp3_portao.py), [`experiments/exp3_analise.py`](../experiments/exp3_analise.py) · **Notebook:** [`notebooks/exp3_portao.ipynb`](../notebooks/exp3_portao.ipynb)

---

## Resumo (média dos 7 textos)

| config | τ | saída | camadas | custo real | ΔNLL | ΔNLL 2ª metade | precisão |
|---|---|---|---|---|---|---|---|
| estático | 0,1 | 22% | 22,5 | 117% | +0,16 | +0,17 | 86% |
| | 0,2 | 36% | 21,5 | 110% | +0,43 | +0,43 | 74% |
| | 0,3 | 48% | 20,6 | 104% | +0,85 | +0,77 | 65% |
| | 0,4 | 63% | 19,6 | 97% | +1,40 | +1,35 | 58% |
| | 0,5 | 78% | 18,6 | 90% | +2,05 | +1,92 | 52% |
| kNN 256 | 0,1 | 29% | 21,9 | 113% | +0,20 | +0,22 | 86% |
| | 0,2 | 45% | 20,9 | 106% | +0,45 | +0,46 | 76% |
| | 0,3 | 58% | 19,9 | 99% | +0,86 | +0,79 | 67% |
| | 0,4 | 72% | 19,0 | 93% | +1,37 | +1,30 | 59% |
| | 0,5 | 85% | 18,0 | 86% | +2,02 | +1,85 | 52% |
| kNN 256 + aud 5% | 0,1 | 28% | 22,0 | 114% | +0,19 | +0,21 | 85% |
| | 0,2 | 42% | 21,0 | 107% | +0,41 | +0,43 | 76% |
| | 0,3 | 55% | 20,1 | 101% | +0,80 | +0,73 | 67% |
| | 0,4 | 68% | 19,2 | 94% | +1,29 | +1,22 | 59% |
| | 0,5 | 81% | 18,3 | 88% | +1,88 | +1,70 | 52% |
| viés lento + kNN 256 + aud 5% | 0,1 | 28% | 22,0 | 114% | +0,19 | +0,22 | 85% |
| | 0,2 | 42% | 21,0 | 107% | +0,40 | +0,48 | 76% |
| | 0,3 | 56% | 20,0 | 100% | +0,74 | +0,82 | 65% |
| | 0,4 | 69% | 19,2 | 94% | +1,13 | +1,23 | 58% |
| | 0,5 | 82% | 18,3 | 88% | +1,65 | +1,66 | 50% |
| r=64 centrada + aud 5% | 0,1 | 26% | 22,2 | 115% | +0,16 | +0,16 | 88% |
| | 0,2 | 42% | 21,0 | 107% | +0,38 | +0,38 | 74% |
| | 0,3 | 57% | 20,0 | 100% | +0,77 | +0,73 | 64% |
| | 0,4 | 74% | 18,8 | 92% | +1,34 | +1,14 | 55% |
| | 0,5 | 88% | 17,9 | 85% | +1,96 | +1,69 | 49% |

**Como ler:** no mesmo τ, a memória faz mais tokens saírem cedo. Comparar linha a linha pelo τ engana; a comparação justa é **na mesma taxa de saída** (que fixa o custo, já que k é o mesmo).

## Ganho na mesma taxa de saída, por grupo

ΔNLL(memória) − ΔNLL(estático interpolado na mesma taxa de saída), média dos τ dentro da faixa do estático, por texto e depois por grupo. Negativo = memória melhor.

| grupo | kNN 256 | kNN 256 + aud | viés lento + kNN + aud | r=64 centrada + aud |
|---|---|---|---|---|
| pt (2) | **−0,63** | **−0,62** | **−0,95** | **−0,84** |
| código-dev (2) | −0,32 | −0,22 | −0,30 | −0,28 |
| código-en (1) | −0,09 | −0,03 | −0,01 | −0,08 |
| prosa-en (2) | −0,04 | −0,01 | +0,04 | +0,03 |

Por texto (kNN 256 sem auditoria): `fast_synapse.py` −0,41 · `test_fast_synapse.py` −0,23 · `README.md` −0,59 · `guia` −0,67 · `activation.py` −0,09 · Robert Boulter −0,08 · Du Fu −0,00.

Na média dos 7 textos, a mesma qualidade sai com **6 a 13 pontos percentuais a mais de saídas**. Exemplo com ΔNLL ≈ 0,8: estático 48%, r=64 57%, viés + kNN 56%.

Taxa de saída com τ = 0,3, total → 2ª metade:
- estático 48% → 52%;
- kNN 256 58% → 64%;
- kNN + aud 55% → 60%;
- viés + kNN + aud 56% → 67%;
- r=64 57% → 62%.

## Achados

1. **No sentido relativo, o H2 se confirma.** As quatro memórias empurram a curva custo × qualidade para o lado certo: menos NLL com a mesma taxa de saída, ou mais saídas com a mesma qualidade. Não há colapso: a auditoria e o kNN, que só corrige o que reconhece, resolveram o laço de seleção da rodada 1.
2. **O ganho está concentrado onde o tradutor erra de forma sistemática.**
   - **Português** (fora do treino do $P_{17}$): −0,6 a −0,95 nats. O viés lento é a melhor variante aqui, o que mostra que boa parte é correção global de idioma, como no Experimento 2.
   - **Código do próprio repositório** (repetitivo e fora da distribuição): −0,2 a −0,4.
   - **Inglês externo, prosa ou código: praticamente zero** (−0,09 a +0,04). O viés e a sinapse linear chegam a piorar a prosa inglesa. Onde o tradutor já está calibrado, o contexto do próprio modelo já cobre o que a memória aprenderia.
3. **No sentido absoluto, nenhum ponto de operação compensa neste modelo.**
   - Todo ponto com ΔNLL abaixo de ~1 nat custa **mais de 100%**: sai mais caro que rodar o modelo completo, porque a leitura rasa paga a cabeça (≈ 9 camadas) em todo token.
   - Mesmo o melhor ponto (τ = 0,1) perde ~0,16 nats (+17% de perplexidade), longe do que um early exit aceitável exige (< 0,05).
4. **Há um pouco de mielinização.** A taxa de saída sobe da 1ª para a 2ª metade em todas as configurações. Com memória sobe mais (+5 a +11 pontos, contra +4 do estático), mas a qualidade na 2ª metade fica parecida.
5. **A auditoria troca saídas por segurança.** Sem auditoria, o kNN sai mais (é mais agressivo) e, na média, tem o maior ganho na mesma taxa de saída. Com 5% de auditoria, o ganho cai um pouco em código e em inglês. Neste conjunto, sem colapso, ela não foi necessária. Mas é a proteção contra o colapso da rodada 1, então continua no desenho.

**Ressalvas:**
- A curva do estático é convexa. Por isso a interpolação linear exagera um pouco a vantagem da memória, sobretudo nos ganhos pequenos (~0,1) com baixa taxa de saída.
- Foi uma rodada só, com uma semente de auditoria, e 1 a 2 textos por grupo.

## O que isso significa

- **A ideia funciona como adaptação ao domínio em tempo de teste, não como aceleração geral.** A memória rápida conserta erros sistemáticos da leitura rasa (idioma novo, código repetitivo de um projeto). Em texto onde o modelo já é bom, não acrescenta nada.
- **No Qwen2.5-0.5B com k = 17, o early exit não paga a conta,** com ou sem MYELIN. O gargalo é estrutural: a cabeça custa ~9 camadas, e sair na camada 17 economiza no máximo 7.

## Rumos possíveis

1. **Modelo maior.** Num modelo de 7B, a cabeça vale ~3 de 28 camadas, e a economia de sair cedo passa a existir. O que precisa ser medido: se o tradutor e o ganho relativo se mantêm.
2. **Rascunho de decodificação especulativa.** A camada rasa propõe tokens e o modelo completo verifica. O professor fica quase contínuo, como na rodada 2b. A métrica passa a ser a taxa de aceitação; o Experimento 2b mediu +3 a +6 pontos de concordância com o kNN. Trabalhos a comparar: Online Speculative Decoding, Kangaroo, Draft & Verify, LayerSkip.
3. **Assumir o enquadramento de adaptação ao domínio** e medir onde ele brilha: textos longos em idioma ou domínio fora do treino do tradutor, contra baselines como o *neural cache* e a *dynamic evaluation*.
