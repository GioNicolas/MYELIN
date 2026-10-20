# Resultados — Experimento 2: a memória rápida aprende algo útil, antes do portão?

**Data:** 20 de outubro de 2026  
**Modelo:** `Qwen/Qwen2.5-0.5B`, pesos congelados, CPU do Colab  
**Configuração:**
- camada rasa k = 17, lida pelo tradutor $P_{17}$;
- córtex num **calendário fixo**, a cada 8 tokens (128 consultas em 1024 tokens), sem portão;
- sinapse: η = 0,5, γ = 1.

**Medição:** a camada rasa é medida em todos os tokens, **antes** de aprender com cada um. ΔNLL = NLL da camada rasa no próximo token real, na 2ª metade do texto, menos a do estático (negativo = ajudou).

**Script:** [`experiments/exp2_diagnostico.py`](../experiments/exp2_diagnostico.py) · **Notebook:** [`notebooks/exp2_diagnostico.ipynb`](../notebooks/exp2_diagnostico.ipynb) · Desenho e motivação: [resultados-exp1.md](resultados-exp1.md) (revisão externa)

---

## Resumo: ΔNLL médio por grupo de textos (nats)

| memória | código-dev (2) | código-en (1) | prosa-en (2) | pt (2) | todos | melhorou em |
|---|---|---|---|---|---|---|
| viés online (sem chave) | −0,08 | +0,14 | +0,11 | **−1,66** | −0,45 | 3/7 |
| r=64, chave centrada (rodada 1) | −0,28 | −0,03 | −0,03 | **−1,84** | −0,62 | **7/7** |
| r=64, chave fixa (entrada) | +4,02 | +4,84 | +3,68 | +1,14 | +3,22 | 0/7 |
| r=64, chave fixa (saída) | +4,55 | +4,88 | +4,05 | +1,33 | +3,53 | 0/7 |
| r=d, chave fixa (entrada) | +0,31 | +0,72 | +0,52 | −1,79 | −0,17 | 3/7 |
| r=d, chave fixa (saída) | +0,66 | +1,16 | +0,94 | −1,62 | +0,16 | 2/7 |
| kNN exato, chave fixa (entrada) | −0,18 | −0,12 | −0,12 | −0,79 | −0,33 | **7/7** |
| kNN exato, chave fixa (saída) | −0,21 | −0,06 | −0,15 | −0,91 | −0,37 | **7/7** |

## Perplexidade da camada rasa na 2ª metade: quanto da distância até o modelo completo o kNN fecha

| texto | estático | melhor kNN | modelo completo | distância fechada (em log) |
|---|---|---|---|---|
| `fast_synapse.py` | 65,2 | 48,2 (−26%) | 4,9 | 12% |
| `test_fast_synapse.py` | 21,7 | 19,3 (−11%) | 3,7 | 7% |
| `README.md` | 7733 | 2981 (−61%) | 17,0 | 16% |
| `guia-de-estudo-e-pitch.md` | 4786 | 2028 (−58%) | 16,3 | 15% |
| `torch/.../activation.py` | 6,9 | 6,2 (−10%) | 2,3 | 10% |
| WikiText: Robert Boulter | 17,7 | 14,5 (−18%) | 6,2 | 19% |
| WikiText: Du Fu | 53,7 | 47,3 (−12%) | 11,9 | 8% |

Interferência entre chaves, |c_tj|, mediana / 99% com r = 64:
- centrada: 0,12–0,13 / 0,61–0,73;
- fixa: 0,10–0,14 / 0,46–0,66.

## Achados

1. **Há sinal, mas é pequeno.**
   - A memória exata (kNN) melhora a camada rasa nos **7 de 7 textos**, inclusive em inglês e em código que o tradutor nunca viu.
   - Mas fecha só **7–19%** da distância até o modelo completo, e a concordância com a final quase não muda (+0 a 3 pontos).
   - A correção do córtex é previsível a partir da chave local, só que pouco.
2. **A memória linear com chaves descorrelacionadas atrapalha muito.**
   - Com r = 64 e chave fixa, a NLL piora de 1 a 5 nats; com posto completo, piora menos, mas ainda piora fora do português.
   - Diferença essencial para o kNN: este só corrige quando há uma chave guardada parecida (similaridade ≥ 0,5) e devolve **zero** para chaves novas. A memória linear sempre devolve alguma coisa: para um token novo, ela soma pedaços das correções de tokens diferentes (c ≈ 0,1 cada), e isso vira ruído.
   - Com 128 consultas e posto 64, a memória fica sobrecarregada.
3. **O colapso da rodada 1 não vinha das chaves.**
   - No calendário fixo, as mesmas chaves centradas da rodada 1 **ajudam nos 7 textos**. E no Qwen real elas não são quase paralelas (mediana |c| ≈ 0,12; o 1,00 era só do teste sintético).
   - A hipótese das chaves paralelas está **refutada** como causa principal. O colapso vem do **laço de seleção** (só aprende quando está em dúvida, e a sinapse faz parar de duvidar), como o Codex apontou.
   - Uma leitura provável de por que as chaves centradas funcionam: elas guardam uma componente comum que faz a memória aprender, em boa parte, um **viés global** útil.
4. **O ganho em português é, em grande parte, correção global do tradutor.**
   - O viés online, sem olhar o contexto, já tira 1,66 nats em português. Em inglês e código ele **piora** (+0,11 a +0,14).
   - O confundidor apontado pela revisão se confirma: o ganho do README na rodada 1 era principalmente o idioma que faltou no treino do $P_{17}$, e não aprendizado do contexto.

## Consequências

- **A hipótese central, na forma atual, é fraca neste cenário.** Mesmo o teto não paramétrico melhora pouco a camada 17 em textos em inglês e código. O contexto (atenção) do próprio modelo já captura a maior parte da repetição.
- **Se a memória continuar, ela deve ser do tipo kNN:** só corrigir quando reconhece a situação. Uma versão de tamanho fixo é viável, com um buffer circular das últimas M consultas (M = 256 dá ~1,8 MB em fp32), e respeita o "memória nunca cresce". Mas aproxima o MYELIN do *neural cache* (Grave et al., 2017).
- **Antes de decidir o rumo, falta medir o teto com mais professor.** Aqui só 1 em cada 8 tokens ensinou. Com o portão, 40–60% dos tokens acordam o córtex, então a memória teria muito mais exemplos.

## Rodada 2b: com mais professor (córtex a cada 2 tokens)

Mesma medição, com 512 consultas ao córtex em 1024 tokens (4× a rodada acima).

> **Atenção:** o `README.md` mudou entre as rodadas (ganhou links para os resultados). Por isso a perplexidade do estático nele passou de 7733 para 5016, e os números do README não se comparam entre rodadas. Os outros 6 textos são idênticos (o estático dá os mesmos valores). Daqui em diante, os textos de avaliação precisam ficar congelados numa cópia.

### ΔNLL médio por grupo (nats), córtex a cada 2 tokens

| memória | código-dev | código-en | prosa-en | pt | todos | melhorou em |
|---|---|---|---|---|---|---|
| viés online | −0,26 | −0,16 | −0,01 | −1,87 | −0,63 | 6/7 |
| r=64, chave centrada | −0,60 | −0,20 | −0,13 | −2,36 | −0,91 | 7/7 |
| kNN exato (saída fixa) | −0,65 | −0,27 | **−0,38** | −2,20 | −0,96 | **7/7** |
| **kNN 256 (tamanho fixo)** | −0,56 | −0,27 | −0,32 | −2,03 | −0,87 | **7/7** |
| viés + kNN 256 | −0,65 | −0,33 | +0,07 | **−2,66** | −0,97 | 5/7 |

As memórias novas a cada 8 tokens: kNN 256 −0,37 (7/7, igual ao kNN exato da rodada acima); viés + kNN 256 −0,35 (3/7, e **+1,14** num artigo do WikiText).

### Fração da distância até o modelo completo fechada (em log), a cada 2 tokens

| texto | kNN exato | kNN 256 | viés + kNN 256 | r=64 centrada |
|---|---|---|---|---|
| `fast_synapse.py` | 31% | 25% | 35% | 32% |
| `test_fast_synapse.py` | 29% | 26% | 22% | 20% |
| `README.md` | 36% | 34% | 44% | 42% |
| `guia-de-estudo-e-pitch.md` | 42% | 38% | 50% | 42% |
| `torch/.../activation.py` | 26% | 24% | 29% | 17% |
| WikiText: Robert Boulter | 37% | 32% | −6% | 12% |
| WikiText: Du Fu | 25% | 20% | −5% | 9% |

Concordância da camada rasa com a final, média dos blocos 5–8 (estático → kNN exato):

| texto | estático | kNN exato |
|---|---|---|
| `fast_synapse.py` | 54% | 58% |
| `test_fast_synapse.py` | 61% | 66% |
| `activation.py` | 71% | 77% |
| Robert Boulter | 70% | 74% |
| Du Fu | 51% | 54% |

## Achados da rodada 2b

1. **O teto cresce com mais professor.** Com 4× mais consultas, o kNN fecha **25–42%** da distância até o modelo completo (antes, 7–19%). O ganho fica em −0,3 a −0,4 nats também em inglês, fora do treino do tradutor.
2. **A memória de tamanho fixo perde pouco.** O kNN com só as 256 últimas consultas (~1,8 MB) fica perto do exato (−0,87 contra −0,96 nats), então o "memória nunca cresce" é viável.
3. **O viés com η = 0,5 é ruidoso.** Ele segue metade do último resíduo em vez de uma média. Ajuda muito onde há erro global (português, −2,66 combinado com o kNN), mas atrapalha onde o tradutor já está calibrado (inglês), e chega a +1,14 nats a cada 8 tokens. Um viés precisa de η bem menor (ex.: 0,02–0,05).
4. **Com mais professor, a memória linear (r = 64, centrada) chega perto do kNN** em código e português, mas fica bem atrás em prosa inglesa (fecha 9–12% contra 25–37%).
5. **A concordância sobe pouco:** +3 a +6 pontos na 2ª metade. A camada rasa melhora em probabilidade (NLL) mais do que no top-1.

## O que isso significa

- **A ideia tem sinal real e que escala com a supervisão**, mas o efeito absoluto é moderado. Mesmo no melhor caso, a camada 17 continua bem pior que a final (ex.: perplexidade 12,0 contra 6,2).
- **Ainda não é a resposta do H2.** A rodada 2b dá professor a 50% dos tokens de graça. A pergunta do H2 (qualidade × custo, com o portão decidindo) só sai do Experimento 3.
- **No enquadramento de decodificação especulativa,** o modelo completo verifica todos os tokens, então o professor é quase contínuo, como aqui. A concordância é a taxa de aceitação do rascunho, e o kNN a sobe em 3 a 6 pontos. Com rascunho na camada 17 de 24, porém, o ganho de velocidade seria pequeno; esse caminho precisaria de uma camada bem mais rasa ou de um modelo maior.

## Próximo: Experimento 3 (o H2 com o portão)

[`experiments/exp3_portao.py`](../experiments/exp3_portao.py) · [`notebooks/exp3_portao.ipynb`](../notebooks/exp3_portao.ipynb)

**Desenho:**
- portão de estresse de volta, com τ de 0,1 a 0,5, refazendo a simulação inteira para cada τ;
- Early-Exit Estático contra:
  - kNN 256, com e sem auditoria de 5%;
  - viés lento (η = 0,03) + kNN 256;
  - sinapse linear r = 64 centrada (γ = 0,999 por token);
- resultado: curvas custo × qualidade.

**Duas contas de custo:**
- camadas nominais;
- custo real, incluindo a cabeça (≈ 9 camadas no Qwen2.5-0.5B).

**Textos:** congelados em `data/eval/`.
