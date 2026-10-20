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
