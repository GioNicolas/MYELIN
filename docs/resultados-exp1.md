# Resultados — Experimento 1: a Curva de Mielinização no Qwen2.5-0.5B

**Data:** 20 de outubro de 2026  
**Modelo:** `Qwen/Qwen2.5-0.5B` (24 camadas), pesos congelados, CPU do Google Colab  
**Configuração:**
- camada rasa k = 17, lida pelo tradutor $P_{17}$ do Experimento 0b;
- portão: entropia, τ = 0,3;
- sinapse: γ = 0,999, η = 0,5.

**Script:** [`experiments/exp1_curva_mielina.py`](../experiments/exp1_curva_mielina.py) · **Notebook:** [`notebooks/exp1_curva_mielina.ipynb`](../notebooks/exp1_curva_mielina.ipynb) · **Wrapper:** [`src/myelin/wrapper.py`](../src/myelin/wrapper.py) (modo simulação)

---

## Rodada 1: chaves centradas pela média causal

Como ler as colunas:
- **saída cedo por bloco:** o texto dividido em 8 trechos consecutivos.
- **precisão:** fração das saídas cedo que concordam com a camada final.
- **rasa concorda:** top-1 da camada 17 igual ao top-1 final, saindo ou não.
- **ppl:** perplexidade das previsões usadas (a do modelo completo vem no título).

### `src/myelin/fast_synapse.py` (modelo completo: ppl 7,53)

| config | saída cedo por bloco | precisão | rasa concorda 1º → últ. | camadas | ppl |
|---|---|---|---|---|---|
| estático | 38 31 41 68 73 59 55 62 % | 75% | 39% → 58% | 20,3 | 19,06 |
| myelin r=16 | 98 96 94 92 88 66 80 81 % | **11%** | **2% → 21%** | 17,9 | **3486** |
| myelin r=16 aud=5% | 65 54 41 65 73 55 60 62 % | 61% | 24% → 50% | 19,9 | 22,82 |
| myelin r=64 | 98 100 97 98 95 93 88 84 % | **8%** | **2% → 13%** | 17,4 | **9136** |
| myelin r=64 aud=5% | 62 48 52 63 79 70 76 66 % | 62% | 36% → 48% | 19,5 | 21,47 |

### `tests/test_fast_synapse.py` (modelo completo: ppl 8,34)

| config | saída cedo por bloco | precisão | rasa concorda 1º → últ. | camadas | ppl |
|---|---|---|---|---|---|
| estático | 18 61 52 33 73 70 75 52 % | 70% | 12% → 49% | 20,2 | 17,15 |
| myelin r=16 | 98 82 60 49 72 73 88 55 % | 33% | 0% → 44% | 19,0 | 330 |
| myelin r=16 aud=5% | 34 52 49 30 67 72 84 48 % | 60% | 14% → 45% | 20,2 | 20,84 |
| myelin r=64 | 98 99 95 52 73 75 77 65 % | 22% | 0% → 38% | 18,4 | 1344 |
| myelin r=64 aud=5% | 46 64 61 34 75 81 84 61 % | 61% | 20% → 52% | 19,6 | 22,11 |

### `README.md` (modelo completo: ppl 19,59)

| config | saída cedo por bloco | precisão | rasa concorda 1º → últ. | camadas | ppl |
|---|---|---|---|---|---|
| estático | 32 27 33 32 38 27 30 25 % | 34% | 15% → 19% | 21,9 | 97,00 |
| myelin r=16 | 38 30 39 31 38 31 38 30 % | 42% | 16% → **30%** | 21,6 | **54,60** |
| myelin r=16 aud=5% | 35 27 37 31 42 27 38 27 % | 44% | 15% → 31% | 21,7 | 51,72 |
| myelin r=64 | 41 38 52 37 45 31 37 35 % | 42% | 17% → **36%** | 21,2 | 60,52 |
| myelin r=64 aud=5% | 34 41 48 34 41 30 37 34 % | 41% | 17% → 31% | 21,4 | 53,93 |

## Achados

1. **No código, o MYELIN sem auditoria colapsa logo no início.**
   - Já no 1º bloco, 98% dos tokens saem cedo, mas a camada rasa concorda com a final em só 0–2%. A perplexidade explode (330 a 9136, contra 17–19 do estático).
   - Mecanismo: as primeiras consolidações deixam a camada rasa confiante em **todos** os tokens, o estresse cai abaixo de τ, e a partir daí ninguém mais acorda o córtex. Como só o córtex ensina, o erro fica travado. É o "confiante e errado" do Experimento 0, agora **criado pela própria sinapse**.
2. **A auditoria de 5% evita o colapso, mas no código não supera o estático.**
   - Concordância final: 45–52% contra 49–58%.
   - Perplexidade: 21–23 contra 17–19.
3. **No README (português, onde o tradutor é fraco), o MYELIN ajuda.**
   - A concordância da camada rasa sobe de 19% (estático) para 30–36% no último bloco.
   - A perplexidade cai quase pela metade: 97 → 52–60.
   - A precisão das saídas sobe de 34% para 41–44%.
   - É o primeiro sinal a favor do H2. Uma leitura possível: a sinapse está corrigindo, ao vivo, a parte da tradução que o $P_{17}$ não aprendeu (português).
4. **Causa provável do colapso: chaves quase paralelas.**
   - O Qwen tem dimensões com ativações gigantes que aparecem só em alguns tokens.
   - Ao subtrair a média, **todo** token passa a desviar muito dela nessas dimensões, e todas as chaves apontam praticamente para o mesmo lado. Num teste sintético com uma dimensão gigante em 30% dos tokens, o |cos| médio entre chaves foi 0,27 com as chaves cruas, **1,00** centradas e 0,14 padronizadas.
   - Com chaves paralelas, cada consolidação vale para o texto inteiro: a camada rasa passa a repetir, com confiança, a resposta do último token que o córtex ensinou. No código, isso é frequente (quebras de linha, indentação).
   - Hipótese a confirmar na rodada 2, que mede o |cos| no Qwen real.

## Rodada 2 (a rodar)

**Chaves:** centradas × **padronizadas**. Cada dimensão é dividida pelo seu desvio causal, e as dimensões gigantes deixam de dominar.

**η:** 0,1 × 0,5.

**Demais parâmetros:** posto 64, com e sem auditoria de 5%.

**Novidade:** o script imprime o |cos| médio entre chaves de cada modo, no texto real.

**Perguntas:**
- com chaves padronizadas, o colapso some sem precisar de auditoria?
- no código, o MYELIN passa a superar o estático?
