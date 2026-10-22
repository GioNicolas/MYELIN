# Resultados — Experimento 7: o ciclo real, linhas de base justas e o reflexo

**Data:** 22 de outubro de 2026  
**Modelo:** `Qwen/Qwen2.5-0.5B` (24 camadas, cabeça ≈ 9,1 camadas, passo completo = 33,1 camadas), pesos congelados, GPU T4 do Kaggle (~21 min)  
**Responde a:** a revisão externa do [Experimento 6](resultados-exp6.md) (seção "Correção").

**Desenho:**
- **Texto gerado pelo próprio modelo.** Para cada documento, os primeiros 256 tokens viram o prompt e o modelo gera 768 tokens gulosos (`repetition_penalty = 1`). O ciclo da auto-especulação gulosa é então simulado de forma **exata** sobre essa continuação: o top-1 da passada completa coincidiu com o token gerado em 100% das posições (`greedy_mismatch = 0`).
- **O ciclo:** o rascunhador propõe até γ tokens; o verificador aceita o maior prefixo igual ao seu top-1 e emite mais um token (a correção, ou o bônus). Só depois a memória aprende, com um ciclo de atraso.
- **Custo do ciclo**, em passos do modelo completo: custo_rascunho/(L+H) + 1 + ε·(tokens rascunhados), com ε = 0,02 para todos. É a conta corrigida do Experimento 6: a verificação custa um passo inteiro.
- **Hiperparâmetros escolhidos na validação** (7 textos de `data/eval/`), separadamente para cada família, e medidos no **teste** (15 textos de `data/eval_dominio/`, 3 por grupo).

**Rascunhadores:**
- **cabeça na camada k** (k = 8 e 17), que paga γ·(k+H) camadas por ciclo:
  - estático (só o tradutor amplo $P_k$);
  - kNN 1024 (a memória de correções do estado oculto, ensinada pelo verificador);
  - **cache neural justo**: rótulo do verificador, porta de familiaridade, capacidade 1024 e λ escolhido na validação;
  - três **controles** da memória: pares chave → correção embaralhados, correção média e última correção.
- **busca de n-gramas** (Prompt Lookup Decoding): se o último n-grama já apareceu no contexto, propõe o que veio depois. Não roda camada nenhuma.
- **reflexo MYELIN** (novo): também não roda camada nenhuma.
  - A chave é o estado oculto da camada l na posição c−1, já calculado na verificação anterior, então de graça.
  - Procura, entre as últimas 1024 posições, a mais parecida (cosseno ≥ τ) **que levou ao mesmo token que acabou de sair**.
  - Propõe o que veio depois dela. Sem vizinho familiar, não especula.
- **PLD → reflexo**: o reflexo só quando a busca de n-gramas não acha nada.

**Escolhas da validação:**
- todas as cabeças com γ = 1;
- kNN com min_sim 0,5; cache neural com λ = 0,5;
- busca de n-gramas com n_max = 4 e γ = 6;
- reflexo na camada 24, com τ = 0,6 e γ = 6.

**Script:** [`experiments/exp7_reflexo.py`](../experiments/exp7_reflexo.py) · [`src/myelin/speculative.py`](../src/myelin/speculative.py) (seção "Experimento 7") · [`src/myelin/controls.py`](../src/myelin/controls.py) · **Notebook:** [`notebooks/exp7_reflexo.ipynb`](../notebooks/exp7_reflexo.ipynb)

---

## Speedup simulado sobre a decodificação normal (teste, média dos textos; > 1 = mais rápido)

| rascunhador | pt | idiomas | código-outro | python | prosa-en | **todos** | aceitação | tok/ciclo | propõe |
|---|---|---|---|---|---|---|---|---|---|
| estático k=8 | 0,71× | 0,76× | 0,76× | 0,71× | 0,87× | **0,76×** | 17% | 1,17 | 100% |
| kNN 1024 k=8 | 1,19× | 1,23× | 1,22× | 1,24× | 1,19× | **1,21×** | 87% | 1,87 | 100% |
| cache neural justo k=8 | 1,19× | 1,22× | 1,22× | 1,24× | 1,20× | **1,21×** | 87% | 1,87 | 100% |
| estático k=17 | 0,97× | 0,97× | 1,02× | 1,05× | 1,03× | **1,01×** | 82% | 1,82 | 100% |
| kNN 1024 k=17 | 1,04× | 1,06× | 1,09× | 1,08× | 1,06× | **1,06×** | 93% | 1,92 | 100% |
| cache neural justo k=17 | 1,02× | 1,06× | 1,08× | 1,07× | 1,04× | **1,06×** | 91% | 1,91 | 100% |
| busca de n-gramas (PLD) | 4,02× | 4,13× | 5,32× | 4,11× | 3,84× | **4,29×** | 74% | 4,72 | 89% |
| **reflexo MYELIN** | 4,12× | 4,79× | 5,39× | 4,81× | 4,02× | **4,62×** | 82% | 5,09 | 84% |
| PLD → reflexo | 4,02× | 4,13× | 5,32× | 4,11× | 3,84× | 4,29× | 74% | 4,72 | 89% |
| controle: pares embaralhados (k=8) | 0,75× | 0,79× | 0,76× | 0,86× | 0,78× | 0,79× | 21% | 1,21 | 100% |
| controle: correção média (k=8) | 0,76× | 0,82× | 0,78× | 0,87× | 0,81× | 0,81× | 24% | 1,24 | 100% |
| controle: última correção (k=8) | 0,67× | 0,70× | 0,69× | 0,66× | 0,71× | 0,68× | 5% | 1,05 | 100% |

**Repetição de 4-gramas no texto gerado:** pt 88%, idiomas 94%, código-outro 95%, python 92%, prosa-en 85%. Por texto, vai de 77% (Du Fu) a 100% (option.rs).

### Por texto: busca de n-gramas → reflexo

| texto | repetição | PLD | reflexo |
|---|---|---|---|
| Dom Casmurro | 81% | 3,03× | 3,26× |
| Braz Cubas | 87% | 3,74× | 3,73× |
| especificação (MYELIN) | 96% | 5,30× | 5,36× |
| Quijote (es) | 97% | 4,14× | **5,32×** |
| Verwandlung (de) | 92% | 4,22× | 4,49× |
| Tour du monde (fr) | 92% | 4,03× | 4,56× |
| option.rs | 100% | 6,18× | 6,23× |
| strings.go | 93% | 4,81× | 4,93× |
| response.js | 93% | 4,98× | 5,01× |
| numpy numeric.py | 98% | 5,23× | 5,15× |
| jinja2 lexer.py | 80% | 3,34× | 3,74× |
| networkx weighted.py | 98% | 3,76× | **5,54×** |
| wikitext: Du Fu | 77% | 3,13× | 3,23× |
| wikitext: Kiss You | 86% | 3,69× | 4,13× |
| wikitext: Ise battleship | 92% | 4,71× | 4,69× |

---

## O que os números dizem

### 1. O texto gerado é quase todo laço, e isso domina tudo

O 0,5B base com geração gulosa entra em laço: 77% a 100% dos 4-gramas do texto gerado já tinham aparecido antes (no option.rs, ele repete o mesmo exemplo de doc-teste sem parar). Num texto assim, **copiar do contexto é quase sempre certo**. Por isso a busca de n-gramas, que não roda camada nenhuma, faz 4,29×, e qualquer rascunho pela cabeça (≤ 1,21×) fica muito atrás.

Os números absolutos de busca de n-gramas e do reflexo **não valem para uso real**: um modelo de instrução, ou amostragem, repete muito menos. O que vale aqui são as **comparações no mesmo texto**.

### 2. O reflexo ganha da busca de n-gramas: 4,62× contra 4,29× (+8%)

- Ganha em todos os 5 grupos e em 12 dos 15 textos. Nos outros três, perde por 0,01× a 0,08×.
- Ganha também na validação (3,77× contra 3,51×), um conjunto de textos diferente.
- Aceita mais (82% contra 74%) e emite mais tokens por ciclo (5,09 contra 4,72), mesmo propondo um pouco menos vezes (84% contra 89%).
- Os ganhos grandes estão onde o laço não é exato: Quijote (+1,18×) e networkx (+1,78×).

**Por quê:** os dois copiam uma continuação do passado. A busca de n-gramas escolhe **de onde copiar** pelo n-grama de superfície mais longo. O reflexo exige só que o token atual coincida e escolhe a ocorrência pelo **estado oculto da última camada**, que resume o contexto inteiro. Quando o mesmo token aparece em vários lugares, o estado oculto escolhe melhor qual deles vai se repetir.

Em termos honestos, o reflexo é uma busca no contexto **guiada pelo estado oculto**, em vez de pela superfície do texto. É barato e não vê o futuro:
- os estados até c−1 já foram calculados na verificação anterior;
- a continuação proposta vai só até o token c, o último emitido.

O custo de comparar o estado com 1024 vizinhos (um produto de 1024 × 896) foi contado como zero; é desprezível perto de uma passada do modelo, mas não é nulo.

### 3. "PLD → reflexo" é idêntico à busca de n-gramas

Não é um defeito da conta. Com n_min = 1, a busca de n-gramas só se cala quando o token atual nunca apareceu antes, e então o reflexo também não tem vizinho com o mesmo token. A ordem certa é a inversa, **reflexo → PLD**, que fica para o próximo experimento.

### 4. Pela primeira vez, um rascunho pela camada rasa acelera

Com o custo corrigido, o kNN 1024 em k = 8 faz **1,21×**, acima de 1 em 15 de 15 textos (o pior faz 1,13×). O estático na mesma camada faz 0,76×: a memória sobe a aceitação de 17% para 87%.

Duas ressalvas:
- **O cache neural justo empata** (1,21×; em 3 textos, números idênticos). No texto em laço, os dois fazem a mesma coisa: reconhecer a situação e repetir o token que o verificador deu. A vantagem de corrigir o estado oculto, em vez de misturar uma distribuição de tokens, **continua não demonstrada**.
- O ganho vem do laço. Em texto com pouca repetição, a aceitação vai cair.

### 5. Os controles confirmam o mecanismo da memória

| k = 8 | speedup | aceitação |
|---|---|---|
| estático | 0,76× | 17% |
| pares embaralhados | 0,79× | 21% |
| correção média | 0,81× | 24% |
| última correção | 0,68× | 5% |
| **kNN 1024** | **1,21×** | **87%** |

As mesmas correções guardadas, mas devolvidas para a chave errada, não ajudam. Uma correção global (a média) também não, e repetir a última correção atrapalha. **O ganho vem de reconhecer a situação pela chave**, não de um ajuste global ou de continuidade local. Era o ponto principal que a revisão pedia para provar.

### 6. Camada 8 contra camada 17

Na camada 17, o rascunho já concorda muito com o modelo completo (82% no estático), mas custa 79% de um passo. Por isso nem 93% de aceitação passa de 1,06×. A camada 8 custa 52% de um passo e, com a memória, chega perto da mesma aceitação: é ali que a memória rende.

---

## Conclusão

| pergunta | resposta |
|---|---|
| A memória rápida faz uma camada rasa acelerar, com o custo certo? | **Sim, 1,21× em k = 8**, em texto com muita repetição |
| É por reconhecer a situação? | **Sim**: os três controles ficam em 0,68×–0,81× |
| A memória de correções ganha do cache neural justo? | **Não**: empate |
| Alguma coisa ganha da busca de n-gramas? | **Sim, o reflexo: 4,62× contra 4,29×**, em 12/15 textos e na validação |
| Os números absolutos valem para uso real? | **Não**: o texto gerado é 77%–100% laço |

A melhor ideia do MYELIN até aqui é **o reflexo**: usar o estado oculto que o modelo já calculou para escolher de onde copiar, em vez de rodar uma cabeça de rascunho. Ele não treina nada, não roda camada nenhuma e ganha da linha de base mais forte deste tipo, mas só foi testado num regime em que copiar é fácil.

## Próximo passo: Experimento 7b, texto com pouca repetição

1. **Texto realista:** `Qwen2.5-0.5B-Instruct` (ou 1,5B) com pedidos de chat (resumir, traduzir, explicar código, responder perguntas e conversa livre), no estilo do Spec-Bench, com geração gulosa. O tradutor $P_k$ é treinado de novo para o modelo de instrução.
2. **Combinações:** reflexo → PLD, e reflexo → kNN k = 8 (a cabeça só quando o reflexo se cala).
3. **Variantes do reflexo:**
   - a chave numa camada intermediária;
   - exigir 2 tokens iguais;
   - ponderar a similaridade pela posição.
4. **Medir o custo** da busca do reflexo e, se o resultado se mantiver, uma implementação real com cache KV e tempo de relógio (Experimento 8).
