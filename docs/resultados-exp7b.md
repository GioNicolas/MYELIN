# Resultados — Experimento 7b: o reflexo em texto com pouca repetição

**Data:** 24 de outubro de 2026  
**Modelo:** `Qwen/Qwen2.5-0.5B-Instruct` (24 camadas, cabeça ≈ 9,1, passo completo = 33,1 camadas), pesos congelados, GPU T4 do Kaggle (~16 min, mais o treino do tradutor)  
**Pergunta:** o reflexo ganhou da busca de n-gramas no [Experimento 7](resultados-exp7.md), mas num texto que era 77%–100% laço. Isso continua valendo em respostas de chat realistas?

**Desenho:** o mesmo ciclo exato da auto-especulação gulosa do Experimento 7 (custo corrigido, professor com um ciclo de atraso, ε = 0,02), agora sobre **respostas do modelo de instrução**:
- **24 pedidos de teste em 6 grupos**, no estilo do Spec-Bench ([`experiments/pedidos_chat.py`](../experiments/pedidos_chat.py)):
  - resumo (5) e tradução para o inglês (4) de um trecho de 448 tokens;
  - explicar código (3) e documentar código (3), também sobre um trecho;
  - conversa (5) e matemática (4), sem texto de apoio.
- **11 pedidos de validação**, com textos e perguntas diferentes, escolhem os hiperparâmetros de cada família. O Du Fu saiu da validação.
- Respostas de 119 a 512 tokens (mediana 350). O top-1 da passada completa coincidiu com o token gerado em 100% das posições.
- Novidades:
  - reflexo com a variante de 2 tokens iguais e a chave também na camada 12;
  - o **custo da busca** do reflexo entra na conta;
  - **correntes** em que cada elo usa o seu próprio γ.

**Escolhas da validação:**
- cabeças com γ = 1 (kNN com min_sim 0,5, cache neural com λ = 0,5);
- busca de n-gramas com n_max = 4 e γ = 6;
- reflexo na camada 24, τ = 0,6, γ = 6 e **1 token** (a variante de 2 tokens não foi escolhida).

O tradutor $P_k$ foi treinado de novo para o modelo de instrução, com o mesmo corpus amplo do [Experimento 4b](resultados-exp4b.md).

**Script:** [`experiments/exp7b_chat.py`](../experiments/exp7b_chat.py) · **Notebook:** [`notebooks/exp7b_chat.ipynb`](../notebooks/exp7b_chat.ipynb)

---

## Speedup simulado sobre a decodificação normal (teste, média das respostas)

| rascunhador | resumo | tradução | código-explicar | código-editar | conversa | matemática | **todos** | **total de tokens** |
|---|---|---|---|---|---|---|---|---|
| estático k=8 | 0,75× | 0,73× | 0,79× | 0,75× | 0,76× | 0,79× | 0,76× | 0,76× |
| kNN 1024 k=8 | 0,78× | 0,77× | 1,01× | 0,89× | 0,85× | 1,02× | 0,88× | 0,89× |
| cache neural justo k=8 | 0,77× | 0,76× | 1,01× | 0,89× | 0,83× | 1,01× | 0,86× | 0,88× |
| estático k=17 | 0,83× | 0,83× | 0,93× | 1,01× | 0,76× | 0,88× | 0,86× | 0,87× |
| kNN 1024 k=17 | 0,85× | 0,85× | 0,97× | 1,03× | 0,82× | 0,94× | 0,90× | 0,91× |
| cache neural justo k=17 | 0,77× | 0,76× | 0,92× | 1,00× | 0,76× | 0,91× | 0,84× | 0,85× |
| **busca de n-gramas (PLD)** | 1,54× | 1,27× | 1,98× | **4,52×** | 1,30× | 1,96× | **1,94×** | 1,73× |
| reflexo MYELIN | 1,52× | 1,23× | 2,07× | 2,81× | 1,31× | 1,95× | 1,73× | 1,68× |
| **reflexo → PLD** | **1,59×** | 1,27× | **2,10×** | 3,62× | **1,33×** | **2,01×** | 1,87× | **1,75×** |
| PLD → reflexo | 1,54× | 1,27× | 1,98× | 4,52× | 1,30× | 1,96× | 1,94× | 1,73× |
| PLD → kNN 1024 k=17 | 1,43× | 1,14× | 1,82× | 4,49× | 1,09× | 1,72× | 1,79× | 1,54× |
| reflexo → kNN 1024 k=17 | 1,27× | 1,03× | 1,77× | 2,51× | 1,04× | 1,62× | 1,46× | 1,39× |
| reflexo → PLD → kNN 1024 k=17 | 1,46× | 1,14× | 1,90× | 3,59× | 1,11× | 1,74× | 1,70× | 1,55× |
| controle: pares embaralhados (k=17) | 0,79× | 0,78× | 0,87× | 0,96× | 0,72× | 0,82× | 0,81× | 0,82× |
| controle: correção média (k=17) | 0,84× | 0,83× | 0,94× | 1,03× | 0,76× | 0,90× | 0,87× | 0,88× |
| controle: última correção (k=17) | 0,78× | 0,79× | 0,87× | 0,95× | 0,72× | 0,81× | 0,80× | 0,82× |

"Total de tokens" é o total de tokens dividido pelo custo total, então as respostas longas pesam mais. É o número que mais se parece com o ganho de tempo de um servidor.

**Repetição de 4-gramas nas respostas:** resumo 26%, tradução 15%, código-explicar 47%, **código-editar 93%**, conversa 15%, matemática 42%. Era 77%–100% no Experimento 7.

**Aceitação e frequência de proposta** (média das respostas):

| | aceitação | tokens por ciclo | propõe |
|---|---|---|---|
| busca de n-gramas | 24% | 2,11 | 67% |
| reflexo | 30% | 1,83 | 40% |
| reflexo → PLD | 23% | 2,03 | 67% |

---

## O que os números dizem

### 1. Sozinho, o reflexo não ganha mais da busca de n-gramas

Em texto realista, o reflexo faz 1,73× (1,68× no total de tokens), contra 1,94× (1,73×) da busca de n-gramas. Ganha em só 11 das 24 respostas. **O resultado principal do Experimento 7 não se sustenta fora do laço.**

A diferença está quase toda em **documentar código** (2,81× contra 4,52×). Ali a resposta copia o código do pedido quase literalmente. A busca de n-gramas acha o lugar exato pelo 4-grama: no numpy, aceita 91% dos tokens rascunhados. O reflexo exige só 1 token igual e escolhe a ocorrência pelo estado oculto, e nesse caso escolhe pior: aceita 40%. Fora desse grupo, os dois empatam: 1,56× para cada um no total de tokens.

### 2. O reflexo é mais preciso, mas propõe menos

Quando propõe, o reflexo acerta mais (30% de aceitação contra 24%), mas propõe em 40% dos ciclos, contra 67%. É o comportamento esperado de uma porta de familiaridade: ele só especula quando o estado oculto reconhece a situação.

### 3. Como complemento, o reflexo ajuda pouco, mas quase sempre

**Reflexo → PLD** consulta o reflexo primeiro e, se ele não reconhecer nada, a busca de n-gramas.
- Ganha da busca de n-gramas sozinha em **20 das 24 respostas**.
- As 4 derrotas são as 3 de documentar código e o Du Fu, que é um empate na terceira casa (1,276× contra 1,278×).
- Fora de documentar código, ganha em 20 de 21 respostas: 1,60× contra 1,56× no total de tokens (**+2,6%**).
- Em documentar código, perde muito: 3,62× contra 4,52×.
- No total dos 24 pedidos, 1,75× contra 1,73× (+1%).

É um ganho pequeno, mas consistente, e de graça: o estado oculto já foi calculado.

A ordem inversa (PLD → reflexo) é idêntica à busca de n-gramas, pelo mesmo motivo do Experimento 7. Com n_min = 1, a busca de n-gramas só se cala quando o token atual nunca apareceu antes, e então o reflexo também não tem vizinho.

### 4. A cabeça rasa com memória não acelera em texto realista

- **Nenhuma** configuração com a cabeça passa de 1× na média. A melhor é o kNN 1024 em k = 17, com 0,90×.
- A memória ainda ajuda o rascunho a concordar com o modelo completo:
  - em k = 8, a aceitação sobe de 17% para 35%, e o speedup de 0,76× para 0,88×;
  - mas não chega ao ponto de empate, que exige 52% de aceitação com γ = 1.
- **Os controles em k = 17 enfraquecem a memória:** a correção média já faz 0,87×, contra 0,90× do kNN. Quase todo o ganho da memória neste regime é uma correção global.
  - Os pares embaralhados (0,81×) e a última correção (0,80×) ficam abaixo do estático (0,86×).
  - No Experimento 7, em texto em laço, a diferença era enorme (1,21× contra 0,81×).
- **Pôr a cabeça no fim da corrente piora:** PLD → kNN faz 1,79×, contra 1,94× da busca de n-gramas sozinha. Quando nada de cópia funciona, pagar a cabeça custa mais do que o ganho.

### 5. O 0,5B continua sendo um caso difícil para a cabeça

A cabeça custa 9,1 camadas, 27% do passo completo. Com γ = 1, o rascunho em k = 8 só se paga com mais de 52% de aceitação. Em texto realista, nem a memória chega lá. Num modelo de 7B, a cabeça custa ~8% do passo, e a conta muda. Mas o [Experimento 5](resultados-exp5.md) mostrou que a concordância das camadas rasas não melhora com o tamanho do modelo.

---

## Conclusão

| pergunta | resposta |
|---|---|
| O reflexo ganha da busca de n-gramas em texto realista? | **Não sozinho** (1,68× contra 1,73× no total de tokens; 11/24 respostas) |
| Como complemento, na frente da busca de n-gramas? | **Sim, por pouco:** 20/24 respostas, +2,6% fora de documentar código |
| Onde ele atrapalha? | Cópia literal longa (documentar código): o 4-grama acha o lugar certo, o estado oculto não |
| A cabeça rasa com memória acelera? | **Não** (no máximo 0,90×), e o ganho dela aqui é quase todo uma correção global |
| Qual é a melhor coisa a fazer no 0,5B? | Copiar do contexto, de graça: busca de n-gramas, com o reflexo na frente quando não há um n-grama longo |

**Balanço honesto do MYELIN até aqui.** A ideia original era uma camada rasa que aprende, durante a leitura, com a camada final. Ela melhora de forma consistente a concordância das camadas rasas com o modelo completo, nos Experimentos 3 a 6. Mas, no 0,5B e com o custo certo, **isso não vira velocidade em texto realista**.

O que sobrevive é mais modesto e mais interessante: o estado oculto que o modelo já calculou serve para escolher **de onde copiar** o contexto. Ele é mais preciso que o n-grama de superfície quando não há um n-grama longo, e pior quando há.

## Próximo passo: Experimento 7c

1. **Um híbrido com princípio:** a busca de n-gramas quando acha um n-grama longo (n ≥ 3, que é cópia literal), senão o reflexo, senão a busca de n-gramas curta. É o que os dados acima sugerem. Mas como foi sugerido pelo próprio teste, precisa ser medido em **pedidos novos**, que ninguém viu.
2. **Pedidos novos:** um terceiro conjunto, escrito antes de rodar, com mais respostas por grupo para dar intervalo de confiança.
3. O tradutor deste modelo já existe (`tuned_lens_Qwen2.5-0.5B-Instruct_multi.pt`): a rodada leva ~20 min.
