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
   - **Refutada pelo Experimento 2:** no calendário fixo, as mesmas chaves centradas ajudam nos 7 textos, e no Qwen real elas não são quase paralelas. O colapso vem do laço de seleção do portão.

## Revisão externa da rodada 1

Duas revisões independentes (Antigravity e Codex) receberam o mesmo prompt, com as equações, os resultados e 7 perguntas. Os pontos abaixo foram conferidos no código antes de serem aceitos.

### Procedem (confirmados no código)

| Ponto | Onde | O que muda |
|---|---|---|
| **γ só decaía quando havia consolidação.** Se o córtex parava de acordar, a sinapse parava de esquecer, e a homeostase (H3) ficava desligada justamente no colapso. | `fast_synapse.py`: `B.mul_(γ)` dentro de `consolidate` | `FastSynapse.decay()` e opção `decay_per_token` na simulação |
| **A padronização causal degenera nos primeiros tokens.** No token 1, com 2 amostras, toda dimensão vira ±1. | `causal_standardize` | Novo modo de chave `fixed`: média e desvio medidos num corpus de calibração separado (`fit_key_stats`), sem o 1º token de cada texto (*attention sink*) |
| **A escala do alvo dependia da própria sinapse** (`rms(q)`), um laço de realimentação desnecessário. | `simulate` | Alvo na escala fixa `rms(P_k(h_k))` (padrão); `target_scale="q"` reproduz a rodada 1 |
| **O cosseno médio (em módulo) antes da projeção não mede a interferência relevante.** O que importa é c_tj = z_t·z_j/‖z_t‖² depois de A, com sinal, e seus extremos. | `key_similarity` | `key_interference`: quantis de \|c_tj\| com e sem a projeção A |
| **A projeção aleatória de posto 64 guarda só ~7% da energia da chave**, e chaves ortogonais deixam de ser ortogonais no subespaço. Com d = 896, a matriz completa ocupa 3,2 MB. | `FastSynapse` | Testar posto completo (r = d), que o código já aceita |
| **O ganho no README pode ser só correção do idioma**, que faltou no treino do tradutor. Além disso, a saída cedo no README **não subiu** ao longo do texto (o H1 falhou lá). | resultados | Ablação "viés online" (a mesma chave para todo token) e textos fora do treino do tradutor (WikiText-2 teste, código do PyTorch) |
| **τ fixo não é o mesmo orçamento.** O H2 pede comparar curvas qualidade × custo, refazendo a simulação para cada τ (B muda com τ). | desenho experimental | Varredura de τ na próxima rodada com portão |
| **"Precisão" é fidelidade ao professor**, não acerto do próximo token real. | nomenclatura | Reportar também a NLL no próximo token real (`shallow_nll`) |
| **Economia:** no Qwen2.5-0.5B a cabeça (896 × 151.936 ≈ 136 M de pesos) custa mais que as 7 camadas (~15 M cada) que uma saída em k = 17 pula. E o KV cache que falta nas camadas profundas não está contabilizado. | viabilidade | Docstring do wrapper corrigida (dizia que a simulação era "exata"); decisão de rumo adiada até sabermos se a memória aprende algo útil |
| **3 textos de 1024 tokens é pouco**, e 2 deles são do próprio projeto. | desenho experimental | Experimento 2 já usa 7 textos em 4 grupos; avaliação final com dezenas de documentos e bootstrap pareado |
| **O colapso é estrutural:** só aprender quando o próprio aluno está em dúvida enviesa a amostra, mesmo com chaves boas. | desenho | Separar as perguntas: primeiro um calendário fixo de consultas ao córtex (Experimento 2), depois o portão |
| **A frase "η = 1 copia a distribuição final" exige γ = 1 e ε → 0.** | docstring | Corrigida |
| **Atribuir a melhora da 2ª metade a *induction heads* não foi testado.** | resultados do Exp. 0 | Reescrito como hipótese |

### Não procedem, ou só em parte

- **"O alvo é sempre confiante":** copiar a distribuição do professor também copia a incerteza dele. O problema real é a interferência, que espalha uma correção confiante para tokens com outra resposta.
- **"KL de treino menor que o de avaliação indica sobreajuste por falta de *weight decay*":** a diferença é explicada principalmente pela mudança de domínio (português nos comentários e no README).
- **"PCA com 64 direções captura mais de 80% da variância":** plausível, mas não medido. Fica como variante a testar, não como fato.
- **Decidir a saída pelo estresse sem a sinapse:** as duas revisões notam que isso impede a sinapse de aumentar a cobertura e mata o H1. Fica só como controle experimental.

### Novidade

As duas revisões convergem: a regra de memória (delta rule / *fast weights*) não é nova, e o MYELIN fica na interseção entre:
- *neural cache* / kNN-LM;
- *dynamic evaluation* / TTT;
- *early exit* (CALM, LayerSkip).

O que pode ser novo é a **autodestilação online e seletiva** do professor profundo para uma leitura intermediária, usando o sinal que o córtex já calculou quando o portão reprovou a saída rasa.

Uma sugestão de rumo (Antigravity) é usar a camada rasa + MYELIN como **rascunho em decodificação especulativa**: o modelo completo verifica todos os tokens, o que resolve ao mesmo tempo o KV cache, o travamento e a qualidade. Antes de seguir por aí, é preciso verificar trabalhos próximos que já atualizam o rascunho online, como *Online Speculative Decoding* (Liu et al., 2023), e que usam saída cedo como rascunho, como *Kangaroo* (2024) e *Draft & Verify* (Zhang et al., 2023).

## Experimento 2: a memória aprende algo útil antes do portão? (resultados em [resultados-exp2.md](resultados-exp2.md))

[`experiments/exp2_diagnostico.py`](../experiments/exp2_diagnostico.py) · [`notebooks/exp2_diagnostico.ipynb`](../notebooks/exp2_diagnostico.ipynb)

**Calendário:** o córtex acorda a cada 8 tokens, igual para todas as memórias, sem portão.

**Medição:** a camada 17 é medida em todos os tokens, antes de aprender com cada um. O resultado principal é o ΔNLL da camada rasa (no próximo token real) na 2ª metade de cada texto, em relação ao estático.

| Memória | Pergunta que responde |
|---|---|
| viés online (sem chave) | o ganho é só correção global do tradutor? |
| r = 64, chave centrada | referência da rodada 1 |
| r = 64, chave fixa (entrada ou saída da camada 17) | a chave padronizada resolve a interferência? A saída da camada, que já inclui a atenção dela, é melhor chave? |
| r = d (posto completo) | a compressão de posto baixo estava atrapalhando? |
| kNN exato | teto: a correção do córtex é previsível a partir da chave? |

**Textos (nenhum usado no treino do tradutor):**
- os 2 arquivos de código do projeto;
- 2 textos em português (README e guia);
- 1 arquivo de código do PyTorch;
- 2 artigos do split de teste do WikiText-2.

**Leitura dos resultados:**
- Se nem o kNN melhorar sobre o estático, a correção não é previsível pela chave local, e a ideia precisa mudar (de chave ou de pergunta).
- Se o kNN e o posto completo melhorarem e o r = 64 não, o culpado é a compressão.
- Se o viés empatar com todos, o ganho era só correção do tradutor.
