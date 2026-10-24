# Fronteira e próximas hipóteses (24 de outubro de 2026)

Depois do [Experimento 7b](resultados-exp7b.md), a regra mudou: **checar a novidade antes de construir**. Este documento:
1. mapeia o que já existe perto do MYELIN;
2. lista o que só nós medimos;
3. propõe hipóteses, cada uma com o estado da novidade e um experimento barato que pode matá-la.

## 1. O mapa: o que já existe

### Copiar do contexto, guiado pelo próprio modelo: ocupado

| trabalho | o que faz | relação com o MYELIN |
|---|---|---|
| **PLD+** (Somasundaram et al., arXiv 2412.01447, dez./2024) | usa a atenção e os estados ocultos da inferência para escolher de onde copiar; sem treino | **é o nosso reflexo**, publicado antes |
| **AdaPLD** (Liu et al., arXiv 2606.05742, jun./2026) | n-grama quando acha; similaridade semântica quando o léxico falha; vários ramos | é o nosso "reflexo → PLD" na ordem inversa, com árvore |
| **SwitchSD** (Eisenstadt et al., arXiv 2609.20186, set./2026) | sondas treinadas nos estados internos detectam a intenção de copiar (AUC > 0,99); alterna entre EAGLE e cópia; até +15% sobre o EAGLE-3 | o "copiar só quando há intenção" que o reflexo fazia sem treino |
| PLD, REST, Cacheback, SuffixDecoding, NEST | busca de n-gramas, no contexto ou num corpus | linhas de base |

**Conclusão:** o reflexo foi uma reinvenção independente do PLD+. Bom sinal de direção, mas sem novidade.

### Camadas rasas como rascunho: maduro

LayerSkip, Draft & Verify, SWIFT, Kangaroo, CLaSp, KNN-SSD (EACL 2026, que usa kNN para escolher quais camadas pular) e LEAP (ICML 2026). E ainda um estudo de caracterização de cinco métodos de auto-especulação.

### Rascunhador que aprende online com o verificador: ativo, sempre com gradiente

| trabalho | como aprende |
|---|---|
| Online Speculative Decoding (Liu et al., ICML 2024) | destilação com gradiente |
| DVI (arXiv 2510.05421, 2025) | LoRA atualizada online por KL |
| OmniDraft (arXiv 2507.02659) | cache de n-gramas + destilação híbrida, no dispositivo |
| EvoSpec (arXiv 2605.27390, 2026) | vocabulário e parâmetros do rascunhador evoluem em tempo real |
| When Drafts Evolve (ICML 2026); Baseten "live draft training" | aprendizado online / treino ao vivo |
| Osprey (arXiv 2609.09338, set./2026) | pré-treino do rascunhador para resistir à **mudança de domínio** |

O problema é reconhecido: o EAGLE aceita 70–80% em chat, mas cai para 40–50% em domínios especializados (tradução, código de domínio), e o Osprey existe por causa disso. **Não achei** uma adaptação **sem gradiente e não paramétrica**, uma memória no estado oculto do rascunhador ensinada pelo verificador. O OmniDraft tem um cache de n-gramas, mas a adaptação do modelo dele é por gradiente.

### "Quais tokens são difíceis": medido

"Think Deep, Not Just Long" (arXiv 2602.13517, ICML 2026) mede os *deep-thinking tokens*: aqueles cuja previsão interna muda nas camadas fundas. É praticamente a nossa análise por camada com o tradutor, aplicada a raciocínio.

## 2. O que só nós medimos

1. **Adaptar durante a leitura ≈ retreinar offline** (Exp 4b): uma leitura rasa, treinada estreita e corrigida online sem gradiente, recupera quase todo o ganho de retreiná-la com um corpus amplo.
2. **O ganho da memória cresce com a distância de domínio** (Exp 4): fora do domínio do treino, quase o dobro de saídas cedo com a mesma qualidade.
3. **Em texto realista, a correção que importa é quase toda global** (Exp 7b, controles): a correção média já dá quase todo o ganho do kNN. Em texto repetitivo, o reconhecimento por chave domina (Exp 7).
4. **O gargalo de custo:** com γ = 1, um rascunho só se paga com aceitação acima do seu custo relativo. No 0,5B, a cabeça custa 27% do passo e mata a auto-especulação. Um rascunhador externo pequeno custa bem menos: o 0,5B custa ~7% de um passo do 7B (~17% do 3B).

A ligação que ninguém parece ter feito: **os rascunhadores treinados (EAGLE, modelos pequenos) são exatamente "tradutores estreitos"**, treinados num domínio (ShareGPT, inglês) e usados em todos. O que no Exp 4b parecia um defeito (a memória "só" compensava o tradutor estreito) vira a aplicação.

## 3. Hipóteses

### H1: o MYELIN como adaptador sem gradiente para rascunhadores treinados ⭐

**Hipótese.** Uma memória de tamanho fixo no estado oculto de um rascunhador treinado, ensinada pelo token do verificador e sem retropropagação, recupera boa parte da aceitação perdida fora do domínio. E faz isso a um custo desprezível perto do rascunhador.

- **Novidade:** plausível, ainda não confirmada.
  - A adaptação online existe, mas sempre com gradiente: OSD, DVI, EvoSpec, OmniDraft.
  - O diferencial seria não precisar de otimizador, de memória de gradientes nem de cópia dos pesos por usuário: só um buffer de 1024 pares por sessão.
  - **Risco:** ser visto como um "kNN-LM no rascunhador", uma combinação de peças conhecidas.
- **Por que pode funcionar:** os achados 1, 2 e 4.
- **Por que pode falhar:** o achado 3. Em texto realista, o ganho do Exp 7b foi pequeno, mas lá o rascunho era uma camada rasa fraca, e não um rascunhador treinado.
- **Experimento que mata (Exp 8, ~1 rodada no Kaggle):**
  - rascunhador = Qwen2.5-0.5B-Instruct; verificador = Qwen2.5-3B-Instruct (cabe numa T4) ou 7B (duas T4);
  - os 24 pedidos do 7b, em 6 grupos, mais pedidos em português e de código de domínio;
  - aceitação e speedup simulado, com o ciclo exato:
    - sem memória;
    - com a memória kNN de correções;
    - com o cache neural justo (rótulo do verificador);
    - com o viés lento (correção global);
    - com os controles;
    - e com o PLD na frente.
- **Morre se:** a memória sobe a aceitação em menos de ~2 pontos fora do domínio, ou o viés lento sozinho faz o mesmo (aí seria só calibração).
- **Seria forte se:** +5 a +10 pontos de aceitação fora do domínio, sem perder dentro dele. Depois disso, o passo 2 é um cabeçote EAGLE de verdade.

### H2: calibração por sessão (a versão mínima da H1)

**Hipótese.** Se o que importa é quase todo global (achado 3), um único vetor de viés por sessão, aprendido online com o verificador, já adapta o rascunhador. São 896 números, sem busca nenhuma.

- **Novidade:** baixa como ideia (é uma calibração), mas útil como controle: ela diz se a H1 vale pelo reconhecimento ou só pela calibração.
- **Teste:** dentro do Exp 8, de graça.

### H3: o reflexo como portão de cópia sem treino

**Hipótese.** A similaridade do reflexo (precisa, mas com pouca cobertura) serve como sinal sem treino de "intenção de copiar", alternando entre o rascunhador neural e a busca de n-gramas.

- **Novidade:** baixa. O SwitchSD faz isso com sondas treinadas e mais precisão; o nosso seria a versão sem treino.
- **Teste:** também dentro do Exp 8 (rascunhador neural + PLD + o portão do reflexo).

### H4: γ pela profundidade de decisão

**Hipótese.** Prever, pelo tradutor, quando os próximos tokens serão "fundos" e encurtar o rascunho ali.

- **Novidade:** baixa. O γ adaptativo tem vários trabalhos, e os deep-thinking tokens já medem a profundidade.
- **Fica de reserva.**

## 4. Recomendação

**Exp 8 = H1, com H2 e H3 como controles dentro da mesma rodada.**
- Reaproveita quase tudo: o simulador exato do ciclo, a memória, o cache neural, os controles, os pedidos e o protocolo de validação e teste.
- A peça nova é um `DraftModelDrafter`: o rascunhador externo, em teacher forcing, com o custo dele na conta.

Antes de construir:
1. **Revisão adversarial da H1** por modelos externos (Codex, Antigravity): é inédita? O teste mata de verdade?
2. Uma busca mais funda de novidade em "training-free drafter adaptation" e "kNN draft model".
