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

"How Do LLMs Use Their Depth?" (Gupta et al., arXiv 2510.18871) mostra o mecanismo "chuta e refina". As camadas rasas propõem tokens frequentes como palpite estatístico, e as fundas corrigem com o contexto. O fator é a frequência **estática** do token, não a repetição dentro do contexto.

### Outros vizinhos que limitam as hipóteses abaixo

| trabalho | o que faz | o que deixa aberto |
|---|---|---|
| **SuffixDecoding** (Oliaro et al., NeurIPS 2025) | árvore de sufixos com as gerações passadas de **todos os pedidos**; ótimo em agentes | só cópia literal: uma paráfrase ou um nome trocado quebra o casamento |
| **Cascatas especulativas** (Narasimhan et al., ICLR 2025, Google) | regra token a token entre o modelo pequeno e o grande | a regra é fixa; o sistema **não aprende** com as intervenções do modelo grande |
| **GRACE / WISE** (edição de modelos) | dicionário chave→valor numa camada, indexado pelo estado oculto, para corrigir fatos | uso em edição pontual, não em adaptação contínua guiada por um professor |
| **Titans** (Google, 2025), TTT | memória que aprende durante o uso, **com gradiente**, dentro da arquitetura | exige treinar o modelo com a memória; não serve para um modelo pronto e congelado |
| **FLy** (Li et al., ICLR 2026) | afrouxa a verificação para aceitar rascunhos "semanticamente certos"; sem treino; 1,62× sobre o EAGLE-3 fora do domínio | é **com perda** (≥ 99% da qualidade), e não adapta o rascunhador |

## 2. O que só nós medimos

1. **Adaptar durante a leitura ≈ retreinar offline** (Exp 4b): uma leitura rasa, treinada estreita e corrigida online sem gradiente, recupera quase todo o ganho de retreiná-la com um corpus amplo.
2. **O ganho da memória cresce com a distância de domínio** (Exp 4): fora do domínio do treino, quase o dobro de saídas cedo com a mesma qualidade.
3. **Em texto realista, a correção que importa é quase toda global** (Exp 7b, controles): a correção média já dá quase todo o ganho do kNN. Em texto repetitivo, o reconhecimento por chave domina (Exp 7).
4. **O gargalo de custo:** com γ = 1, um rascunho só se paga com aceitação acima do seu custo relativo. No 0,5B, a cabeça custa 27% do passo e mata a auto-especulação. Um rascunhador externo pequeno custa bem menos: o 0,5B custa ~7% de um passo do 7B (~17% do 3B).

A ligação que ninguém parece ter feito: **os rascunhadores treinados (EAGLE, modelos pequenos) são exatamente "tradutores estreitos"**, treinados num domínio (ShareGPT, inglês) e usados em todos. O que no Exp 4b parecia um defeito (a memória "só" compensava o tradutor estreito) vira a aplicação.

## 3. A tese e o portfólio ambicioso

**Princípio: ambição na pergunta, rigor no teste.** Cada hipótese afirma a coisa mais forte que os nossos dados permitem imaginar, e vem com o experimento mais barato capaz de matá-la. Várias correm em paralelo; os revisores tentam derrubar todas; as que sobreviverem são construídas.

### A tese do MYELIN

> **Um modelo de linguagem pode formar hábitos durante o uso.** Ele pode transformar computação funda e cara em atalhos rasos e baratos, sem gradiente, usando a própria saída final como professor.

É a metáfora original da mielina, agora em escala de sistema. As quatro grandes hipóteses são faces dessa tese:

| | face | pergunta |
|---|---|---|
| **M1** | velocidade | um modelo servido fica mais barato quanto mais é usado? |
| **M2** | qualidade | um modelo pequeno aprende com o grande enquanto conversa e precisa dele cada vez menos? |
| **M3** | geometria | o que separa as camadas rasas da resposta final é pequeno o bastante para aprender em poucos tokens? |
| **M4** | ciência | o modelo congelado já forma hábitos sozinho, dentro do contexto? |

M3 e M4 são baratas (uma rodada cada, com o tradutor e o código que já existem) e decidem o desenho de M1 e M2. Por isso vêm primeiro.

---

### M1 🚀 Inferência que mieliniza: fica mais barata com o uso, sem treino e sem perda

**Afirmação.** Numa sequência de pedidos do mesmo usuário ou domínio, o custo por token de um sistema especulativo **cai continuamente com o número de pedidos já atendidos**, sem treino e sem mudar a saída. Os componentes são:
- uma memória do rascunhador, ensinada pelo verificador e **persistente entre pedidos**;
- o reflexo e a busca de n-gramas como camada de cópia.

E cai mais do que com a cópia literal entre pedidos (SuffixDecoding), porque a memória reconhece **paráfrases**, e não só sequências idênticas.

- **Por que dá para sonhar:**
  - a memória sem gradiente ≈ retreino offline (achado 1);
  - o ganho cresce fora do domínio (achado 2);
  - o rascunhador externo é barato (achado 4).
- **Vizinhos:**
  - o OSD também melhora com o uso, mas com gradiente;
  - o SuffixDecoding melhora com o uso, mas só com cópia literal.
- **O diferencial de sistema:**
  - um buffer por usuário custa kilobytes, e uma LoRA por usuário custa uma cópia de pesos e um otimizador;
  - o buffer passa a valer **no ciclo seguinte**;
  - **apagar a memória apaga a adaptação**, o que dá privacidade e "desaprendizado" por construção.
- **Teste que mata (Exp 8b):**
  - montagem: ~150 pedidos sobre o mesmo material (um código-fonte, um manual em português); rascunhador Qwen2.5-0,5B-Instruct, verificador 3B;
  - curva da aceitação e do speedup contra o índice do pedido, para quatro sistemas: estático, memória persistente, cópia entre pedidos (estilo SuffixDecoding) e os dois juntos;
  - mais um controle: os mesmos pedidos **reescritos com nomes e frases trocados**.
- **Morre se:**
  - a curva da memória for plana (inclinação dentro do ruído); ou
  - a cópia entre pedidos fizer o mesmo, inclusive nos pedidos reescritos.
- **Seria forte se:** a aceitação subir 5 a 10 pontos ao longo da sequência, ainda subindo no fim, com memória de tamanho fixo, e se a vantagem sobre a cópia literal **crescer** nos pedidos reescritos.

### M2 🚀 Um modelo pequeno que aprende com o grande enquanto conversa

**Afirmação.** Numa cascata em que o modelo pequeno gera e o grande só intervém nos tokens incertos, guardar cada intervenção como par (estado oculto do pequeno → token do grande) faz a **taxa de intervenção cair ao longo da sessão, com a qualidade parada no nível do modelo grande**. A cascata deixa de ser uma regra fixa e passa a aprender.

- **Por que dá para sonhar:** é o achado 1 trocando "a camada final ensina a rasa" por "o modelo grande ensina o pequeno". O ganho do achado 2 é maior justamente onde o pequeno é fraco.
- **Vizinhos:**
  - cascatas especulativas e Co-LLM: a regra de adiar é fixa;
  - GRACE: dicionário de correções, mas para edição pontual;
  - destilação online: com gradiente.
- **Teste que mata:**
  - fluxo de ~200 tarefas com resposta checável (aritmética de vários passos, tradução para o inglês com chrF);
  - 0,5B gera; o 3B intervém quando a entropia do 0,5B passa de um limiar;
  - comparar, **com a mesma qualidade**, a fração de tokens que precisam do 3B, com e sem memória, ao longo do fluxo.
- **Morre se:**
  - a memória não reduzir as intervenções em pelo menos 20% (relativo) no fim do fluxo, com a mesma qualidade; ou
  - as correções memorizadas piorarem a qualidade, por transferirem para o lugar errado.
- **Seria forte se:** a curva de intervenções cair de forma sustentada (por exemplo, pela metade em 200 tarefas). Isso seria "um 0,5B que vira 3B no seu domínio, de graça".

### M3 🔬 O abismo entre as camadas é um deslocamento de domínio de baixo posto

**Afirmação.** Depois do tradutor $P_k$, o que ainda separa a previsão da camada rasa da resposta final é, em grande parte, um **deslocamento de baixo posto e específico do domínio**: poucos vetores, diferentes por domínio, que se aprendem online em dezenas de tokens.
- Isso explicaria por que a saída cedo, os tradutores e os rascunhadores caem fora do domínio (Osprey, o EAGLE em 40–50%).
- E explicaria por que a correção média quase empatou com o kNN no Exp 7b.

- **Vizinhos:**
  - o tuned lens é uma correção afim **global e offline**;
  - "How Do LLMs Use Their Depth?" descreve o "chuta e refina", sem decompor o resíduo por domínio;
  - vetores de direção (*steering*) são escritos à mão, para comportamento.
- **Teste que mata (barato, ~1 rodada):**
  1. com o tradutor que já existe, coletar o resíduo (estado final − $P_k(h_k)$) em 6+ domínios (os grupos do 7b, mais português e código);
  2. medir o espectro (quanto da energia cabe nos primeiros r vetores), os ângulos entre os subespaços de domínios diferentes e quanto um ajuste online de posto r, em N tokens, recupera do ganho do kNN.
- **Morre se:**
  - o espectro for plano (r = 8 explica menos de metade); ou
  - os subespaços forem iguais entre domínios (aí é só um defeito do tradutor, corrigível offline); ou
  - o posto baixo online recuperar menos de 70% do ganho do kNN.
- **Seria forte se:** r ≤ 8, aprendido em ≤ 64 tokens, recuperar ≥ 80% do ganho em todos os domínios e nos dois tamanhos de modelo. Isso seria uma afirmação geral sobre a geometria dos transformers, com um método de adaptação de 8 × 896 números.

### M4 🔬 Automatização em contexto: o modelo congelado obedece à lei da prática?

**Afirmação.** Quando o modelo volta a escrever algo que já está no contexto, a **profundidade de decisão** cai: é a primeira camada a partir da qual a previsão do tradutor já é a final e não muda mais. E cai mais a cada repetição, numa **curva de prática**, como a lei de potência da prática na psicologia humana: cada repetição economiza menos que a anterior, mas economiza.

- Se for verdade, o modelo já "mieliniza" sozinho, e o MYELIN deixa de inventar o hábito e passa a **aproveitá-lo**. A profundidade prevista vira um sinal de graça para escolher γ e a camada de saída, e explica de onde vem o reflexo.
- **Vizinhos:**
  - as cabeças de indução explicam a cópia;
  - "How Do LLMs Use Their Depth?" mostra o efeito da frequência **estática**;
  - os *deep-thinking tokens* medem a profundidade, mas não a relação com a repetição.
- **Teste que mata (barato, ~1 rodada):**
  1. medir a profundidade de decisão de cada token contra o número de ocorrências anteriores do mesmo 4-grama no contexto (0, 1, 2, 3, 5+);
  2. controlar pela frequência do token no corpus e pela posição;
  3. repetir com repetições **parafraseadas** (a mesma ideia, outras palavras).
- **Morre se:**
  - a queda for menor que ~2 camadas; ou
  - a frequência e a posição explicarem a queda; ou
  - ela acontecer só na 1.ª repetição, sem curva (aí é só a indução, já conhecida).
- **Seria forte se:** houver uma curva monótona com a forma de uma lei de potência, que também apareça, menor, nas paráfrases. Seria um fato novo sobre como os LLMs usam a profundidade.

---

### Apostas fortes (mais perto de virar código)

#### S1: o MYELIN como adaptador sem gradiente de rascunhadores treinados (o antigo H1)

É o primeiro degrau da M1, dentro de uma única sessão. Detalhes abaixo.

#### S2: cópia por paráfrase entre pedidos

**Afirmação.** Buscar continuações nas gerações passadas de **outros pedidos** pelo estado oculto (o reflexo entre pedidos) bate o SuffixDecoding quando os pedidos se repetem com outras palavras: variáveis renomeadas, a mesma pergunta reescrita.
- **Morre se** não houver ganho sobre a árvore de sufixos nos pedidos reescritos.
- **Novidade:** o AdaPLD usa similaridade semântica, mas dentro do contexto; o REST, num corpus fixo.

### Detalhe da S1 (o antigo H1)

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

### De reserva e controles (novidade baixa)

#### H2: calibração por sessão (a versão mínima da S1)

**Hipótese.** Se o que importa é quase todo global (achado 3), um único vetor de viés por sessão, aprendido online com o verificador, já adapta o rascunhador. São 896 números, sem busca nenhuma.

- **Novidade:** baixa como ideia (é uma calibração), mas útil como controle: ela diz se a H1 vale pelo reconhecimento ou só pela calibração.
- **Teste:** dentro do Exp 8, de graça.

#### H3: o reflexo como portão de cópia sem treino

**Hipótese.** A similaridade do reflexo (precisa, mas com pouca cobertura) serve como sinal sem treino de "intenção de copiar", alternando entre o rascunhador neural e a busca de n-gramas.

- **Novidade:** baixa. O SwitchSD faz isso com sondas treinadas e mais precisão; o nosso seria a versão sem treino.
- **Teste:** também dentro do Exp 8 (rascunhador neural + PLD + o portão do reflexo).

#### H4: γ pela profundidade de decisão

**Hipótese.** Prever, pelo tradutor, quando os próximos tokens serão "fundos" e encurtar o rascunho ali.

- **Novidade:** baixa sozinha. O γ adaptativo tem vários trabalhos, e os deep-thinking tokens já medem a profundidade.
- **Se a M4 sobreviver,** a H4 vira a aplicação dela.

## 4. Ordem de ataque

1. **Revisão adversarial de todo o portfólio**, por dois modelos externos (Codex e Antigravity), em paralelo. Para cada hipótese:
   - já existe?
   - o teste mata de verdade?
   - qual confusão estragaria o resultado?
   - o que a faria virar um artigo de primeira linha?
2. **Ciência primeiro, porque é barata e decide o resto:**
   - **Exp 8 = M3 + M4** numa única rodada no Kaggle, com o 0,5B e o tradutor que já existem, e depois um segundo tamanho de modelo;
   - a M3 diz se a memória precisa ser um kNN ou se basta um subespaço;
   - a M4 diz se o hábito já existe dentro do modelo.
3. **Sistemas depois:**
   - S1 → M1: a peça nova é um `DraftModelDrafter`, o rascunhador externo em teacher forcing com o custo dele na conta, e depois a memória persistente entre pedidos;
   - M2 em paralelo: reaproveita o mesmo rascunhador.
4. **Regra de saída:** cada hipótese tem o seu critério de morte escrito **antes** de rodar. O que morrer vira uma linha honesta na história do MYELIN; o que sobreviver vira o artigo.
